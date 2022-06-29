import numpy as np
from functools import partial
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR

from dassl.data import DataManager
from dassl.optim import build_optimizer
from dassl.utils import count_num_param
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import ReverseGrad
from dassl.engine.trainer import SimpleNet
from dassl.data.transforms.transforms import build_transform


def custom_scheduler(iter, max_iter=None, alpha=10, beta=0.75, init_lr=0.001):
    """Custom LR Annealing

    https://arxiv.org/pdf/1409.7495.pdf
    """
    if max_iter is None:
        return init_lr
    return (1 + float(iter / max_iter) * alpha)**(-1.0 * beta)


class AAC(nn.Module):

    def forward(self, sim_mat, prob_u, prob_us):

        P = prob_u.matmul(prob_us.t())

        loss = -(
            sim_mat * torch.log(P + 1e-7) +
            (1.-sim_mat) * torch.log(1. - P + 1e-7)
        )
        return loss.mean()


class Prototypes(nn.Module):

    def __init__(self, fdim, num_classes, temp=0.05):
        super().__init__()
        self.prototypes = nn.Linear(fdim, num_classes, bias=False)
        self.temp = temp
        self.revgrad = ReverseGrad()

    def forward(self, x, reverse=False):
        if reverse:
            x = self.revgrad(x)
        x = F.normalize(x, p=2, dim=1)
        out = self.prototypes(x)
        out = out / self.temp
        return out


@TRAINER_REGISTRY.register()
class CDAC(TrainerXU):
    """Cross Domain Adaptive Clustering.

    https://arxiv.org/pdf/2104.09415.pdf
    """

    def __init__(self, cfg):
        self.rampup_coef = cfg.TRAINER.CDAC.RAMPUP_COEF
        self.rampup_iters = cfg.TRAINER.CDAC.RAMPUP_ITRS
        self.lr_multi = cfg.TRAINER.CDAC.CLASS_LR_MULTI
        self.topk = cfg.TRAINER.CDAC.TOPK_MATCH
        self.p_thresh = cfg.TRAINER.CDAC.P_THRESH
        self.aac_criterion = AAC()
        super().__init__(cfg)

    def check_cfg(self, cfg):
        assert len(
            cfg.TRAINER.CDAC.STRONG_TRANSFORMS
        ) > 0, "Strong augmentations are necessary to run CDAC"
        assert cfg.DATALOADER.K_TRANSFORMS == 2, "CDAC needs two strong augmentations of the same image."

    def build_data_loader(self):

        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.CDAC.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        self.dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = self.dm.train_loader_x
        self.train_loader_u = self.dm.train_loader_u
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader
        self.num_classes = self.dm.num_classes
        self.lab2cname = self.dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        # Custom LR Scheduler for CDAC
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len(self.train_loader_x)
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len(self.len_train_loader_u)
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(
                len(self.train_loader_x), len(self.train_loader_u)
            )
        self.max_iter = self.max_epoch * self.num_batches
        print("Max Iterations: %d" % self.max_iter)

        print("Building F")
        self.F = SimpleNet(cfg, cfg.MODEL, 0)
        self.F.to(self.device)
        print("# params: {:,}".format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        custom_lr_F = partial(
            custom_scheduler, max_iter=self.max_iter, init_lr=cfg.OPTIM.LR
        )
        self.sched_F = LambdaLR(self.optim_F, custom_lr_F)
        self.register_model("F", self.F, self.optim_F, self.sched_F)

        print("Building C")
        self.C = Prototypes(self.F.fdim, self.num_classes)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.OPTIM)

        # Multiply the learning rate of C by lr_multi
        for group_param in self.optim_C.param_groups:
            group_param['lr'] *= self.lr_multi
        custom_lr_C = partial(
            custom_scheduler,
            max_iter=self.max_iter,
            init_lr=cfg.OPTIM.LR * self.lr_multi
        )
        self.sched_C = LambdaLR(self.optim_C, custom_lr_C)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {
            "acc_thre": acc_thre,
            "acc_raw": acc_raw,
            "keep_rate": keep_rate
        }
        return output

    def forward_backward(self, batch_x, batch_u):

        current_itr = self.epoch * self.num_batches + self.batch_idx

        input_x, label_x, input_u, input_us, input_us2, label_u = self.parse_batch_train(
            batch_x, batch_u
        )

        # Paper Reference Eq. 2 - Supervised Loss

        feat_x = self.F(input_x)
        logit_x = self.C(feat_x)
        loss_x = F.cross_entropy(logit_x, label_x)

        self.model_backward_and_update(loss_x)

        feat_u = self.F(input_u)
        feat_us = self.F(input_us)
        feat_us2 = self.F(input_us2)

        # Paper Reference Eq.3 - Adversarial Adaptive Loss
        logit_u = self.C(feat_u, reverse=True)
        logit_us = self.C(feat_us, reverse=True)
        prob_u, prob_us = F.softmax(logit_u, dim=1), F.softmax(logit_us, dim=1)

        # Get similarity matrix s_ij
        sim_mat = self.get_similarity_matrix(feat_u, self.topk, self.device)

        aac_loss = (-1. * self.aac_criterion(sim_mat, prob_u, prob_us))

        # Paper Reference Eq. 4 - Pseudo label Loss
        logit_u = self.C(feat_u)
        logit_us = self.C(feat_us)
        logit_us2 = self.C(feat_us2)
        prob_u, prob_us, prob_us2 = F.softmax(
            logit_u, dim=1
        ), F.softmax(
            logit_us, dim=1
        ), F.softmax(
            logit_us2, dim=1
        )
        prob_u = prob_u.detach()
        max_probs, max_idx = torch.max(prob_u, dim=-1)
        mask = max_probs.ge(self.p_thresh).float()
        p_u_stats = self.assess_y_pred_quality(max_idx, label_u, mask)

        pl_loss = (
            F.cross_entropy(logit_us2, max_idx, reduction='none') * mask
        ).mean()

        # Paper Reference Eq. 8 - Consistency Loss
        cons_multi = self.sigmoid_rampup(
            current_itr=current_itr, rampup_itr=self.rampup_iters
        ) * self.rampup_coef
        cons_loss = cons_multi * F.mse_loss(prob_us, prob_us2)

        loss_u = aac_loss + pl_loss + cons_loss

        self.model_backward_and_update(loss_u)

        loss_summary = {
            "loss_x": loss_x.item(),
            "acc_x": compute_accuracy(logit_x, label_x)[0].item(),
            "loss_u": loss_u.item(),
            "aac_loss": aac_loss.item(),
            "pl_loss": pl_loss.item(),
            "cons_loss": cons_loss.item(),
            "p_u_pred_acc": p_u_stats["acc_raw"],
            "p_u_pred_acc_thre": p_u_stats["acc_thre"],
            "p_u_pred_keep": p_u_stats["keep_rate"]
        }

        # Update LR after every iteration as mentioned in the paper

        self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):

        input_x = batch_x["img"][0]
        label_x = batch_x["label"]

        input_u = batch_u["img"][0]
        input_us = batch_u["img2"][0]
        input_us2 = batch_u["img2"][1]
        label_u = batch_u["label"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)

        input_u = input_u.to(self.device)
        input_us = input_us.to(self.device)
        input_us2 = input_us2.to(self.device)
        label_u = label_u.to(self.device)

        return input_x, label_x, input_u, input_us, input_us2, label_u

    def model_inference(self, input):
        return self.C(self.F(input))

    @staticmethod
    def get_similarity_matrix(feat, topk, device):

        feat_d = feat.detach()

        feat_d = torch.sort(
            torch.argsort(feat_d, dim=1, descending=True)[:, :topk], dim=1
        )[0]
        sim_mat = torch.zeros((feat_d.shape[0], feat_d.shape[0])).to(device)
        for row in range(feat_d.shape[0]):
            sim_mat[row, torch.all(feat_d == feat_d[row, :], dim=1)] = 1
        return sim_mat

    @staticmethod
    def sigmoid_rampup(current_itr, rampup_itr):
        """Exponential Rampup
        https://arxiv.org/abs/1610.02242
        """
        if rampup_itr == 0:
            return 1.0
        else:
            var = np.clip(current_itr, 0.0, rampup_itr)
            phase = 1.0 - var/rampup_itr
            return float(np.exp(-5.0 * phase * phase))
