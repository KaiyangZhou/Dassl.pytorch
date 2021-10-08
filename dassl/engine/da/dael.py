import torch
import torch.nn as nn

from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.engine.trainer import SimpleNet
from dassl.data.transforms import build_transform
from dassl.modeling.ops.utils import create_onehot


class Experts(nn.Module):

    def __init__(self, n_source, fdim, num_classes):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(fdim, num_classes) for _ in range(n_source)]
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, i, x):
        x = self.linears[i](x)
        x = self.softmax(x)
        return x


@TRAINER_REGISTRY.register()
class DAEL(TrainerXU):
    """Domain Adaptive Ensemble Learning.

    https://arxiv.org/abs/2003.07325.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        n_domain = cfg.DATALOADER.TRAIN_X.N_DOMAIN
        batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        if n_domain <= 0:
            n_domain = self.num_source_domains
        self.split_batch = batch_size // n_domain
        self.n_domain = n_domain

        self.weight_u = cfg.TRAINER.DAEL.WEIGHT_U
        self.conf_thre = cfg.TRAINER.DAEL.CONF_THRE

    def check_cfg(self, cfg):
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "RandomDomainSampler"
        assert not cfg.DATALOADER.TRAIN_U.SAME_AS_X
        assert len(cfg.TRAINER.DAEL.STRONG_TRANSFORMS) > 0

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.DAEL.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building F")
        self.F = SimpleNet(cfg, cfg.MODEL, 0)
        self.F.to(self.device)
        print("# params: {:,}".format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        self.register_model("F", self.F, self.optim_F, self.sched_F)
        fdim = self.F.fdim

        print("Building E")
        self.E = Experts(self.num_source_domains, fdim, self.num_classes)
        self.E.to(self.device)
        print("# params: {:,}".format(count_num_param(self.E)))
        self.optim_E = build_optimizer(self.E, cfg.OPTIM)
        self.sched_E = build_lr_scheduler(self.optim_E, cfg.OPTIM)
        self.register_model("E", self.E, self.optim_E, self.sched_E)

    def forward_backward(self, batch_x, batch_u):
        parsed_data = self.parse_batch_train(batch_x, batch_u)
        input_x, input_x2, label_x, domain_x, input_u, input_u2 = parsed_data

        input_x = torch.split(input_x, self.split_batch, 0)
        input_x2 = torch.split(input_x2, self.split_batch, 0)
        label_x = torch.split(label_x, self.split_batch, 0)
        domain_x = torch.split(domain_x, self.split_batch, 0)
        domain_x = [d[0].item() for d in domain_x]

        # Generate pseudo label
        with torch.no_grad():
            feat_u = self.F(input_u)
            pred_u = []
            for k in range(self.num_source_domains):
                pred_uk = self.E(k, feat_u)
                pred_uk = pred_uk.unsqueeze(1)
                pred_u.append(pred_uk)
            pred_u = torch.cat(pred_u, 1)  # (B, K, C)
            # Get the highest probability and index (label) for each expert
            experts_max_p, experts_max_idx = pred_u.max(2)  # (B, K)
            # Get the most confident expert
            max_expert_p, max_expert_idx = experts_max_p.max(1)  # (B)
            pseudo_label_u = []
            for i, experts_label in zip(max_expert_idx, experts_max_idx):
                pseudo_label_u.append(experts_label[i])
            pseudo_label_u = torch.stack(pseudo_label_u, 0)
            pseudo_label_u = create_onehot(pseudo_label_u, self.num_classes)
            pseudo_label_u = pseudo_label_u.to(self.device)
            label_u_mask = (max_expert_p >= self.conf_thre).float()

        loss_x = 0
        loss_cr = 0
        acc_x = 0

        feat_x = [self.F(x) for x in input_x]
        feat_x2 = [self.F(x) for x in input_x2]
        feat_u2 = self.F(input_u2)

        for feat_xi, feat_x2i, label_xi, i in zip(
            feat_x, feat_x2, label_x, domain_x
        ):
            cr_s = [j for j in domain_x if j != i]

            # Learning expert
            pred_xi = self.E(i, feat_xi)
            loss_x += (-label_xi * torch.log(pred_xi + 1e-5)).sum(1).mean()
            expert_label_xi = pred_xi.detach()
            acc_x += compute_accuracy(pred_xi.detach(),
                                      label_xi.max(1)[1])[0].item()

            # Consistency regularization
            cr_pred = []
            for j in cr_s:
                pred_j = self.E(j, feat_x2i)
                pred_j = pred_j.unsqueeze(1)
                cr_pred.append(pred_j)
            cr_pred = torch.cat(cr_pred, 1)
            cr_pred = cr_pred.mean(1)
            loss_cr += ((cr_pred - expert_label_xi)**2).sum(1).mean()

        loss_x /= self.n_domain
        loss_cr /= self.n_domain
        acc_x /= self.n_domain

        # Unsupervised loss
        pred_u = []
        for k in range(self.num_source_domains):
            pred_uk = self.E(k, feat_u2)
            pred_uk = pred_uk.unsqueeze(1)
            pred_u.append(pred_uk)
        pred_u = torch.cat(pred_u, 1)
        pred_u = pred_u.mean(1)
        l_u = (-pseudo_label_u * torch.log(pred_u + 1e-5)).sum(1)
        loss_u = (l_u * label_u_mask).mean()

        loss = 0
        loss += loss_x
        loss += loss_cr
        loss += loss_u * self.weight_u
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss_x": loss_x.item(),
            "acc_x": acc_x,
            "loss_cr": loss_cr.item(),
            "loss_u": loss_u.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        input_x2 = batch_x["img2"]
        label_x = batch_x["label"]
        domain_x = batch_x["domain"]
        input_u = batch_u["img"]
        input_u2 = batch_u["img2"]

        label_x = create_onehot(label_x, self.num_classes)

        input_x = input_x.to(self.device)
        input_x2 = input_x2.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)
        input_u2 = input_u2.to(self.device)

        return input_x, input_x2, label_x, domain_x, input_u, input_u2

    def model_inference(self, input):
        f = self.F(input)
        p = []
        for k in range(self.num_source_domains):
            p_k = self.E(k, f)
            p_k = p_k.unsqueeze(1)
            p.append(p_k)
        p = torch.cat(p, 1)
        p = p.mean(1)
        return p
