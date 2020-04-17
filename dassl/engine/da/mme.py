import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import ReverseGrad
from dassl.engine.trainer import SimpleNet


class Prototypes(nn.Module):

    def __init__(self, fdim, num_classes, temp=0.05):
        super().__init__()
        self.prototypes = nn.Linear(fdim, num_classes, bias=False)
        self.temp = temp

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        out = self.prototypes(x)
        out = out / self.temp
        return out


@TRAINER_REGISTRY.register()
class MME(TrainerXU):
    """Minimax Entropy.

    https://arxiv.org/abs/1904.06487.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lmda = cfg.TRAINER.MME.LMDA

    def build_model(self):
        cfg = self.cfg

        print('Building F')
        self.F = SimpleNet(cfg, cfg.MODEL, 0)
        self.F.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        self.register_model('F', self.F, self.optim_F, self.sched_F)

        print('Building C')
        self.C = Prototypes(self.F.fdim, self.num_classes)
        self.C.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.OPTIM)
        self.register_model('C', self.C, self.optim_C, self.sched_C)

        self.revgrad = ReverseGrad()

    def forward_backward(self, batch_x, batch_u):
        input_x, label_x, input_u = self.parse_batch_train(batch_x, batch_u)

        feat_x = self.F(input_x)
        logit_x = self.C(feat_x)
        loss_x = F.cross_entropy(logit_x, label_x)
        self.model_backward_and_update(loss_x)

        feat_u = self.F(input_u)
        feat_u = self.revgrad(feat_u)
        logit_u = self.C(feat_u)
        prob_u = F.softmax(logit_u, 1)
        loss_u = -(-prob_u * torch.log(prob_u + 1e-5)).sum(1).mean()
        self.model_backward_and_update(loss_u * self.lmda)

        loss_summary = {
            'loss_x': loss_x.item(),
            'acc_x': compute_accuracy(logit_x, label_x)[0].item(),
            'loss_u': loss_u.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, input):
        return self.C(self.F(input))
