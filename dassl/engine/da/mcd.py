import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.engine.trainer import SimpleNet


@TRAINER_REGISTRY.register()
class MCD(TrainerXU):
    """Maximum Classifier Discrepancy.

    https://arxiv.org/abs/1712.02560.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.n_step_F = cfg.TRAINER.MCD.N_STEP_F

    def build_model(self):
        cfg = self.cfg

        print('Building F')
        self.F = SimpleNet(cfg, cfg.MODEL, 0)
        self.F.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        self.register_model('F', self.F, self.optim_F, self.sched_F)
        fdim = self.F.fdim

        print('Building C1')
        self.C1 = nn.Linear(fdim, self.num_classes)
        self.C1.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.C1)))
        self.optim_C1 = build_optimizer(self.C1, cfg.OPTIM)
        self.sched_C1 = build_lr_scheduler(self.optim_C1, cfg.OPTIM)
        self.register_model('C1', self.C1, self.optim_C1, self.sched_C1)

        print('Building C2')
        self.C2 = nn.Linear(fdim, self.num_classes)
        self.C2.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.C2)))
        self.optim_C2 = build_optimizer(self.C2, cfg.OPTIM)
        self.sched_C2 = build_lr_scheduler(self.optim_C2, cfg.OPTIM)
        self.register_model('C2', self.C2, self.optim_C2, self.sched_C2)

    def forward_backward(self, batch_x, batch_u):
        parsed = self.parse_batch_train(batch_x, batch_u)
        input_x, label_x, input_u = parsed

        # Step A
        feat_x = self.F(input_x)
        logit_x1 = self.C1(feat_x)
        logit_x2 = self.C2(feat_x)
        loss_x1 = F.cross_entropy(logit_x1, label_x)
        loss_x2 = F.cross_entropy(logit_x2, label_x)
        loss_step_A = loss_x1 + loss_x2
        self.model_backward_and_update(loss_step_A)

        # Step B
        with torch.no_grad():
            feat_x = self.F(input_x)
        logit_x1 = self.C1(feat_x)
        logit_x2 = self.C2(feat_x)
        loss_x1 = F.cross_entropy(logit_x1, label_x)
        loss_x2 = F.cross_entropy(logit_x2, label_x)
        loss_x = loss_x1 + loss_x2

        with torch.no_grad():
            feat_u = self.F(input_u)
        pred_u1 = F.softmax(self.C1(feat_u), 1)
        pred_u2 = F.softmax(self.C2(feat_u), 1)
        loss_dis = self.discrepancy(pred_u1, pred_u2)

        loss_step_B = loss_x - loss_dis
        self.model_backward_and_update(loss_step_B, ['C1', 'C2'])

        # Step C
        for _ in range(self.n_step_F):
            feat_u = self.F(input_u)
            pred_u1 = F.softmax(self.C1(feat_u), 1)
            pred_u2 = F.softmax(self.C2(feat_u), 1)
            loss_step_C = self.discrepancy(pred_u1, pred_u2)
            self.model_backward_and_update(loss_step_C, 'F')

        loss_summary = {
            'loss_step_A': loss_step_A.item(),
            'loss_step_B': loss_step_B.item(),
            'loss_step_C': loss_step_C.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def discrepancy(self, y1, y2):
        return (y1 - y2).abs().mean()

    def model_inference(self, input):
        feat = self.F(input)
        return self.C1(feat)
