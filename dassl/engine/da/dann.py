import numpy as np
import torch
import torch.nn as nn

from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.modeling import build_head
from dassl.modeling.ops import ReverseGrad


@TRAINER_REGISTRY.register()
class DANN(TrainerXU):
    """Domain-Adversarial Neural Networks.

    https://arxiv.org/abs/1505.07818.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.build_critic()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def build_critic(self):
        cfg = self.cfg

        print("Building critic network")
        fdim = self.model.fdim
        critic_body = build_head(
            "mlp",
            verbose=cfg.VERBOSE,
            in_features=fdim,
            hidden_layers=[fdim, fdim],
            activation="leaky_relu",
        )
        self.critic = nn.Sequential(critic_body, nn.Linear(fdim, 1))
        print("# params: {:,}".format(count_num_param(self.critic)))
        self.critic.to(self.device)
        self.optim_c = build_optimizer(self.critic, cfg.OPTIM)
        self.sched_c = build_lr_scheduler(self.optim_c, cfg.OPTIM)
        self.register_model("critic", self.critic, self.optim_c, self.sched_c)
        self.revgrad = ReverseGrad()

    def forward_backward(self, batch_x, batch_u):
        input_x, label_x, input_u = self.parse_batch_train(batch_x, batch_u)
        domain_x = torch.ones(input_x.shape[0], 1).to(self.device)
        domain_u = torch.zeros(input_u.shape[0], 1).to(self.device)

        global_step = self.batch_idx + self.epoch * self.num_batches
        progress = global_step / (self.max_epoch * self.num_batches)
        lmda = 2 / (1 + np.exp(-10 * progress)) - 1

        logit_x, feat_x = self.model(input_x, return_feature=True)
        _, feat_u = self.model(input_u, return_feature=True)

        loss_x = self.ce(logit_x, label_x)

        feat_x = self.revgrad(feat_x, grad_scaling=lmda)
        feat_u = self.revgrad(feat_u, grad_scaling=lmda)
        output_xd = self.critic(feat_x)
        output_ud = self.critic(feat_u)
        loss_d = self.bce(output_xd, domain_x) + self.bce(output_ud, domain_u)

        loss = loss_x + loss_d
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss_x": loss_x.item(),
            "acc_x": compute_accuracy(logit_x, label_x)[0].item(),
            "loss_d": loss_d.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
