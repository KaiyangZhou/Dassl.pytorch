import copy
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.modeling.ops.utils import sigmoid_rampup


@TRAINER_REGISTRY.register()
class SelfEnsembling(TrainerXU):
    """Self-ensembling for visual domain adaptation.

    https://arxiv.org/abs/1706.05208.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.teacher = copy.deepcopy(self.model)
        self.teacher.train()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        self.ema_alpha = cfg.TRAINER.SE.EMA_ALPHA
        self.conf_thre = cfg.TRAINER.SE.CONF_THRE
        self.rampup_length = cfg.TRAINER.SE.RAMPUP_LENGTH

    def check_cfg(self, cfg):
        assert cfg.DATALOADER.K_TRANSFORMS == 2

    def forward_backward(self, batch_x, batch_u):
        global_step = self.batch_idx + self.epoch * self.num_batches
        parsed = self.parse_batch_train(batch_x, batch_u)
        input_x, label_x, input_u1, input_u2 = parsed

        logit_x = self.model(input_x)
        loss_x = F.cross_entropy(logit_x, label_x)

        prob_u = F.softmax(self.model(input_u1), 1)
        t_prob_u = F.softmax(self.teacher(input_u2), 1)
        loss_u = ((prob_u - t_prob_u)**2).sum(1)

        if self.conf_thre:
            max_prob = t_prob_u.max(1)[0]
            mask = (max_prob > self.conf_thre).float()
            loss_u = (loss_u * mask).mean()
        else:
            weight_u = sigmoid_rampup(global_step, self.rampup_length)
            loss_u = loss_u.mean() * weight_u

        loss = loss_x + loss_u
        self.model_backward_and_update(loss)

        # Update teacher
        alpha = min(1 - 1 / (global_step+1), self.ema_alpha)
        for tp, p in zip(self.teacher.parameters(), self.model.parameters()):
            tp.data.mul_(alpha).add_(1 - alpha, p.data)

        output_dict = {
            'loss_x': loss_x.item(),
            'acc_x': compute_accuracy(logit_x.detach(), label_x)[0].item(),
            'loss_u': loss_u.item(),
            'lr': self.optim.param_groups[0]['lr']
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return output_dict

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x['img'][0]
        label_x = batch_x['label']
        input_u = batch_u['img']
        input_u1, input_u2 = input_u

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u1 = input_u1.to(self.device)
        input_u2 = input_u2.to(self.device)

        return input_x, label_x, input_u1, input_u2
