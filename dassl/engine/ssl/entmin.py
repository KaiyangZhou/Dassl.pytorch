import torch
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy


@TRAINER_REGISTRY.register()
class EntMin(TrainerXU):
    """Entropy Minimization.

    http://papers.nips.cc/paper/2740-semi-supervised-learning-by-entropy-minimization.pdf.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lmda = cfg.TRAINER.ENTMIN.LMDA

    def forward_backward(self, batch_x, batch_u):
        input_x, label_x, input_u = self.parse_batch_train(batch_x, batch_u)

        output_x = self.model(input_x)
        loss_x = F.cross_entropy(output_x, label_x)

        output_u = F.softmax(self.model(input_u), 1)
        loss_u = (-output_u * torch.log(output_u + 1e-5)).sum(1).mean()

        loss = loss_x + loss_u * self.lmda

        self.model_backward_and_update(loss)

        loss_summary = {
            'loss_x': loss_x.item(),
            'acc_x': compute_accuracy(output_x, label_x)[0].item(),
            'loss_u': loss_u.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
