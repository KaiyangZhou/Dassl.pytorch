import torch

from dassl.engine import TRAINER_REGISTRY, TrainerXU


@TRAINER_REGISTRY.register()
class AdaBN(TrainerXU):
    """Adaptive Batch Normalization.

    https://arxiv.org/abs/1603.04779.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.reset_bn_stats = False

    def check_cfg(self, cfg):
        assert cfg.MODEL.INIT_WEIGHTS

    def before_epoch(self):
        if not self.reset_bn_stats:
            for m in self.model.modules():
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.reset_running_stats()

            self.reset_bn_stats = True

    def forward_backward(self, batch_x, batch_u):
        input_u = batch_u['img'].to(self.device)

        with torch.no_grad():
            self.model(input_u)

        output_dict = {'dummy_var': 1}

        return output_dict
