import torch
from torch.nn import functional as F

from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.modeling import build_network
from dassl.engine.trainer import SimpleNet


@TRAINER_REGISTRY.register()
class DDAIG(TrainerX):
    """Deep Domain-Adversarial Image Generation.

    https://arxiv.org/abs/2003.06054.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lmda = cfg.TRAINER.DDAIG.LMDA
        self.clamp = cfg.TRAINER.DDAIG.CLAMP
        self.clamp_min = cfg.TRAINER.DDAIG.CLAMP_MIN
        self.clamp_max = cfg.TRAINER.DDAIG.CLAMP_MAX
        self.warmup = cfg.TRAINER.DDAIG.WARMUP
        self.alpha = cfg.TRAINER.DDAIG.ALPHA

    def build_model(self):
        cfg = self.cfg

        print('Building F')
        self.F = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        self.F.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        self.register_model('F', self.F, self.optim_F, self.sched_F)

        print('Building D')
        self.D = SimpleNet(cfg, cfg.MODEL, self.dm.num_source_domains)
        self.D.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.D)))
        self.optim_D = build_optimizer(self.D, cfg.OPTIM)
        self.sched_D = build_lr_scheduler(self.optim_D, cfg.OPTIM)
        self.register_model('D', self.D, self.optim_D, self.sched_D)

        print('Building G')
        self.G = build_network(cfg.TRAINER.DDAIG.G_ARCH, verbose=cfg.VERBOSE)
        self.G.to(self.device)
        print('# params: {:,}'.format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model('G', self.G, self.optim_G, self.sched_G)

    def forward_backward(self, batch):
        input, label, domain = self.parse_batch_train(batch)

        #############
        # Update G
        #############
        input_p = self.G(input, lmda=self.lmda)
        if self.clamp:
            input_p = torch.clamp(
                input_p, min=self.clamp_min, max=self.clamp_max
            )
        loss_g = 0
        # Minimize label loss
        loss_g += F.cross_entropy(self.F(input_p), label)
        # Maximize domain loss
        loss_g -= F.cross_entropy(self.D(input_p), domain)
        self.model_backward_and_update(loss_g, 'G')

        # Perturb data with new G
        with torch.no_grad():
            input_p = self.G(input, lmda=self.lmda)
            if self.clamp:
                input_p = torch.clamp(
                    input_p, min=self.clamp_min, max=self.clamp_max
                )

        #############
        # Update F
        #############
        loss_f = F.cross_entropy(self.F(input), label)
        if (self.epoch + 1) > self.warmup:
            loss_fp = F.cross_entropy(self.F(input_p), label)
            loss_f = (1. - self.alpha) * loss_f + self.alpha * loss_fp
        self.model_backward_and_update(loss_f, 'F')

        #############
        # Update D
        #############
        loss_d = F.cross_entropy(self.D(input), domain)
        self.model_backward_and_update(loss_d, 'D')

        loss_summary = {
            'loss_g': loss_g.item(),
            'loss_f': loss_f.item(),
            'loss_d': loss_d.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, input):
        return self.F(input)
