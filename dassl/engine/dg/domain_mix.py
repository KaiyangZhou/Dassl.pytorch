import torch
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy

__all__ = ["DomainMix"]


@TRAINER_REGISTRY.register()
class DomainMix(TrainerX):
    """DomainMix.
    
    Dynamic Domain Generalization.

    https://github.com/MetaVisionLab/DDG
    """

    def __init__(self, cfg):
        super(DomainMix, self).__init__(cfg)
        self.mix_type = cfg.TRAINER.DOMAINMIX.TYPE
        self.alpha = cfg.TRAINER.DOMAINMIX.ALPHA
        self.beta = cfg.TRAINER.DOMAINMIX.BETA
        self.dist_beta = torch.distributions.Beta(self.alpha, self.beta)

    def forward_backward(self, batch):
        images, label_a, label_b, lam = self.parse_batch_train(batch)
        output = self.model(images)
        loss = lam * F.cross_entropy(
            output, label_a
        ) + (1-lam) * F.cross_entropy(output, label_b)
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label_a)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        images = batch["img"]
        target = batch["label"]
        domain = batch["domain"]
        images = images.to(self.device)
        target = target.to(self.device)
        domain = domain.to(self.device)
        images, target_a, target_b, lam = self.domain_mix(
            images, target, domain
        )
        return images, target_a, target_b, lam

    def domain_mix(self, x, target, domain):
        lam = (
            self.dist_beta.rsample((1, ))
            if self.alpha > 0 else torch.tensor(1)
        ).to(x.device)

        # random shuffle
        perm = torch.randperm(x.size(0), dtype=torch.int64, device=x.device)
        if self.mix_type == "crossdomain":
            domain_list = torch.unique(domain)
            if len(domain_list) > 1:
                for idx in domain_list:
                    cnt_a = torch.sum(domain == idx)
                    idx_b = (domain != idx).nonzero().squeeze(-1)
                    cnt_b = idx_b.shape[0]
                    perm_b = torch.ones(cnt_b).multinomial(
                        num_samples=cnt_a, replacement=bool(cnt_a > cnt_b)
                    )
                    perm[domain == idx] = idx_b[perm_b]
        elif self.mix_type != "random":
            raise NotImplementedError(
                f"Chooses {'random', 'crossdomain'}, but got {self.mix_type}."
            )
        mixed_x = lam*x + (1-lam) * x[perm, :]
        target_a, target_b = target, target[perm]
        return mixed_x, target_a, target_b, lam
