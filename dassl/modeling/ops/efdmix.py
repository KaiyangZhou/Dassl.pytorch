import random
from contextlib import contextmanager
import torch
import torch.nn as nn


def deactivate_efdmix(m):
    if type(m) == EFDMix:
        m.set_activation_status(False)


def activate_efdmix(m):
    if type(m) == EFDMix:
        m.set_activation_status(True)


def random_efdmix(m):
    if type(m) == EFDMix:
        m.update_mix_method("random")


def crossdomain_efdmix(m):
    if type(m) == EFDMix:
        m.update_mix_method("crossdomain")


@contextmanager
def run_without_efdmix(model):
    # Assume MixStyle was initially activated
    try:
        model.apply(deactivate_efdmix)
        yield
    finally:
        model.apply(activate_efdmix)


@contextmanager
def run_with_efdmix(model, mix=None):
    # Assume MixStyle was initially deactivated
    if mix == "random":
        model.apply(random_efdmix)

    elif mix == "crossdomain":
        model.apply(crossdomain_efdmix)

    try:
        model.apply(activate_efdmix)
        yield
    finally:
        model.apply(deactivate_efdmix)


class EFDMix(nn.Module):
    """EFDMix.

    Reference:
      Zhang et al. Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization. CVPR 2022.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return (
            f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})"
        )

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B, C, W, H = x.size(0), x.size(1), x.size(2), x.size(3)
        x_view = x.view(B, C, -1)
        value_x, index_x = torch.sort(x_view)  # sort inputs
        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == "random":
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == "crossdomain":
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1)  # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(perm_b.shape[0])]
            perm_a = perm_a[torch.randperm(perm_a.shape[0])]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        inverse_index = index_x.argsort(-1)
        x_view_copy = value_x[perm].gather(-1, inverse_index)
        new_x = x_view + (x_view_copy - x_view.detach()) * (1-lmda)
        return new_x.view(B, C, W, H)
