import numpy as np
import torch


def sharpen_prob(p, temperature=2):
    """Sharpening probability with a temperature.

    Args:
        p (torch.Tensor): probability matrix (batch_size, n_classes)
        temperature (float): temperature.
    """
    p = p.pow(temperature)
    return p / p.sum(1, keepdim=True)


def reverse_index(data, label):
    """Reverse order."""
    inv_idx = torch.arange(data.size(0) - 1, -1, -1).long()
    return data[inv_idx], label[inv_idx]


def shuffle_index(data, label):
    """Shuffle order."""
    rnd_idx = torch.randperm(data.shape[0])
    return data[rnd_idx], label[rnd_idx]


def create_onehot(label, num_classes):
    """Create one-hot tensor.

    Args:
        label (torch.Tensor): 1-D tensor.
        num_classes (int): number of classes.
    """
    onehot = torch.zeros(label.shape[0], num_classes)
    return onehot.scatter(1, label.unsqueeze(1).data.cpu(), 1)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup.

    Args:
        current (int): current step.
        rampup_length (int): maximum step.
    """
    assert rampup_length > 0
    current = np.clip(current, 0., rampup_length)
    phase = 1. - current/rampup_length
    return float(np.exp(-5. * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup.

    Args:
        current (int): current step.
        rampup_length (int): maximum step.
    """
    assert rampup_length > 0
    ratio = np.clip(current / rampup_length, 0., 1.)
    return float(ratio)


def ema_model_update(model, ema_model, alpha):
    """Exponential moving average of model parameters.

    Args:
        model (nn.Module): model being trained.
        ema_model (nn.Module): ema of the model.
        alpha (float): ema decay rate.
    """
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
