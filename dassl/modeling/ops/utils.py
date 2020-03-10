import numpy as np
import torch


def sharpen_prob(p, temperature=2):
    """Sharpening probability using temperature."""
    p = p.pow(temperature)
    return p / p.sum(1, keepdim=True)


def reverse_index(tensor, label):
    """Reverse the order of tensor and label tensors."""
    inv_idx = torch.arange(tensor.size(0) - 1, -1, -1).long()
    return tensor[inv_idx], label[inv_idx]


def shuffle_index(tensor, label):
    """Shuffle the order of tensor and label tensors."""
    rnd_idx = torch.randperm(tensor.shape[0])
    return tensor[rnd_idx], label[rnd_idx]


def create_onehot(label, num_classes):
    """Create one-hot tensor."""
    onehot = torch.zeros(label.shape[0], num_classes)
    return onehot.scatter(1, label.unsqueeze(1).data.cpu(), 1)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242."""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current/rampup_length
        return float(np.exp(-5.0 * phase * phase))
