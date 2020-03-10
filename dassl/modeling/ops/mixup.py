import torch


def mixup(x1, x2, y1, y2, beta, preserve_order=False):
    """Mixup.

    Args:
        x1 (torch.Tensor): data with shape of (batch_size, n_channels, height, width).
        x2 (torch.Tensor): data with shape of (batch_size, n_channels, height, width).
        y1 (torch.Tensor): label with shape of (batch_size, n_classes).
        y2 (torch.Tensor): label with shape of (batch_size, n_classes).
        beta (float): hyper-parameter for Beta sampling.
        preserve_order (bool): whether to apply lmda = max(lmda, 1-lmda).
            Default is False.
    """
    lmda = torch.distributions.Beta(beta, beta).sample([x1.shape[0], 1, 1, 1])
    if preserve_order:
        lmda = torch.max(lmda, 1 - lmda)
    lmda = lmda.to(x1.device)
    xmix = x1*lmda + x2 * (1-lmda)
    lmda = lmda[:, :, 0, 0]
    ymix = y1*lmda + y2 * (1-lmda)
    return xmix, ymix
