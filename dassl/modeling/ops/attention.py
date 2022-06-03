import torch.nn as nn
from torch.nn import functional as F

__all__ = ["Attention"]


class Attention(nn.Module):
    """Attention from `"Dynamic Domain Generalization" <https://github.com/MetaVisionLab/DDG>`_.
    """

    def __init__(
        self,
        in_channels: int,
        out_features: int,
        squeeze=None,
        bias: bool = True
    ):
        super(Attention, self).__init__()
        self.squeeze = squeeze if squeeze else in_channels // 16
        assert self.squeeze > 0
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, self.squeeze, bias=bias)
        self.fc2 = nn.Linear(self.squeeze, out_features, bias=bias)
        self.sf = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.avg_pool(x).view(x.shape[:-2])
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return self.sf(x)
