import torch.nn as nn

from .build import BACKBONE_REGISTRY
from .backbone import Backbone


class Convolution(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.conv(x))


class ConvNet(Backbone):

    def __init__(self, c_in=3, c_hidden=64, nb=4):
        super().__init__()
        backbone = []
        backbone += [Convolution(c_in, c_hidden)]
        backbone += [nn.MaxPool2d(2)]
        for i in range(nb - 1):
            backbone += [Convolution(c_hidden, c_hidden)]
            backbone += [nn.MaxPool2d(2)]
        self.backbone = nn.Sequential(*backbone)

        self._out_features = 2**2 * c_hidden

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 32 and W == 32, \
            'Input to network must be 32x32, ' \
            'but got {}x{}'.format(H, W)

    def forward(self, x):
        self._check_input(x)
        f = self.backbone(x)
        return f.view(f.size(0), -1)


@BACKBONE_REGISTRY.register()
def cnn_digitsdg(**kwargs):
    """
    This architecture was used for DigitsDG dataset in:

        - Zhou et al. Deep Domain-Adversarial Image Generation
        for Domain Generalisation. AAAI 2020.
    """
    return ConvNet(c_hidden=64, nb=4)
