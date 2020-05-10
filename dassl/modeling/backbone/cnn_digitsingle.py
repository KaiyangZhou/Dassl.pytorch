"""
This model is built based on
https://github.com/ricvolpi/generalize-unseen-domains/blob/master/model.py
"""
import torch.nn as nn
from torch.nn import functional as F

from dassl.utils import init_network_weights

from .build import BACKBONE_REGISTRY
from .backbone import Backbone


class CNN(Backbone):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc3 = nn.Linear(5 * 5 * 128, 1024)
        self.fc4 = nn.Linear(1024, 1024)

        self._out_features = 1024

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 32 and W == 32, \
            'Input to network must be 32x32, ' \
            'but got {}x{}'.format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.relu(x)

        return x


@BACKBONE_REGISTRY.register()
def cnn_digitsingle(**kwargs):
    model = CNN()
    init_network_weights(model, init_type='kaiming')
    return model
