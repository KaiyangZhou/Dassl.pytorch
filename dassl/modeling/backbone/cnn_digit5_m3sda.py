"""
Reference

https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/tree/master/M3SDA
"""
import torch.nn as nn
from torch.nn import functional as F

from .build import BACKBONE_REGISTRY
from .backbone import Backbone


class FeatureExtractor(Backbone):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)

        self._out_features = 2048

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 32 and W == 32, \
            'Input to network must be 32x32, ' \
            'but got {}x{}'.format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 8192)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x


@BACKBONE_REGISTRY.register()
def cnn_digit5_m3sda(**kwargs):
    """
    This architecture was used for the Digit-5 dataset in:

        - Peng et al. Moment Matching for Multi-Source
        Domain Adaptation. ICCV 2019.
    """
    return FeatureExtractor()
