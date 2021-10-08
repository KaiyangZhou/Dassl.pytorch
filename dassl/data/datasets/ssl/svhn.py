from .cifar import CIFAR10
from ..build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SVHN(CIFAR10):
    """SVHN for SSL.

    Reference:
        - Netzer et al. Reading Digits in Natural Images with
        Unsupervised Feature Learning. NIPS-W 2011.
    """

    dataset_dir = "svhn"

    def __init__(self, cfg):
        super().__init__(cfg)
