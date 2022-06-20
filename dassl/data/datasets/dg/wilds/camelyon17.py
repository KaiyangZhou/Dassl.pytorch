from dassl.data.datasets import DATASET_REGISTRY

from .wilds_base import WILDSBase


@DATASET_REGISTRY.register()
class Camelyon17(WILDSBase):
    """Tumor tissue recognition.

    2 classes (whether a given region of tissue contains tumor tissue).

    Reference:
        - Bandi et al. "From detection of individual metastases to classification of lymph
        node status at the patient level: the CAMELYON17 challenge." TMI 2021.
        - Koh et al. "Wilds: A benchmark of in-the-wild distribution shifts." ICML 2021.
    """

    dataset_dir = "camelyon17_v1.0"

    def __init__(self, cfg):
        super().__init__(cfg)

    def load_classnames(self):
        return {0: "healthy tissue", 1: "tumor tissue"}
