import os.path as osp
import pandas as pd

from dassl.data.datasets import DATASET_REGISTRY

from .wilds_base import WILDSBase


@DATASET_REGISTRY.register()
class IWildCam(WILDSBase):
    """Animal species recognition.

    182 classes (species).

    Reference:
        - Beery et al. "The iwildcam 2021 competition dataset." arXiv 2021.
        - Koh et al. "Wilds: A benchmark of in-the-wild distribution shifts." ICML 2021.
    """

    dataset_dir = "iwildcam_v2.0"

    def __init__(self, cfg):
        super().__init__(cfg)

    def get_image_path(self, dataset, idx):
        image_name = dataset._input_array[idx]
        image_path = osp.join(self.dataset_dir, "train", image_name)
        return image_path

    def load_classnames(self):
        df = pd.read_csv(osp.join(self.dataset_dir, "categories.csv"))
        return dict(df["name"])
