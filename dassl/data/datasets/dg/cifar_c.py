import os.path as osp

from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase

AVAI_C_TYPES = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur",
]


@DATASET_REGISTRY.register()
class CIFAR10C(DatasetBase):
    """CIFAR-10 -> CIFAR-10-C.

    Dataset link: https://zenodo.org/record/2535967#.YFwtV2Qzb0o

    Statistics:
        - 2 domains: the normal CIFAR-10 vs. a corrupted CIFAR-10
        - 10 categories

    Reference:
        - Hendrycks et al. Benchmarking neural network robustness
        to common corruptions and perturbations. ICLR 2019.
    """

    dataset_dir = ""
    domains = ["cifar10", "cifar10_c"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = root

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )
        source_domain = cfg.DATASET.SOURCE_DOMAINS[0]
        target_domain = cfg.DATASET.TARGET_DOMAINS[0]
        assert source_domain == self.domains[0]
        assert target_domain == self.domains[1]

        c_type = cfg.DATASET.CIFAR_C_TYPE
        c_level = cfg.DATASET.CIFAR_C_LEVEL

        if not c_type:
            raise ValueError(
                "Please specify DATASET.CIFAR_C_TYPE in the config file"
            )

        assert (
            c_type in AVAI_C_TYPES
        ), f'C_TYPE is expected to belong to {AVAI_C_TYPES}, but got "{c_type}"'
        assert 1 <= c_level <= 5

        train_dir = osp.join(self.dataset_dir, source_domain, "train")
        test_dir = osp.join(
            self.dataset_dir, target_domain, c_type, str(c_level)
        )

        if not osp.exists(test_dir):
            raise ValueError

        train = self._read_data(train_dir)
        test = self._read_data(test_dir)

        super().__init__(train_x=train, test=test)

    def _read_data(self, data_dir):
        class_names = listdir_nohidden(data_dir)
        class_names.sort()
        items = []

        for label, class_name in enumerate(class_names):
            class_dir = osp.join(data_dir, class_name)
            imnames = listdir_nohidden(class_dir)

            for imname in imnames:
                impath = osp.join(class_dir, imname)
                item = Datum(impath=impath, label=label, domain=0)
                items.append(item)

        return items


@DATASET_REGISTRY.register()
class CIFAR100C(CIFAR10C):
    """CIFAR-100 -> CIFAR-100-C.

    Dataset link: https://zenodo.org/record/3555552#.YFxpQmQzb0o

    Statistics:
        - 2 domains: the normal CIFAR-100 vs. a corrupted CIFAR-100
        - 10 categories

    Reference:
        - Hendrycks et al. Benchmarking neural network robustness
        to common corruptions and perturbations. ICLR 2019.
    """

    dataset_dir = ""
    domains = ["cifar100", "cifar100_c"]

    def __init__(self, cfg):
        super().__init__(cfg)
