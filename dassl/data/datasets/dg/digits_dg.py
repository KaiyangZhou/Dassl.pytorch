import glob
import os.path as osp

from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class DigitsDG(DatasetBase):
    """Digits-DG.

    It contains 4 digit datasets:
        - MNIST: hand-written digits.
        - MNIST-M: variant of MNIST with blended background.
        - SVHN: street view house number.
        - SYN: synthetic digits.

    Reference:
        - Lecun et al. Gradient-based learning applied to document
        recognition. IEEE 1998.
        - Ganin et al. Domain-adversarial training of neural networks.
        JMLR 2016.
        - Netzer et al. Reading digits in natural images with unsupervised
        feature learning. NIPS-W 2011.
        - Zhou et al. Deep Domain-Adversarial Image Generation for Domain
        Generalisation. AAAI 2020.
    """

    dataset_dir = "digits_dg"
    domains = ["mnist", "mnist_m", "svhn", "syn"]
    data_url = "https://drive.google.com/uc?id=15V7EsHfCcfbKgsDmzQKj_DfXt_XYp_P7"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, "digits_dg.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train = self.read_data(
            self.dataset_dir, cfg.DATASET.SOURCE_DOMAINS, "train"
        )
        val = self.read_data(
            self.dataset_dir, cfg.DATASET.SOURCE_DOMAINS, "val"
        )
        test = self.read_data(
            self.dataset_dir, cfg.DATASET.TARGET_DOMAINS, "all"
        )

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_data(dataset_dir, input_domains, split):

        def _load_data_from_directory(directory):
            folders = listdir_nohidden(directory)
            folders.sort()
            items_ = []

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(directory, folder, "*.jpg"))

                for impath in impaths:
                    items_.append((impath, label))

            return items_

        items = []

        for domain, dname in enumerate(input_domains):
            if split == "all":
                train_dir = osp.join(dataset_dir, dname, "train")
                impath_label_list = _load_data_from_directory(train_dir)
                val_dir = osp.join(dataset_dir, dname, "val")
                impath_label_list += _load_data_from_directory(val_dir)
            else:
                split_dir = osp.join(dataset_dir, dname, split)
                impath_label_list = _load_data_from_directory(split_dir)

            for impath, label in impath_label_list:
                class_name = impath.split("/")[-2].lower()
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=class_name
                )
                items.append(item)

        return items
