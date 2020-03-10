import os.path as osp

from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class Office31(DatasetBase):
    """Office-31.

    Statistics:
        - 4,110 images.
        - 31 classes related to office objects.
        - 3 domains: Amazon, Webcam, Dslr.
        - URL: https://people.eecs.berkeley.edu/~jhoffman/domainadapt/.

    Reference:
        - Saenko et al. Adapting visual category models to
        new domains. ECCV 2010.
    """
    dataset_dir = 'office31'
    domains = ['amazon', 'webcam', 'dslr']

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS)
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS)
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS)

        super().__init__(train_x=train_x, train_u=train_u, test=test)

    def _read_data(self, input_domains):
        items = []

        for domain, dname in enumerate(input_domains):
            domain_dir = osp.join(self.dataset_dir, dname)
            class_names = listdir_nohidden(domain_dir)
            class_names.sort()

            for label, class_name in enumerate(class_names):
                class_path = osp.join(domain_dir, class_name)
                imnames = listdir_nohidden(class_path)

                for imname in imnames:
                    impath = osp.join(class_path, imname)
                    item = Datum(impath=impath, label=label, domain=domain)
                    items.append(item)

        return items
