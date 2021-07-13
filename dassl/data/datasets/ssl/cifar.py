import math
import random
import os.path as osp

from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class CIFAR10(DatasetBase):
    """CIFAR10 for SSL.
    
    Reference:
        - Krizhevsky. Learning Multiple Layers of Features
        from Tiny Images. Tech report.
    """
    dataset_dir = 'cifar10'

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        train_dir = osp.join(self.dataset_dir, 'train')
        test_dir = osp.join(self.dataset_dir, 'test')

        assert cfg.DATASET.NUM_LABELED > 0

        train_x, train_u, val = self._read_data_train(
            train_dir, cfg.DATASET.NUM_LABELED, cfg.DATASET.VAL_PERCENT
        )
        test = self._read_data_test(test_dir)

        if cfg.DATASET.ALL_AS_UNLABELED:
            train_u = train_u + train_x

        if len(val) == 0:
            val = None

        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)

    def _read_data_train(self, data_dir, num_labeled, val_percent):
        class_names = listdir_nohidden(data_dir)
        class_names.sort()
        num_labeled_per_class = num_labeled / len(class_names)
        items_x, items_u, items_v = [], [], []

        for label, class_name in enumerate(class_names):
            class_dir = osp.join(data_dir, class_name)
            imnames = listdir_nohidden(class_dir)

            # Split into train and val following Oliver et al. 2018
            # Set cfg.DATASET.VAL_PERCENT to 0 to not use val data
            num_val = math.floor(len(imnames) * val_percent)
            imnames_train = imnames[num_val:]
            imnames_val = imnames[:num_val]

            # Note we do shuffle after split
            random.shuffle(imnames_train)

            for i, imname in enumerate(imnames_train):
                impath = osp.join(class_dir, imname)
                item = Datum(impath=impath, label=label)

                if (i + 1) <= num_labeled_per_class:
                    items_x.append(item)

                else:
                    items_u.append(item)

            for imname in imnames_val:
                impath = osp.join(class_dir, imname)
                item = Datum(impath=impath, label=label)
                items_v.append(item)

        return items_x, items_u, items_v

    def _read_data_test(self, data_dir):
        class_names = listdir_nohidden(data_dir)
        class_names.sort()
        items = []

        for label, class_name in enumerate(class_names):
            class_dir = osp.join(data_dir, class_name)
            imnames = listdir_nohidden(class_dir)

            for imname in imnames:
                impath = osp.join(class_dir, imname)
                item = Datum(impath=impath, label=label)
                items.append(item)

        return items


@DATASET_REGISTRY.register()
class CIFAR100(CIFAR10):
    """CIFAR100 for SSL.
    
    Reference:
        - Krizhevsky. Learning Multiple Layers of Features
        from Tiny Images. Tech report.
    """
    dataset_dir = 'cifar100'

    def __init__(self, cfg):
        super().__init__(cfg)
