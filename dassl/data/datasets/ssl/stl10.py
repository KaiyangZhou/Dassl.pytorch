import numpy as np
import os.path as osp

from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class STL10(DatasetBase):
    """STL-10 dataset.

    Description:
    - 10 classes: airplane, bird, car, cat, deer, dog, horse,
    monkey, ship, truck.
    - Images are 96x96 pixels, color.
    - 500 training images per class, 800 test images per class.
    - 100,000 unlabeled images for unsupervised learning.
    
    Reference:
        - Coates et al. An Analysis of Single Layer Networks in
        Unsupervised Feature Learning. AISTATS 2011.
    """
    dataset_dir = 'stl10'

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        train_dir = osp.join(self.dataset_dir, 'train')
        test_dir = osp.join(self.dataset_dir, 'test')
        unlabeled_dir = osp.join(self.dataset_dir, 'unlabeled')
        fold_file = osp.join(
            self.dataset_dir, 'stl10_binary', 'fold_indices.txt'
        )

        # Only use the first five splits
        assert 0 <= cfg.DATASET.STL10_FOLD <= 4

        train_x = self._read_data_train(
            train_dir, cfg.DATASET.STL10_FOLD, fold_file
        )
        train_u = self._read_data_all(unlabeled_dir)
        test = self._read_data_all(test_dir)

        if cfg.DATASET.ALL_AS_UNLABELED:
            train_u = train_u + train_x

        super().__init__(train_x=train_x, train_u=train_u, test=test)

    def _read_data_train(self, data_dir, fold, fold_file):
        imnames = listdir_nohidden(data_dir)
        imnames.sort()
        items = []

        list_idx = list(range(len(imnames)))
        if fold >= 0:
            with open(fold_file, 'r') as f:
                str_idx = f.read().splitlines()[fold]
                list_idx = np.fromstring(str_idx, dtype=np.uint8, sep=' ')

        for i in list_idx:
            imname = imnames[i]
            impath = osp.join(data_dir, imname)
            label = osp.splitext(imname)[0].split('_')[1]
            label = int(label)
            item = Datum(impath=impath, label=label)
            items.append(item)

        return items

    def _read_data_all(self, data_dir):
        imnames = listdir_nohidden(data_dir)
        items = []

        for imname in imnames:
            impath = osp.join(data_dir, imname)
            label = osp.splitext(imname)[0].split('_')[1]
            if label == 'none':
                label = -1
            else:
                label = int(label)
            item = Datum(impath=impath, label=label)
            items.append(item)

        return items
