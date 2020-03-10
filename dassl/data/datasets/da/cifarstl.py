import os.path as osp

from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class CIFARSTL(DatasetBase):
    """CIFAR-10 and STL-10.

    CIFAR-10:
        - 60,000 32x32 colour images.
        - 10 classes, with 6,000 images per class.
        - 50,000 training images and 10,000 test images.
        - URL: https://www.cs.toronto.edu/~kriz/cifar.html.

    STL-10:
        - 10 classes: airplane, bird, car, cat, deer, dog, horse,
        monkey, ship, truck.
        - Images are 96x96 pixels, color.
        - 500 training images (10 pre-defined folds), 800 test images
        per class.
        - URL: https://cs.stanford.edu/~acoates/stl10/.

    Reference:
        - Krizhevsky. Learning Multiple Layers of Features
        from Tiny Images. Tech report.
        - Coates et al. An Analysis of Single Layer Networks in
        Unsupervised Feature Learning. AISTATS 2011.
    """
    dataset_dir = 'cifar_stl'
    domains = ['cifar', 'stl']

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split='train')
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS, split='train')
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, split='test')

        super().__init__(train_x=train_x, train_u=train_u, test=test)

    def _read_data(self, input_domains, split='train'):
        items = []

        for domain, dname in enumerate(input_domains):
            data_dir = osp.join(self.dataset_dir, dname, split)
            class_names = listdir_nohidden(data_dir)

            for class_name in class_names:
                class_dir = osp.join(data_dir, class_name)
                imnames = listdir_nohidden(class_dir)
                label = int(class_name.split('_')[0])

                for imname in imnames:
                    impath = osp.join(class_dir, imname)
                    item = Datum(impath=impath, label=label, domain=domain)
                    items.append(item)

        return items
