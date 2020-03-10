import random
import os.path as osp

from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase

# Folder names for train and test sets
MNIST = {'train': 'train_images', 'test': 'test_images'}
MNIST_M = {'train': 'train_images', 'test': 'test_images'}
SVHN = {'train': 'train_images', 'test': 'test_images'}
SYN = {'train': 'train_images', 'test': 'test_images'}
USPS = {'train': 'train_images', 'test': 'test_images'}


def read_image_list(im_dir, n_max=None, n_repeat=None):
    items = []

    for imname in listdir_nohidden(im_dir):
        imname_noext = osp.splitext(imname)[0]
        label = int(imname_noext.split('_')[1])
        impath = osp.join(im_dir, imname)
        items.append((impath, label))

    if n_max is not None:
        items = random.sample(items, n_max)

    if n_repeat is not None:
        items *= n_repeat

    return items


def load_mnist(dataset_dir, split='train'):
    data_dir = osp.join(dataset_dir, MNIST[split])
    n_max = 25000 if split == 'train' else 9000
    return read_image_list(data_dir, n_max=n_max)


def load_mnist_m(dataset_dir, split='train'):
    data_dir = osp.join(dataset_dir, MNIST_M[split])
    n_max = 25000 if split == 'train' else 9000
    return read_image_list(data_dir, n_max=n_max)


def load_svhn(dataset_dir, split='train'):
    data_dir = osp.join(dataset_dir, SVHN[split])
    n_max = 25000 if split == 'train' else 9000
    return read_image_list(data_dir, n_max=n_max)


def load_syn(dataset_dir, split='train'):
    data_dir = osp.join(dataset_dir, SYN[split])
    n_max = 25000 if split == 'train' else 9000
    return read_image_list(data_dir, n_max=n_max)


def load_usps(dataset_dir, split='train'):
    data_dir = osp.join(dataset_dir, USPS[split])
    n_repeat = 3 if split == 'train' else None
    return read_image_list(data_dir, n_repeat=n_repeat)


@DATASET_REGISTRY.register()
class Digit5(DatasetBase):
    """Five digit datasets.

    It contains:
        - MNIST: hand-written digits.
        - MNIST-M: variant of MNIST with blended background.
        - SVHN: street view house number.
        - SYN: synthetic digits.
        - USPS: hand-written digits, slightly different from MNIST.

    For MNIST, MNIST-M, SVHN and SYN, we randomly sample 25,000 images from
    the training set and 9,000 images from the test set. For USPS which has only
    9,298 images in total, we use the entire dataset but replicate its training
    set for 3 times so as to match the training set size of other domains.
    
    Reference:
        - Lecun et al. Gradient-based learning applied to document
        recognition. IEEE 1998.
        - Ganin et al. Domain-adversarial training of neural networks.
        JMLR 2016.
        - Netzer et al. Reading digits in natural images with unsupervised
        feature learning. NIPS-W 2011.
    """
    dataset_dir = 'digit5'
    domains = ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']

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
            func = 'load_' + dname
            domain_dir = osp.join(self.dataset_dir, dname)
            items_d = eval(func)(domain_dir, split=split)

            for impath, label in items_d:
                item = Datum(impath=impath, label=label, domain=domain)
                items.append(item)

        return items
