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
        # Note that the sampling process is NOT random,
        # which follows that in Volpi et al. NIPS'18.
        items = items[:n_max]

    if n_repeat is not None:
        items *= n_repeat

    return items


def load_mnist(dataset_dir, split='train'):
    data_dir = osp.join(dataset_dir, MNIST[split])
    n_max = 10000 if split == 'train' else None
    return read_image_list(data_dir, n_max=n_max)


def load_mnist_m(dataset_dir, split='train'):
    data_dir = osp.join(dataset_dir, MNIST_M[split])
    n_max = 10000 if split == 'train' else None
    return read_image_list(data_dir, n_max=n_max)


def load_svhn(dataset_dir, split='train'):
    data_dir = osp.join(dataset_dir, SVHN[split])
    n_max = 10000 if split == 'train' else None
    return read_image_list(data_dir, n_max=n_max)


def load_syn(dataset_dir, split='train'):
    data_dir = osp.join(dataset_dir, SYN[split])
    n_max = 10000 if split == 'train' else None
    return read_image_list(data_dir, n_max=n_max)


def load_usps(dataset_dir, split='train'):
    data_dir = osp.join(dataset_dir, USPS[split])
    return read_image_list(data_dir)


@DATASET_REGISTRY.register()
class DigitSingle(DatasetBase):
    """Digit recognition datasets for single-source domain generalization.

    There are five digit datasets:
        - MNIST: hand-written digits.
        - MNIST-M: variant of MNIST with blended background.
        - SVHN: street view house number.
        - SYN: synthetic digits.
        - USPS: hand-written digits, slightly different from MNIST.

    Protocol:
        Volpi et al. train a model using 10,000 images from MNIST and
        evaluate the model on the test split of the other four datasets. However,
        the code does not restrict you to only use MNIST as the source dataset.
        Instead, you can use any dataset as the source. But note that only 10,000
        images will be sampled from the source dataset for training.
    
    Reference:
        - Lecun et al. Gradient-based learning applied to document
        recognition. IEEE 1998.
        - Ganin et al. Domain-adversarial training of neural networks.
        JMLR 2016.
        - Netzer et al. Reading digits in natural images with unsupervised
        feature learning. NIPS-W 2011.
        - Volpi et al. Generalizing to Unseen Domains via Adversarial Data
        Augmentation. NIPS 2018.
    """
    # Reuse the digit-5 folder instead of creating a new folder
    dataset_dir = 'digit5'
    domains = ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split='train')
        val = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split='test')
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, split='test')

        super().__init__(train_x=train, val=val, test=test)

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
