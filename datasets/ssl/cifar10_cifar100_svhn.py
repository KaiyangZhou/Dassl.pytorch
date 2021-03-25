import sys
import os.path as osp
from torchvision.datasets import SVHN, CIFAR10, CIFAR100

from dassl.utils import mkdir_if_missing


def extract_and_save_image(dataset, save_dir):
    if osp.exists(save_dir):
        print('Folder "{}" already exists'.format(save_dir))
        return

    print('Extracting images to "{}" ...'.format(save_dir))
    mkdir_if_missing(save_dir)

    for i in range(len(dataset)):
        img, label = dataset[i]
        class_dir = osp.join(save_dir, str(label).zfill(3))
        mkdir_if_missing(class_dir)
        impath = osp.join(class_dir, str(i + 1).zfill(5) + '.jpg')
        img.save(impath)


def download_and_prepare(name, root):
    print('Dataset: {}'.format(name))
    print('Root: {}'.format(root))

    if name == 'cifar10':
        train = CIFAR10(root, train=True, download=True)
        test = CIFAR10(root, train=False)
    elif name == 'cifar100':
        train = CIFAR100(root, train=True, download=True)
        test = CIFAR100(root, train=False)
    elif name == 'svhn':
        train = SVHN(root, split='train', download=True)
        test = SVHN(root, split='test', download=True)
    else:
        raise ValueError

    train_dir = osp.join(root, name, 'train')
    test_dir = osp.join(root, name, 'test')

    extract_and_save_image(train, train_dir)
    extract_and_save_image(test, test_dir)


if __name__ == '__main__':
    download_and_prepare('cifar10', sys.argv[1])
    download_and_prepare('cifar100', sys.argv[1])
    download_and_prepare('svhn', sys.argv[1])
