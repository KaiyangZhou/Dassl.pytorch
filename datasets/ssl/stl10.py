import sys
import os.path as osp
from torchvision.datasets import STL10

from dassl.utils import mkdir_if_missing


def extract_and_save_image(dataset, save_dir):
    if osp.exists(save_dir):
        print('Folder "{}" already exists'.format(save_dir))
        return

    print('Extracting images to "{}" ...'.format(save_dir))
    mkdir_if_missing(save_dir)

    for i in range(len(dataset)):
        img, label = dataset[i]
        if label == -1:
            label_name = 'none'
        else:
            label_name = str(label)
        imname = str(i).zfill(6) + '_' + label_name + '.jpg'
        impath = osp.join(save_dir, imname)
        img.save(impath)


def download_and_prepare(root):
    train = STL10(root, split='train', download=True)
    test = STL10(root, split='test')
    unlabeled = STL10(root, split='unlabeled')

    train_dir = osp.join(root, 'train')
    test_dir = osp.join(root, 'test')
    unlabeled_dir = osp.join(root, 'unlabeled')

    extract_and_save_image(train, train_dir)
    extract_and_save_image(test, test_dir)
    extract_and_save_image(unlabeled, unlabeled_dir)


if __name__ == '__main__':
    download_and_prepare(sys.argv[1])
