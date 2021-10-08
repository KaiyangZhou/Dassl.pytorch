"""
This script
- creates a folder named "cifar10_c" under the same directory as 'CIFAR-10-C'
- extracts images from .npy files and save them as .jpg.
"""
import os
import sys
import numpy as np
import os.path as osp
from PIL import Image

from dassl.utils import mkdir_if_missing


def extract_and_save(images, labels, level, dst):
    # level denotes the corruption intensity level (0-based)
    assert 0 <= level <= 4

    for i in range(10000):
        real_i = i + level*10000
        im = Image.fromarray(images[real_i])
        label = int(labels[real_i])
        category_dir = osp.join(dst, str(label).zfill(3))
        mkdir_if_missing(category_dir)
        save_path = osp.join(category_dir, str(i + 1).zfill(5) + ".jpg")
        im.save(save_path)


def main(npy_folder):
    npy_folder = osp.abspath(osp.expanduser(npy_folder))
    dataset_cap = osp.basename(npy_folder)

    assert dataset_cap in ["CIFAR-10-C", "CIFAR-100-C"]

    if dataset_cap == "CIFAR-10-C":
        dataset = "cifar10_c"
    else:
        dataset = "cifar100_c"

    if not osp.exists(npy_folder):
        print('The given folder "{}" does not exist'.format(npy_folder))

    root = osp.dirname(npy_folder)
    im_folder = osp.join(root, dataset)

    mkdir_if_missing(im_folder)

    dirnames = os.listdir(npy_folder)
    dirnames.remove("labels.npy")
    if "README.txt" in dirnames:
        dirnames.remove("README.txt")
    assert len(dirnames) == 19
    labels = np.load(osp.join(npy_folder, "labels.npy"))

    for dirname in dirnames:
        corruption = dirname.split(".")[0]
        corruption_folder = osp.join(im_folder, corruption)
        mkdir_if_missing(corruption_folder)

        npy_filename = osp.join(npy_folder, dirname)
        images = np.load(npy_filename)
        assert images.shape[0] == 50000

        for level in range(5):
            dst = osp.join(corruption_folder, str(level + 1))
            mkdir_if_missing(dst)
            print('Saving images to "{}"'.format(dst))
            extract_and_save(images, labels, level, dst)


if __name__ == "__main__":
    # sys.argv[1] contains the path to CIFAR-10-C or CIFAR-100-C
    main(sys.argv[1])
