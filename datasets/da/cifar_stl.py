import sys
import pprint as pp
import os.path as osp
from torchvision.datasets import STL10, CIFAR10

from dassl.utils import mkdir_if_missing

cifar_label2name = {
    0: "airplane",
    1: "car",  # the original name was 'automobile'
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",  # conflict class
    7: "horse",
    8: "ship",
    9: "truck",
}

stl_label2name = {
    0: "airplane",
    1: "bird",
    2: "car",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "horse",
    7: "monkey",  # conflict class
    8: "ship",
    9: "truck",
}

new_name2label = {
    "airplane": 0,
    "bird": 1,
    "car": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "horse": 6,
    "ship": 7,
    "truck": 8,
}


def extract_and_save_image(dataset, save_dir, discard, label2name):
    if osp.exists(save_dir):
        print('Folder "{}" already exists'.format(save_dir))
        return

    print('Extracting images to "{}" ...'.format(save_dir))
    mkdir_if_missing(save_dir)

    for i in range(len(dataset)):
        img, label = dataset[i]
        if label == discard:
            continue
        class_name = label2name[label]
        label_new = new_name2label[class_name]
        class_dir = osp.join(
            save_dir,
            str(label_new).zfill(3) + "_" + class_name
        )
        mkdir_if_missing(class_dir)
        impath = osp.join(class_dir, str(i + 1).zfill(5) + ".jpg")
        img.save(impath)


def download_and_prepare(name, root, discarded_label, label2name):
    print("Dataset: {}".format(name))
    print("Root: {}".format(root))
    print("Old labels:")
    pp.pprint(label2name)
    print("Discarded label: {}".format(discarded_label))
    print("New labels:")
    pp.pprint(new_name2label)

    if name == "cifar":
        train = CIFAR10(root, train=True, download=True)
        test = CIFAR10(root, train=False)
    else:
        train = STL10(root, split="train", download=True)
        test = STL10(root, split="test")

    train_dir = osp.join(root, name, "train")
    test_dir = osp.join(root, name, "test")

    extract_and_save_image(train, train_dir, discarded_label, label2name)
    extract_and_save_image(test, test_dir, discarded_label, label2name)


if __name__ == "__main__":
    download_and_prepare("cifar", sys.argv[1], 6, cifar_label2name)
    download_and_prepare("stl", sys.argv[1], 7, stl_label2name)
