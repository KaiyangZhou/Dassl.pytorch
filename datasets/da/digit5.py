import os
import numpy as np
import os.path as osp
import argparse
from PIL import Image
from scipy.io import loadmat


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        os.makedirs(directory)


def extract_and_save(data, label, save_dir):
    for i, (x, y) in enumerate(zip(data, label)):
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        if y == 10:
            y = 0
        x = Image.fromarray(x, mode="RGB")
        save_path = osp.join(
            save_dir,
            str(i + 1).zfill(6) + "_" + str(y) + ".jpg"
        )
        x.save(save_path)


def load_mnist(data_dir, raw_data_dir):
    filepath = osp.join(raw_data_dir, "mnist_data.mat")
    data = loadmat(filepath)

    train_data = np.reshape(data["train_32"], (55000, 32, 32, 1))
    test_data = np.reshape(data["test_32"], (10000, 32, 32, 1))

    train_label = np.nonzero(data["label_train"])[1]
    test_label = np.nonzero(data["label_test"])[1]

    return train_data, test_data, train_label, test_label


def load_mnist_m(data_dir, raw_data_dir):
    filepath = osp.join(raw_data_dir, "mnistm_with_label.mat")
    data = loadmat(filepath)

    train_data = data["train"]
    test_data = data["test"]

    train_label = np.nonzero(data["label_train"])[1]
    test_label = np.nonzero(data["label_test"])[1]

    return train_data, test_data, train_label, test_label


def load_svhn(data_dir, raw_data_dir):
    train = loadmat(osp.join(raw_data_dir, "svhn_train_32x32.mat"))
    train_data = train["X"].transpose(3, 0, 1, 2)
    train_label = train["y"][:, 0]

    test = loadmat(osp.join(raw_data_dir, "svhn_test_32x32.mat"))
    test_data = test["X"].transpose(3, 0, 1, 2)
    test_label = test["y"][:, 0]

    return train_data, test_data, train_label, test_label


def load_syn(data_dir, raw_data_dir):
    filepath = osp.join(raw_data_dir, "syn_number.mat")
    data = loadmat(filepath)

    train_data = data["train_data"]
    test_data = data["test_data"]

    train_label = data["train_label"][:, 0]
    test_label = data["test_label"][:, 0]

    return train_data, test_data, train_label, test_label


def load_usps(data_dir, raw_data_dir):
    filepath = osp.join(raw_data_dir, "usps_28x28.mat")
    data = loadmat(filepath)["dataset"]

    train_data = data[0][0].transpose(0, 2, 3, 1)
    test_data = data[1][0].transpose(0, 2, 3, 1)

    train_data *= 255
    test_data *= 255

    train_data = train_data.astype(np.uint8)
    test_data = test_data.astype(np.uint8)

    train_label = data[0][1][:, 0]
    test_label = data[1][1][:, 0]

    return train_data, test_data, train_label, test_label


def main(data_dir):
    data_dir = osp.abspath(osp.expanduser(data_dir))
    raw_data_dir = osp.join(data_dir, "Digit-Five")

    if not osp.exists(data_dir):
        raise FileNotFoundError('"{}" does not exist'.format(data_dir))

    datasets = ["mnist", "mnist_m", "svhn", "syn", "usps"]

    for name in datasets:
        print("Creating {}".format(name))

        output = eval("load_" + name)(data_dir, raw_data_dir)
        train_data, test_data, train_label, test_label = output

        print("# train: {}".format(train_data.shape[0]))
        print("# test: {}".format(test_data.shape[0]))

        train_dir = osp.join(data_dir, name, "train_images")
        mkdir_if_missing(train_dir)
        test_dir = osp.join(data_dir, name, "test_images")
        mkdir_if_missing(test_dir)

        extract_and_save(train_data, train_label, train_dir)
        extract_and_save(test_data, test_label, test_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", type=str, help="directory containing Digit-Five/"
    )
    args = parser.parse_args()
    main(args.data_dir)
