"""
Goal
---
1. Read test results from log.txt files
2. Compute mean and std across different seeds

Usage
---
Assume the output files are saved under output/my_experiment,
which contains results of different seeds, e.g.,

my_experiment/
    seed1/
        log.txt
    seed2/
        log.txt
    seed3/
        log.txt

Run the following command from the root directory:

$ python tools/parse_test_res.py output/my_experiment

Add --ci95 to the argument if you wanna get 95% confidence
interval instead of standard deviation:

$ python tools/parse_test_res.py output/my_experiment --ci95

If my_experiment/ has the following structure,

my_experiment/
    exp-1/
        seed1/
            log.txt
            ...
        seed2/
            log.txt
            ...
        seed3/
            log.txt
            ...
    exp-2/
        ...
    exp-3/
        ...

Run

$ python tools/parse_test_res.py output/my_experiment --multi-exp
"""
import re
import numpy as np
import os.path as osp
import argparse

from dassl.utils import check_isfile, listdir_nohidden


def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))


def parse_dir(directory, end_signal, regex_acc, regex_err, args):
    print('Parsing {}'.format(directory))
    subdirs = listdir_nohidden(directory, sort=True)

    valid_fpaths = []
    valid_accs = []
    valid_errs = []

    for subdir in subdirs:
        fpath = osp.join(directory, subdir, 'log.txt')
        assert check_isfile(fpath)
        good_to_go = False

        with open(fpath, 'r') as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                if line == end_signal:
                    good_to_go = True

                match_acc = regex_acc.search(line)
                if match_acc and good_to_go:
                    acc = float(match_acc.group(1))
                    valid_accs.append(acc)
                    valid_fpaths.append(fpath)

                match_err = regex_err.search(line)
                if match_err and good_to_go:
                    err = float(match_err.group(1))
                    valid_errs.append(err)

    for fpath, acc, err in zip(valid_fpaths, valid_accs, valid_errs):
        print('file: {}. acc: {:.2f}%. err: {:.2f}%'.format(fpath, acc, err))

    acc_mean = np.mean(valid_accs)
    acc_std = compute_ci95(valid_accs) if args.ci95 else np.std(valid_accs)

    err_mean = np.mean(valid_errs)
    err_std = compute_ci95(valid_errs) if args.ci95 else np.std(valid_errs)

    print('===')
    print('outcome of directory: {}'.format(directory))
    print('* acc: {:.2f}% +- {:.2f}%'.format(acc_mean, acc_std))
    print('* err: {:.2f}% +- {:.2f}%'.format(err_mean, err_std))
    print('===')

    return acc_mean, err_mean


def main(args, end_signal):
    regex_acc = re.compile(r'\* accuracy: ([\.\deE+-]+)%')
    regex_err = re.compile(r'\* error: ([\.\deE+-]+)%')

    if args.multi_exp:
        accs, errs = [], []
        for directory in listdir_nohidden(args.directory, sort=True):
            directory = osp.join(args.directory, directory)
            acc, err = parse_dir(
                directory, end_signal, regex_acc, regex_err, args
            )
            accs.append(acc)
            errs.append(err)
        acc_mean = np.mean(accs)
        err_mean = np.mean(errs)
        print('overall average')
        print('* acc: {:.2f}%'.format(acc_mean))
        print('* err: {:.2f}%'.format(err_mean))
    else:
        parse_dir(args.directory, end_signal, regex_acc, regex_err, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, help='Path to directory')
    parser.add_argument(
        '--ci95',
        action='store_true',
        help=r'Compute 95\% confidence interval'
    )
    parser.add_argument(
        '--test-log', action='store_true', help='Process test log'
    )
    parser.add_argument(
        '--multi-exp', action='store_true', help='multiple experiments'
    )
    args = parser.parse_args()
    end_signal = 'Finished training'
    if args.test_log:
        end_signal = '=> result'
    main(args, end_signal)
