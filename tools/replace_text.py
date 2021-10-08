"""
Replace text in python files.
"""
import glob
import os.path as osp
import argparse
import fileinput

EXTENSION = ".py"


def is_python_file(filename):
    ext = osp.splitext(filename)[1]
    return ext == EXTENSION


def update_file(filename, text_to_search, replacement_text):
    print("Processing {}".format(filename))
    with fileinput.FileInput(filename, inplace=True, backup="") as file:
        for line in file:
            print(line.replace(text_to_search, replacement_text), end="")


def recursive_update(directory, text_to_search, replacement_text):
    filenames = glob.glob(osp.join(directory, "*"))

    for filename in filenames:
        if osp.isfile(filename):
            if not is_python_file(filename):
                continue
            update_file(filename, text_to_search, replacement_text)
        elif osp.isdir(filename):
            recursive_update(filename, text_to_search, replacement_text)
        else:
            raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_or_dir", type=str, help="path to file or directory"
    )
    parser.add_argument("text_to_search", type=str, help="name to be replaced")
    parser.add_argument("replacement_text", type=str, help="new name")
    parser.add_argument(
        "--ext", type=str, default=".py", help="file extension"
    )
    args = parser.parse_args()

    file_or_dir = args.file_or_dir
    text_to_search = args.text_to_search
    replacement_text = args.replacement_text
    extension = args.ext

    global EXTENSION
    EXTENSION = extension

    if osp.isfile(file_or_dir):
        if not is_python_file(file_or_dir):
            return
        update_file(file_or_dir, text_to_search, replacement_text)
    elif osp.isdir(file_or_dir):
        recursive_update(file_or_dir, text_to_search, replacement_text)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
