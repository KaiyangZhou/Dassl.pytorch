# How to Install Datasets

`$DATA` denotes the location where datasets are installed, e.g.

```
$DATA/
|–– office31/
|–– office_home/
|–– visda17/
```

[Domain Adaptation](#domain-adaptation)
- [Office-31](#office-31)
- [Office-Home](#office-home)
- [VisDA17](#visda17)
- [CIFAR10-STL10](#cifar10-stl10)
- [Digit-5](#digit-5)
- [DomainNet](#domainnet)
- [miniDomainNet](#miniDomainNet)

[Domain Generalization](#domain-generalization)
- [PACS](#pacs)
- [VLCS](#vlcs)
- [Office-Home-DG](#office-home-dg)
- [Digits-DG](#digits-dg)
- [Digit-Single](#digit-single)
- [CIFAR-10-C](#cifar-10-c)
- [CIFAR-100-C](#cifar-100-c)

[Semi-Supervised Learning](#semi-supervised-learning)
- [CIFAR10/100 and SVHN](#cifar10100-and-svhn)
- [STL10](#stl10)

## Domain Adaptation

### Office-31

Download link: https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code.

File structure:

```
office31/
|–– amazon/
|   |–– back_pack/
|   |–– bike/
|   |–– ...
|–– dslr/
|   |–– back_pack/
|   |–– bike/
|   |–– ...
|–– webcam/
|   |–– back_pack/
|   |–– bike/
|   |–– ...
```

Note that within each domain folder you need to move all class folders out of the `images/` folder and then delete the `images/` folder.

### Office-Home

Download link: http://hemanthdv.org/OfficeHome-Dataset/.

File structure:

```
office_home/
|–– art/
|–– clipart/
|–– product/
|–– real_world/
```

### VisDA17

Download link: http://ai.bu.edu/visda-2017/.

The dataset can also be downloaded using our script at `datasets/da/visda17.sh`. Run the following command in your terminal under `Dassl.pytorch/datasets/da`,

```bash
sh visda17.sh $DATA
```

Once the download is finished, the file structure will look like

```
visda17/
|–– train/
|–– test/
|–– validation/
```

### CIFAR10-STL10

Run the following command in your terminal under `Dassl.pytorch/datasets/da`,

```bash
python cifar_stl.py $DATA/cifar_stl
```

This will create a folder named `cifar_stl` under `$DATA`. The file structure will look like

```
cifar_stl/
|–– cifar/
|   |–– train/
|   |–– test/
|–– stl/
|   |–– train/
|   |–– test/
```

Note that only 9 classes shared by both datasets are kept.

### Digit-5

Create a folder `$DATA/digit5` and download to this folder the dataset from [here](https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/tree/master/M3SDA/code_MSDA_digit#digit-five-download). This should give you

```
digit5/
|–– Digit-Five/
```

Then, run the following command in your terminal under `Dassl.pytorch/datasets/da`,

```bash 
python digit5.py $DATA/digit5
```

This will extract the data and organize the file structure as

```
digit5/
|–– Digit-Five/
|–– mnist/
|–– mnist_m/
|–– usps/
|–– svhn/
|–– syn/
```

### DomainNet

Download link: http://ai.bu.edu/M3SDA/. (Please download the cleaned version of split files)

File structure:

```
domainnet/
|–– clipart/
|–– infograph/
|–– painting/
|–– quickdraw/
|–– real/
|–– sketch/
|–– splits/
|   |–– clipart_train.txt
|   |–– clipart_test.txt
|   |–– ...
```

### miniDomainNet

You need to download the DomainNet dataset first. The miniDomainNet's split files can be downloaded at this [google drive](https://drive.google.com/open?id=15rrLDCrzyi6ZY-1vJar3u7plgLe4COL7). After the zip file is extracted, you should have the folder `$DATA/domainnet/splits_mini/`.

## Domain Generalization

### PACS

Download link: [google drive](https://drive.google.com/open?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE).

File structure:

```
pacs/
|–– images/
|–– splits/
```

You do not necessarily have to manually download this dataset. Once you run ``tools/train.py``, the code will detect if the dataset exists or not and automatically download the dataset to ``$DATA`` if missing. This also applies to VLCS, Office-Home-DG, and Digits-DG.

### VLCS

Download link: [google drive](https://drive.google.com/file/d/1r0WL5DDqKfSPp9E3tRENwHaXNs1olLZd/view?usp=sharing) (credit to https://github.com/fmcarlucci/JigenDG#vlcs)

File structure:

```
VLCS/
|–– CALTECH/
|–– LABELME/
|–– PASCAL/
|–– SUN/
```

### Office-Home-DG

Download link: [google drive](https://drive.google.com/open?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa).

File structure:

```
office_home_dg/
|–– art/
|–– clipart/
|–– product/
|–– real_world/
```

### Digits-DG

Download link: [google driv](https://drive.google.com/open?id=15V7EsHfCcfbKgsDmzQKj_DfXt_XYp_P7).

File structure:

```
digits_dg/
|–– mnist/
|–– mnist_m/
|–– svhn/
|–– syn/
```

### Digit-Single
Follow the steps for [Digit-5](#digit-5) to organize the dataset.

### CIFAR-10-C

First download the CIFAR-10-C dataset from https://zenodo.org/record/2535967#.YFxHEWQzb0o to, e.g., $DATA, and extract the file under the same directory. Then, navigate to `Dassl.pytorch/datasets/dg` and run the following command in your terminal
```bash
python cifar_c.py $DATA/CIFAR-10-C
```
where the first argument denotes the path to the (uncompressed) CIFAR-10-C dataset.

The script will extract images from the `.npy` files and save them to `cifar10_c/` created under $DATA. The file structure will look like
```
cifar10_c/
|–– brightness/
|   |–– 1/ # 5 intensity levels in total
|   |–– 2/
|   |–– 3/
|   |–– 4/
|   |–– 5/
|–– ... # 19 corruption types in total
```

Note that `cifar10_c/` only contains the test images. The training images are the normal CIFAR-10 images. See [CIFAR10/100 and SVHN](#cifar10100-and-svhn) for how to prepare the CIFAR-10 dataset.

### CIFAR-100-C

First download the CIFAR-100-C dataset from https://zenodo.org/record/3555552#.YFxpQmQzb0o to, e.g., $DATA, and extract the file under the same directory. Then, navigate to `Dassl.pytorch/datasets/dg` and run the following command in your terminal
```bash
python cifar_c.py $DATA/CIFAR-100-C
```
where the first argument denotes the path to the (uncompressed) CIFAR-100-C dataset.

The script will extract images from the `.npy` files and save them to `cifar100_c/` created under $DATA. The file structure will look like
```
cifar100_c/
|–– brightness/
|   |–– 1/ # 5 intensity levels in total
|   |–– 2/
|   |–– 3/
|   |–– 4/
|   |–– 5/
|–– ... # 19 corruption types in total
```

Note that `cifar100_c/` only contains the test images. The training images are the normal CIFAR-100 images. See [CIFAR10/100 and SVHN](#cifar10100-and-svhn) for how to prepare the CIFAR-100 dataset.

## Semi-Supervised Learning

### CIFAR10/100 and SVHN

Run the following command in your terminal under `Dassl.pytorch/datasets/ssl`,

```bash
python cifar10_cifar100_svhn.py $DATA
```

This will create three folders under `$DATA`, i.e.

```
cifar10/
|–– train/
|–– test/
cifar100/
|–– train/
|–– test/
svhn/
|–– train/
|–– test/
```

### STL10

Run the following command in your terminal under `Dassl.pytorch/datasets/ssl`,

```bash
python stl10.py $DATA/stl10
```

This will create a folder named `stl10` under `$DATA` and extract the data into three folders, i.e. `train`, `test` and `unlabeled`. Then, download from http://ai.stanford.edu/~acoates/stl10/ the "Binary files" and extract it under `stl10`.

The file structure will look like

```
stl10/
|–– train/
|–– test/
|–– unlabeled/
|–– stl10_binary/
```