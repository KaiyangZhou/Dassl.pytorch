# How to Prepare Datasets

We suggest you put datasets under the same directory `$DATA`, which looks like

```
$DATA/
    office31/
    office_home/
    visda17/
    ...
```

Please follow the download links and file structures to organize the datasets.

## Domain Adaptation

### Office-31

Download link: https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code.

File structure:

```
office31/
    amazon/
    dslr/
    webcam/
```

### Office-Home

Download link: http://hemanthdv.org/OfficeHome-Dataset/.

File structure:

```
office_home/
    art/
    clipart/
    product/
    real_world/
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
    train/
    test/
    validation/
```

### CIFAR10-STL10

Run the following command in your terminal under `Dassl.pytorch/datasets/da`,

```bash
python cifar_stl.py $DATA/cifar_stl
```

This will create a folder named `cifar_stl` under `$DATA`. The file structure will look like

```
cifar_stl/
    cifar/
        train/
        test/
    stl/
        train/
        test/
```

### Digit-5

Create a folder `$DATA/digit5` and download to this folder the dataset from [here](https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/tree/master/M3SDA/code_MSDA_digit#digit-five-download). This should give you

```
digit5/
    Digit-Five/
```

Then, run the following command in your terminal under `Dassl.pytorch/datasets/da`,

```bash 
python digit5.py $DATA
```

This will extract the data and organize the file structure as

```
digit5/
    Digit-Five/
    mnist/
    mnist_m/
    usps/
    svhn/
    syn/
```

### DomainNet

Download link: http://ai.bu.edu/M3SDA/. (Please download the cleaned version of split files)

File structure:

```
domainnet/
    clipart/
    infograph/
    painting/
    quickdraw/
    real/
    sketch/
    splits/
        clipart_train.txt
        clipart_test.txt
        ...
```

### miniDomainNet

You need to download the DomainNet dataset first. The miniDomainNet's split files can be downloaded at this [google drive](https://drive.google.com/open?id=15rrLDCrzyi6ZY-1vJar3u7plgLe4COL7). After the zip file is extracted, you should have the folder `$DATA/domainnet/splits_mini/`.

## Domain Generalization

### PACS

Download link: [google drive](https://drive.google.com/open?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE).

File structure:

```
pacs/
    images/
    splits/
```

It is ok to not manually download this dataset because once you run ``tools/train.py``, the code will detect if the dataset exists or not and automatically download the dataset to ``$DATA`` if missing. This applies to PACS, Office-Home and Digits-DG.

### Office-Home

Download link: [google drive](https://drive.google.com/open?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa).

File structure:

```
office_home_dg/
    art/
    clipart/
    product/
    real_world/
```

### Digits-DG

Download link: [google driv](https://drive.google.com/open?id=15V7EsHfCcfbKgsDmzQKj_DfXt_XYp_P7).

File structure:

```
digits_dg/
    mnist/
    mnist_m/
    svhn/
    syn/
```

## Semi-Supervised Learning

### CIFAR10/100 and SVHN

Run the following command in your terminal under `Dassl.pytorch/datasets/ssl`,

```bash
python cifar10_cifar100_svhn.py $DATA
```

This will create three folders under `$DATA`, i.e.

```
ssl_cifar10/
    train/
    test/
ssl_cifar100/
    train/
    test/
ssl_svhn/
    train/
    test/
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
    train/
    test/
    unlabeled/
    stl10_binary/
```