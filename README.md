# Dassl

## Introduction

Dassl is a [PyTorch](https://pytorch.org) toolbox initially developed for our project [Domain Adaptive Ensemble Learning (DAEL)](https://arxiv.org/abs/2003.07325) to support research in domain adaptation and generalization---since in DAEL we study how to unify these two problems in a single learning framework. Given that domain adaptation is closely related to semi-supervised learning---both study how to exploit unlabeled data---we also incorporate components that support research for the latter.

Why the name "Dassl"? Dassl combines the initials of domain adaptation (DA) and semi-supervised learning (SSL), which sounds natural and informative.

Dassl has a modular design and unified interfaces, allowing fast prototyping and experimentation of new DA/DG/SSL methods. With Dassl, a new method can be implemented with only a few lines of code. Don't believe? Take a look at the [engine](https://github.com/KaiyangZhou/Dassl.pytorch/tree/master/dassl/engine) folder, which contains the implementations of many existing methods (then you will come back and star this repo). :-)

Basically, Dassl is perfect for doing research in the following areas:
- Domain adaptation
- Domain generalization
- Semi-supervised learning

BUT, thanks to the neat design, Dassl can also be used as a codebase to develop any deep learning projects, like [this](https://github.com/KaiyangZhou/CoOp). :-)

A drawback of Dassl is that it doesn't (yet? hmm) support distributed multi-GPU training (Dassl uses `DataParallel` to wrap a model, which is less efficient than `DistributedDataParallel`).

We don't provide detailed documentations for Dassl, unlike another [project](https://kaiyangzhou.github.io/deep-person-reid/) of ours. This is because Dassl is developed for research purpose and as a researcher, we think it's important to be able to read source code and we highly encourage you to do so---definitely not because we are lazy. :-)

## What's new
- **[Oct 2022]** New paper "[On-Device Domain Generalization](https://arxiv.org/abs/2209.07521)" is out! Code, models and datasets: https://github.com/KaiyangZhou/on-device-dg.

<details>
    <summary>More</summary>

- **[Jun 2022]** `v0.6.0`: Make `cfg.TRAINER.METHOD_NAME` consistent with the method class name.
- **[Jun 2022]** A new domain adaptation method [CDAC (CVPR'21)](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Cross-Domain_Adaptive_Clustering_for_Semi-Supervised_Domain_Adaptation_CVPR_2021_paper.pdf) is added by [Shreejal Trivedi](https://github.com/shreejalt). See [here](https://github.com/KaiyangZhou/Dassl.pytorch/pull/44) for more details.
- **[Jun 2022]** Adds three datasets from the [WILDS](https://wilds.stanford.edu/) benchmark: iWildCam, FMoW and Camelyon17. See [here](https://github.com/KaiyangZhou/Dassl.pytorch/commit/7f7eab8e22f6e176b97a539100eca12d6a403909) for more details.
- **[May 2022]** A new domain generalization method [DDG](https://arxiv.org/abs/2205.13913) developed by [Zhishu Sun](https://github.com/siaimes) and to appear at IJCAI'22 is added to this repo. See [here](https://github.com/MetaVisionLab/DDG) for more details.
- **[Mar 2022]** A new domain generalization method [EFDM](https://arxiv.org/abs/2203.07740) developed by [Yabin Zhang (PolyU)](https://ybzh.github.io/) and to appear at CVPR'22 is added to this repo. See [here](https://github.com/KaiyangZhou/Dassl.pytorch/pull/36) for more details.
- **[Feb 2022]** In case you don't know, a class in the painting domain of DomainNet (the official splits) only has test images (no training images), which could affect performance. See section 4.a in our [paper](https://arxiv.org/abs/2003.07325) for more details.
- **[Oct 2021]** `v0.5.0`: **Important changes** made to `transforms.py`. 1) `center_crop` becomes a default transform in testing (applied after resizing the smaller edge to a certain size to keep the image aspect ratio). 2) For training, `Resize(cfg.INPUT.SIZE)` is deactivated when `random_crop` or `random_resized_crop` is used. These changes won't make any difference to the training transforms used in existing config files, nor to the testing transforms unless the raw images are not squared (the only difference is that now the image aspect ratio is respected).
- **[Oct 2021]** `v0.4.3`: Copy the attributes in `self.dm` (data manager) to `SimpleTrainer` and make `self.dm` optional, which means from now on, you can build data loaders from any source you like rather than being forced to use `DataManager`.
- **[Sep 2021]** `v0.4.2`: An important update is to set `drop_last=is_train and len(data_source)>=batch_size` when constructing a data loader to avoid 0-length.

</details>

## Overview

Dassl has implemented the following methods:

- Single-source domain adaptation
    - [Cross Domain Adaptive Clustering for Semi Supervised Domain Adaptation (CVPR'21)](https://arxiv.org/pdf/2104.09415.pdf) [[dassl/engine/da/cdac.py](dassl/engine/da/cdac.py)]
    - [Semi-supervised Domain Adaptation via Minimax Entropy (ICCV'19)](https://arxiv.org/abs/1904.06487) [[dassl/engine/da/mme.py](dassl/engine/da/mme.py)]
    - [Maximum Classifier Discrepancy for Unsupervised Domain Adaptation (CVPR'18)](https://arxiv.org/abs/1712.02560https://arxiv.org/abs/1712.02560) [[dassl/engine/da/mcd.py](dassl/engine/da/mcd.py)]
    - [Self-ensembling for visual domain adaptation (ICLR'18)](https://arxiv.org/abs/1706.05208) [[dassl/engine/da/self_ensembling.py](dassl/engine/da/self_ensembling.py)]
    - [Revisiting Batch Normalization For Practical Domain Adaptation (ICLR-W'17)](https://arxiv.org/abs/1603.04779) [[dassl/engine/da/adabn.py](dassl/engine/da/adabn.py)]
    - [Adversarial Discriminative Domain Adaptation (CVPR'17)](https://arxiv.org/abs/1702.05464) [[dassl/engine/da/adda.py](dassl/engine/da/adda.py)]
    - [Domain-Adversarial Training of Neural Networks (JMLR'16) ](https://arxiv.org/abs/1505.07818) [[dassl/engine/da/dann.py](dassl/engine/da/dann.py)]

- Multi-source domain adaptation
    - [Domain Aadaptive Ensemble Learning](https://arxiv.org/abs/2003.07325) [[dassl/engine/da/dael.py](dassl/engine/da/dael.py)]
    - [Moment Matching for Multi-Source Domain Adaptation (ICCV'19)](https://arxiv.org/abs/1812.01754) [[dassl/engine/da/m3sda.py](dassl/engine/da/m3sda.py)]

- Domain generalization
    - [Dynamic Domain Generalization (IJCAI'22)](https://arxiv.org/abs/2205.13913) [[dassl/modeling/backbone/resnet_dynamic.py](dassl/modeling/backbone/resnet_dynamic.py)] [[dassl/engine/dg/domain_mix.py](dassl/engine/dg/domain_mix.py)]
    - [Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization (CVPR'22)](https://arxiv.org/abs/2203.07740) [[dassl/modeling/ops/efdmix.py](dassl/modeling/ops/efdmix.py)]
    - [Domain Generalization with MixStyle (ICLR'21)](https://openreview.net/forum?id=6xHJ37MVxxp) [[dassl/modeling/ops/mixstyle.py](dassl/modeling/ops/mixstyle.py)]
    - [Deep Domain-Adversarial Image Generation for Domain Generalisation (AAAI'20)](https://arxiv.org/abs/2003.06054) [[dassl/engine/dg/ddaig.py](dassl/engine/dg/ddaig.py)]
    - [Generalizing Across Domains via Cross-Gradient Training (ICLR'18)](https://arxiv.org/abs/1804.10745) [[dassl/engine/dg/crossgrad.py](dassl/engine/dg/crossgrad.py)]

- Semi-supervised learning
    - [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685) [[dassl/engine/ssl/fixmatch.py](dassl/engine/ssl/fixmatch.py)]
    - [MixMatch: A Holistic Approach to Semi-Supervised Learning (NeurIPS'19)](https://arxiv.org/abs/1905.02249) [[dassl/engine/ssl/mixmatch.py](dassl/engine/ssl/mixmatch.py)]
    - [Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results (NeurIPS'17)](https://arxiv.org/abs/1703.01780) [[dassl/engine/ssl/mean_teacher.py](dassl/engine/ssl/mean_teacher.py)]
    - [Semi-supervised Learning by Entropy Minimization (NeurIPS'04)](http://papers.nips.cc/paper/2740-semi-supervised-learning-by-entropy-minimization.pdf) [[dassl/engine/ssl/entmin.py](dassl/engine/ssl/entmin.py)]

*Feel free to make a [PR](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) to add your methods here to make it easier for others to benchmark!*

Dassl supports the following datasets:

- Domain adaptation
    - [Office-31](https://scalable.mpi-inf.mpg.de/files/2013/04/saenko_eccv_2010.pdf)
    - [Office-Home](http://hemanthdv.org/OfficeHome-Dataset/)
    - [VisDA17](http://ai.bu.edu/visda-2017/)
    - [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)-[STL10](https://cs.stanford.edu/~acoates/stl10/)
    - [Digit-5](https://github.com/VisionLearningGroup/VisionLearningGroup.github.io/tree/master/M3SDA/code_MSDA_digit#digit-five-download)
    - [DomainNet](http://ai.bu.edu/M3SDA/)
    - [miniDomainNet](https://arxiv.org/abs/2003.07325)

- Domain generalization
    - [PACS](https://arxiv.org/abs/1710.03077)
    - [VLCS](https://people.csail.mit.edu/torralba/publications/datasets_cvpr11.pdf)
    - [Office-Home](http://hemanthdv.org/OfficeHome-Dataset/)
    - [Digits-DG](https://arxiv.org/abs/2003.06054)
    - [Digit-Single](https://arxiv.org/abs/1805.12018)
    - [CIFAR-10-C](https://arxiv.org/abs/1807.01697)
    - [CIFAR-100-C](https://arxiv.org/abs/1807.01697)
    - [iWildCam-WILDS](https://wilds.stanford.edu/datasets/#iwildcam)
    - [Camelyon17-WILDS](https://wilds.stanford.edu/datasets/#camelyon17)
    - [FMoW-WILDS](https://wilds.stanford.edu/datasets/#fmow)

- Semi-supervised learning
    - [CIFAR10/100](https://www.cs.toronto.edu/~kriz/cifar.html.)
    - [SVHN](http://ufldl.stanford.edu/housenumbers/)
    - [STL10](https://cs.stanford.edu/~acoates/stl10/)

## Get started

### Installation

Make sure [conda](https://www.anaconda.com/distribution/) is installed properly.

```bash
# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Create a conda environment
conda create -y -n dassl python=3.8

# Activate the environment
conda activate dassl

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

Follow the instructions in [DATASETS.md](./DATASETS.md) to preprocess the datasets.

### Training

The main interface is implemented in `tools/train.py`, which basically does

1. initialize the config with `cfg = setup_cfg(args)` where `args` contains the command-line input (see `tools/train.py` for the list of input arguments);
2. instantiate a `trainer` with `build_trainer(cfg)` which loads the dataset and builds a deep neural network model;
3. call `trainer.train()` for training and evaluating the model.

Below we provide an example for training a source-only baseline on the popular domain adaptation dataset, Office-31,

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root $DATA \
--trainer SourceOnly \
--source-domains amazon \
--target-domains webcam \
--dataset-config-file configs/datasets/da/office31.yaml \
--config-file configs/trainers/da/source_only/office31.yaml \
--output-dir output/source_only_office31
```

`$DATA` denotes the location where datasets are installed. `--dataset-config-file` loads the common setting for the dataset (Office-31 in this case) such as image size and model architecture. `--config-file` loads the algorithm-specific setting such as hyper-parameters and optimization parameters.

To use multiple sources, namely the multi-source domain adaptation task, one just needs to add more sources to `--source-domains`. For instance, to train a source-only baseline on miniDomainNet, one can do

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root $DATA \
--trainer SourceOnly \
--source-domains clipart painting real \
--target-domains sketch \
--dataset-config-file configs/datasets/da/mini_domainnet.yaml \
--config-file configs/trainers/da/source_only/mini_domainnet.yaml \
--output-dir output/source_only_minidn
```

After the training finishes, the model weights will be saved under the specified output directory, along with a log file and a tensorboard file for visualization.

To print out the results saved in the log file (so you do not need to exhaustively go through all log files and calculate the mean/std by yourself), you can use `tools/parse_test_res.py`. The instruction can be found in the code.

For other trainers such as `MCD`, you can set `--trainer MCD` while keeping the config file unchanged, i.e. using the same training parameters as `SourceOnly` (in the simplest case). To modify the hyper-parameters in MCD, like `N_STEP_F` (number of steps to update the feature extractor), you can append `TRAINER.MCD.N_STEP_F 4` to the existing input arguments (otherwise the default value will be used). Alternatively, you can create a new `.yaml` config file to store your custom setting. See [here](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/config/defaults.py#L176) for a complete list of algorithm-specific hyper-parameters.

### Test
Model testing can be done by using `--eval-only`, which asks the code to run `trainer.test()`. You also need to provide the trained model and specify which model file (i.e. saved at which epoch) to use. For example, to use `model.pth.tar-20` saved at `output/source_only_office31/model`, you can do

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
--root $DATA \
--trainer SourceOnly \
--source-domains amazon \
--target-domains webcam \
--dataset-config-file configs/datasets/da/office31.yaml \
--config-file configs/trainers/da/source_only/office31.yaml \
--output-dir output/source_only_office31_test \
--eval-only \
--model-dir output/source_only_office31 \
--load-epoch 20
```

Note that `--model-dir` takes as input the directory path which was specified in `--output-dir` in the training stage.

### Write a new trainer
A good practice is to go through `dassl/engine/trainer.py` to get familar with the base trainer classes, which provide generic functions and training loops. To write a trainer class for domain adaptation or semi-supervised learning, the new class can subclass `TrainerXU`. For domain generalization, the new class can subclass `TrainerX`. In particular, `TrainerXU` and `TrainerX` mainly differ in whether using a data loader for unlabeled data. With the base classes, a new trainer may only need to implement the `forward_backward()` method, which performs loss computation and model update. See `dassl/enigne/da/source_only.py` for example.

### Add a new backbone/head/network
`backbone` corresponds to a convolutional neural network model which performs feature extraction. `head` (which is an optional module) is mounted on top of `backbone` for further processing, which can be, for example, a MLP. `backbone` and `head` are basic building blocks for constructing a `SimpleNet()` (see `dassl/engine/trainer.py`) which serves as the primary model for a task. `network` contains custom neural network models, such as an image generator.

To add a new module, namely a backbone/head/network, you need to first register the module using the corresponding `registry`, i.e. `BACKBONE_REGISTRY` for `backbone`, `HEAD_REGISTRY` for `head` and `NETWORK_RESIGTRY` for `network`. Note that for a new `backbone`, we require the model to subclass `Backbone` as defined in `dassl/modeling/backbone/backbone.py` and specify the `self._out_features` attribute.

We provide an example below for how to add a new `backbone`.
```python
from dassl.modeling import Backbone, BACKBONE_REGISTRY

class MyBackbone(Backbone):

    def __init__(self):
        super().__init__()
        # Create layers
        self.conv = ...

        self._out_features = 2048

    def forward(self, x):
        # Extract and return features

@BACKBONE_REGISTRY.register()
def my_backbone(**kwargs):
    return MyBackbone()
```
Then, you can set `MODEL.BACKBONE.NAME` to `my_backbone` to use your own architecture. For more details, please refer to the source code in `dassl/modeling`.

### Add a dataset
An example code structure is shown below. Make sure you subclass `DatasetBase` and register the dataset with `@DATASET_REGISTRY.register()`. All you need is to load `train_x`, `train_u` (optional), `val` (optional) and `test`, among which `train_u` and `val` could be `None` or simply ignored. Each of these variables contains a list of `Datum` objects. A `Datum` object (implemented [here](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/datasets/base_dataset.py#L12)) contains information for a single image, like `impath` (string) and `label` (int).

```python
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

@DATASET_REGISTRY.register()
class NewDataset(DatasetBase):

    dataset_dir = ''

    def __init__(self, cfg):
        
        train_x = ...
        train_u = ...  # optional, can be None
        val = ...  # optional, can be None
        test = ...

        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)
```

We suggest you take a look at the datasets code in some projects like [this](https://github.com/KaiyangZhou/CoOp), which is built on top of Dassl.

## Relevant Research

We would like to share here our research relevant to Dassl.

- [On-Device Domain Generalization](https://arxiv.org/abs/2209.07521)
- [Domain Generalization: A Survey](https://arxiv.org/abs/2103.02503) (TPAMI 2022)
- [Domain Adaptive Ensemble Learning](https://arxiv.org/abs/2003.07325) (TIP 2021)
- [MixStyle Neural Networks for Domain Generalization and Adaptation](https://arxiv.org/abs/2107.02053)
- [Semi-Supervised Domain Generalization with Stochastic StyleMatch](https://arxiv.org/abs/2106.00592)
- [Domain Generalization with MixStyle](https://openreview.net/forum?id=6xHJ37MVxxp) (ICLR 2021)
- [Learning to Generate Novel Domains for Domain Generalization](https://arxiv.org/abs/2007.03304) (ECCV 2020)
- [Deep Domain-Adversarial Image Generation for Domain Generalisation](https://arxiv.org/abs/2003.06054) (AAAI 2020)

## Citation

If you find this code useful to your research, please give credit to the following paper

```
@article{zhou2022domain,
  title={Domain generalization: A survey},
  author={Zhou, Kaiyang and Liu, Ziwei and Qiao, Yu and Xiang, Tao and Loy, Chen Change},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}

@article{zhou2021domain,
  title={Domain adaptive ensemble learning},
  author={Zhou, Kaiyang and Yang, Yongxin and Qiao, Yu and Xiang, Tao},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={8008--8018},
  year={2021},
  publisher={IEEE}
}
```
