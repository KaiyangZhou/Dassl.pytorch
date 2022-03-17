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
- Mar 2022: A new domain generalization method [EFDM](https://arxiv.org/abs/2203.07740) developed by [Yabin Zhang (PolyU)](https://ybzh.github.io/) and to appear at CVPR'22 is added to this repo. See [here](https://github.com/KaiyangZhou/Dassl.pytorch/pull/36) for more details.
- Feb 2022: In case you don't know, a class in the painting domain of DomainNet (the official splits) only has test images (no training images), which could affect performance. See section 4.a in our [paper](https://arxiv.org/abs/2003.07325) for more details.
- Oct 2021: `v0.5.0`: **Important changes** made to `transforms.py`. 1) `center_crop` becomes a default transform in testing (applied after resizing the smaller edge to a certain size to keep the image aspect ratio). 2) For training, `Resize(cfg.INPUT.SIZE)` is deactivated when `random_crop` or `random_resized_crop` is used. These changes won't make any difference to the training transforms used in existing config files, nor to the testing transforms unless the raw images are not squared (the only difference is that now the image aspect ratio is respected).
- Oct 2021: `v0.4.3`: Copy the attributes in `self.dm` (data manager) to `SimpleTrainer` and make `self.dm` optional, which means from now on, you can build data loaders from any source you like rather than being forced to use `DataManager`.
- Sep 2021: `v0.4.2`: An important update is to set `drop_last=is_train and len(data_source)>=batch_size` when constructing a data loader to avoid 0-length.

<details>
    <summary>More</summary>

- Aug 2021: `v0.4.0`: The most noteworthy update is adding the learning rate warmup scheduler. The implementation is detailed [here](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/optim/lr_scheduler.py#L10) and the config variables are specified [here](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/config/defaults.py#L171).
- Jul 2021: `v0.3.4`: Adds a new function `generate_fewshot_dataset()` to the base dataset class, which allows for the generation of a few-shot learning setting. One can customize a few-shot dataset by specifying `_C.DATASET.NUM_SHOTS` and give it to `generate_fewshot_dataset()`.
- Jul 2021: `v0.3.2`: Adds `_C.INPUT.INTERPOLATION` (default: `bilinear`). Available interpolation modes are `bilinear`, `nearest`, and `bicubic`.
- Jul 2021 `v0.3.1`: Now you can use `*.register(force=True)` to replace previously registered modules.
- Jul 2021 `v0.3.0`: Allows to deploy the model with the best validation performance for final test (for the purpose of model selection). Specifically, a new config variable named `_C.TEST.FINAL_MODEL` is introduced, which takes either `"last_step"` (default) or `"best_val"`. When set to `"best_val"`, the model will be evaluated on the `val` set after each epoch and the one with the best validation performance will be saved and used for final test (see this [code](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/engine/trainer.py#L412)).
- Jul 2021 `v0.2.7`: Adds attribute `classnames` to the base dataset class. Now you can get a list of class names ordered by numeric labels by calling `trainer.dm.dataset.classnames`.
- Jun 2021 `v0.2.6`: Merges `MixStyle2` to `MixStyle`. A new variable `self.mix` is used to switch between random mixing and cross-domain mixing. Please see [this](https://github.com/KaiyangZhou/Dassl.pytorch/issues/23) for more details on the new features.
- Jun 2021 `v0.2.5`: Fixs a [bug](https://github.com/KaiyangZhou/Dassl.pytorch/commit/29881c7faee7405f80f5f674de4bbbf80d5dc77a) in the calculation of per-class recognition accuracy.
- Jun 2021 `v0.2.4`: Adds `extend_cfg(cfg)` to `train.py`. This function is particularly useful when you build your own methods on top of Dassl.pytorch and need to define some custom variables. Please see the repository [mixstyle-release](https://github.com/KaiyangZhou/mixstyle-release) or [ssdg-benchmark](https://github.com/KaiyangZhou/ssdg-benchmark) for examples.
- Jun 2021 New benchmarks for semi-supervised domain generalization at https://github.com/KaiyangZhou/ssdg-benchmark.
- Apr 2021 Do you know you can use `tools/parse_test_res.py` to read the log files and automatically calculate and print out the results including mean and standard deviation? Check the instructions in `tools/parse_test_res.py` for more details.
- Apr 2021 `v0.2.3`: A [MixStyle](https://openreview.net/forum?id=6xHJ37MVxxp) layer can now be deactivated or activated by using `model.apply(deactivate_mixstyle)` or `model.apply(activate_mixstyle)` without modifying the source code. See [dassl/modeling/ops/mixstyle.py](dassl/modeling/ops/mixstyle.py) for the details.
- Apr 2021 `v0.2.2`: Adds `RandomClassSampler`, which samples from a certain number of classes a certain number of images to form a minibatch (the code is modified from [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)).
- Apr 2021 `v0.2.1`: Slightly adjusts the ordering in `setup_cfg()` (see `tools/train.py`).
- Apr 2021 `v0.2.0`: Adds `_C.DATASET.ALL_AS_UNLABELED` (for the SSL setting) to the config variable list. When this variable is set to `True`, all labeled data will be included in the unlabeled data set.
- Apr 2021 `v0.1.9`: Adds [VLCS](https://people.csail.mit.edu/torralba/publications/datasets_cvpr11.pdf) to the benchmark datasets (see `dassl/data/datasets/dg/vlcs.py`).
- Mar 2021 `v0.1.8`: Allows `optim` and `sched` to be `None` in `register_model()`.
- Mar 2021 `v0.1.7`: Adds [MixStyle](https://openreview.net/forum?id=6xHJ37MVxxp) models to [dassl/modeling/backbone/resnet.py](dassl/modeling/backbone/resnet.py). The training configs in `configs/trainers/dg/vanilla` can be used to train MixStyle models.
- Mar 2021 `v0.1.6`: Adds [CIFAR-10/100-C](https://arxiv.org/abs/1807.01697) to the benchmark datasets for evaluating a model's robustness to image corruptions.
- Mar 2021 We have just released a survey on domain generalization at https://arxiv.org/abs/2103.02503, which summarizes the ten-year development in this topic with coverage on the history, related problems, datasets, methodologies, potential directions, and so on.
- Jan 2021 Our recent work, [MixStyle](https://openreview.net/forum?id=6xHJ37MVxxp) (mixing instance-level feature statistics of samples of different domains for improving domain generalization), is accepted to ICLR'21. The code is available at https://github.com/KaiyangZhou/mixstyle-release where the cross-domain image classification part is based on Dassl.pytorch.
- May 2020 `v0.1.3`: Adds the `Digit-Single` dataset for benchmarking single-source DG methods. The corresponding CNN model is [dassl/modeling/backbone/cnn_digitsingle.py](dassl/modeling/backbone/cnn_digitsingle.py) and the dataset config file is [configs/datasets/dg/digit_single.yaml](configs/datasets/dg/digit_single.yaml). See [Volpi et al. NIPS'18](https://arxiv.org/abs/1805.12018) for how to do evaluation.
- May 2020 `v0.1.2`: 1) Adds [EfficientNet](https://arxiv.org/abs/1905.11946) models (B0-B7) (credit to https://github.com/lukemelas/EfficientNet-PyTorch). To use EfficientNet, set `MODEL.BACKBONE.NAME` to `efficientnet_b{N}` where `N={0, ..., 7}`. 2) `dassl/modeling/models` is renamed to `dassl/modeling/network` (`build_model()` to `build_network()` and `MODEL_REGISTRY` to `NETWORK_RESIGTRY`).

</details>

## Overview

Dassl has implemented the following methods:

- Single-source domain adaptation
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
conda create -n dassl python=3.7

# Activate the environment
conda activate dassl

# Install dependencies
pip install -r requirements.txt

# Install torch (version >= 1.7.1) and torchvision
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

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

- [Domain Adaptive Ensemble Learning](https://arxiv.org/abs/2003.07325), TIP, 2021.
- [MixStyle Neural Networks for Domain Generalization and Adaptation](https://arxiv.org/abs/2107.02053), arxiv preprint, 2021.
- [Semi-Supervised Domain Generalization with Stochastic StyleMatch](https://arxiv.org/abs/2106.00592), arxiv preprint, 2021.
- [Domain Generalization in Vision: A Survey](https://arxiv.org/abs/2103.02503), arxiv preprint, 2021.
- [Domain Generalization with MixStyle](https://openreview.net/forum?id=6xHJ37MVxxp), in ICLR 2021.
- [Learning to Generate Novel Domains for Domain Generalization](https://arxiv.org/abs/2007.03304), in ECCV 2020.
- [Deep Domain-Adversarial Image Generation for Domain Generalisation](https://arxiv.org/abs/2003.06054), in AAAI 2020.

## Citation

If you find this code useful to your research, please give credit to the following paper

```
@article{zhou2020domain,
  title={Domain Adaptive Ensemble Learning},
  author={Zhou, Kaiyang and Yang, Yongxin and Qiao, Yu and Xiang, Tao},
  journal={IEEE Transactions on Image Processing (TIP)},
  year={2021}
}
```
