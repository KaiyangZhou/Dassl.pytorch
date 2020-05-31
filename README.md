# Dassl

Dassl is a [PyTorch](https://pytorch.org) toolbox for domain adaptation and semi-supervised learning. It has a modular design and unified interfaces, allowing fast prototyping and experimentation. With Dassl, a new method can be implemented with only a few lines of code.

Besides the efforts to facilitate algorithm development and push state of the art, Dassl is also aimed at providing a uniform benchmarking platform, which allows methods to be evaluated on a common ground using the same set of environment and parameters.

You can use Dassl as a library for the following research:

- Domain adaptation
- Domain generalization
- Semi-supervised learning

## What's new
- [May 2020] `v0.1.3`: Added the `Digit-Single` dataset for benchmarking single-source DG methods. The corresponding CNN model is [dassl/modeling/backbone/cnn_digitsingle.py](dassl/modeling/backbone/cnn_digitsingle.py) and the dataset config file is [configs/datasets/dg/digit_single.yaml](configs/datasets/dg/digit_single.yaml). See [Volpi et al. NIPS'18](https://arxiv.org/abs/1805.12018) for how to evaluate your method.
- [May 2020] `v0.1.2`: 1) Added [EfficientNet](https://arxiv.org/abs/1905.11946) models (B0-B7) (credit to https://github.com/lukemelas/EfficientNet-PyTorch). To use EfficientNet, set `MODEL.BACKBONE.NAME` to `efficientnet_b{N}` where `N={0, ..., 7}`. 2) `dassl/modeling/models` has been renamed to `dassl/modeling/network`, including the `build_model()` method changed to `build_network()` and the `MODEL_REGISTRY` to `NETWORK_RESIGTRY`.

## Overview

Dassl has implemented the following papers:

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
    - [Deep Domain-Adversarial Image Generation for Domain Generalisation (AAAI'20)](https://arxiv.org/abs/2003.06054) [[dassl/engine/dg/ddaig.py](dassl/engine/dg/ddaig.py)]
    - [Generalizing Across Domains via Cross-Gradient Training (ICLR'18)](https://arxiv.org/abs/1804.10745) [[dassl/engine/dg/crossgrad.py](dassl/engine/dg/crossgrad.py)]

- Semi-supervised learning
    - [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685) [[dassl/engine/ssl/fixmatch.py](dassl/engine/ssl/fixmatch.py)]
    - [MixMatch: A Holistic Approach to Semi-Supervised Learning (NeurIPS'19)](https://arxiv.org/abs/1905.02249) [[dassl/engine/ssl/mixmatch.py](dassl/engine/ssl/mixmatch.py)]
    - [Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results (NeurIPS'17)](https://arxiv.org/abs/1703.01780) [[dassl/engine/ssl/mean_teacher.py](dassl/engine/ssl/mean_teacher.py)]
    - [Semi-supervised Learning by Entropy Minimization (NeurIPS'04)](http://papers.nips.cc/paper/2740-semi-supervised-learning-by-entropy-minimization.pdf) [[dassl/engine/ssl/entmin.py](dassl/engine/ssl/entmin.py)]

Dassl supports the following datasets.

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
    - [Office-Home](http://hemanthdv.org/OfficeHome-Dataset/)
    - [Digits-DG](https://arxiv.org/abs/2003.06054)
    - [Digit-Single](https://arxiv.org/abs/1805.12018)

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

# Install torch and torchvision (select a version that suits your machine)
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

Follow the instructions in [DATASETS.md](./DATASETS.md) to install the datasets.

### Training

The main interface is implemented in `tools/train.py`, which basically does three things:

1. Initialize the config with `cfg = setup_cfg(args)` where `args` contains the command-line input (see `tools/train.py` for the list of input arguments).
2. Instantiate a `trainer` with `build_trainer(cfg)` which loads the dataset and builds a deep neural network model.
3. Call `trainer.train()` for training and evaluating the model.

Below we provide an example for training a source-only baseline on the popular domain adaptation dataset -- Office-31,

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

`$DATA` denotes the location where datasets are installed. `--dataset-config-file` loads the common setting for the dataset such as image size and model architecture. `--config-file` loads the algorithm-specific setting such as hyper-parameters and optimization parameters.

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

### Test
Testing can be achieved by using `--eval-only`, which tells the script to run `trainer.test()`. You also need to provide the trained model and specify which model file (i.e. saved at which epoch) to use. For example, to use `model.pth.tar-20` saved at `output/source_only_office31/model`, you can do

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

## Citation
Please cite the following paper if you find Dassl useful to your research.

```
@article{zhou2020domain,
  title={Domain Adaptive Ensemble Learning},
  author={Zhou, Kaiyang and Yang, Yongxin and Qiao, Yu and Xiang, Tao},
  journal={arXiv preprint arXiv:2003.07325},
  year={2020}
}
```