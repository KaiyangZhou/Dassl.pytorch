# Dassl

Dassl is a research library for domain adaptation and semi-supervised learning, written in [PyTorch](https://pytorch.org).

It is developed for the following tasks:

- Single-source domain adaptation
- Multi-source domain adaptation
- Domain generalization
- Semi-supervised learning

## Overview

Dassl has implemented the following papers:

- Single-source domain adaptation
    - [Semi-supervised Domain Adaptation via Minimax Entropy (ICCV'19)](https://arxiv.org/abs/1904.06487)
    - [Maximum Classifier Discrepancy for Unsupervised Domain Adaptation (CVPR'18)](https://arxiv.org/abs/1712.02560https://arxiv.org/abs/1712.02560)
    - [Self-ensembling for visual domain adaptation (ICLR'18)](https://arxiv.org/abs/1706.05208)
    - [Revisiting Batch Normalization For Practical Domain Adaptation (ICLR-W'17)](https://arxiv.org/abs/1603.04779)
    - [Adversarial Discriminative Domain Adaptation (CVPR'17)](https://arxiv.org/abs/1702.05464)
    - [Domain-Adversarial Training of Neural Networks (JMLR'16) ](https://arxiv.org/abs/1505.07818)

- Multi-source domain adaptation
    - [Domain Aadaptive Ensemble Learning]()
    - [Moment Matching for Multi-Source Domain Adaptation (ICCV'19)](https://arxiv.org/abs/1812.01754)

- Domain generalization
    - [Generalizing Across Domains via Cross-Gradient Training (ICLR'18)](https://arxiv.org/abs/1804.10745)

- Semi-supervised learning
    - [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)
    - [MixMatch: A Holistic Approach to Semi-Supervised Learning (NeurIPS'19)](https://arxiv.org/abs/1905.02249)
    - [Semi-supervised Learning by Entropy Minimization (NeurIPS'04)](http://papers.nips.cc/paper/2740-semi-supervised-learning-by-entropy-minimization.pdf)

Dassl supports the following datasets.

- Domain adaptation
    - [Office-31](https://scalable.mpi-inf.mpg.de/files/2013/04/saenko_eccv_2010.pdf)
    - [Office-Home](http://hemanthdv.org/OfficeHome-Dataset/)
    - [VisDA17](http://ai.bu.edu/visda-2017/)
    - [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)-[STL10](https://cs.stanford.edu/~acoates/stl10/)
    - Digit-5
    - [DomainNet](http://ai.bu.edu/M3SDA/)
    - miniDomainNet

- Domain generalization
    - [PACS](https://arxiv.org/abs/1710.03077)
    - [Office-Home](http://hemanthdv.org/OfficeHome-Dataset/)
    - Digits-DG

- Semi-supervised learning
    - [CIFAR10/100](https://www.cs.toronto.edu/~kriz/cifar.html.)
    - [SVHN](http://ufldl.stanford.edu/housenumbers/)
    - [STL10](https://cs.stanford.edu/~acoates/stl10/)

## Installation

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

## Quick Start

The main interface is implemented in `tools/train.py`, which basically does three things:

1. Initialize config with `cfg = setup_cfg(args)` where `args` contains command-line input.
2. Instantiate a `trainer` with `build_trainer(cfg)`.
3. Call `trainer.train()` for training and evaluating a model.

Below we provide an example for training a source-only baseline on a domain adaptation dataset,

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

`$DATA` denotes the path to the dataset folder.

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