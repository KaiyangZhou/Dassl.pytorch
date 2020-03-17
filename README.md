# Dassl

Dassl is a research toolbox for domain adaptation and semi-supervised learning, written in [PyTorch](https://pytorch.org).

It is designed for the following tasks:

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
    - [Domain Aadaptive Ensemble Learning](https://arxiv.org/abs/2003.07325)
    - [Moment Matching for Multi-Source Domain Adaptation (ICCV'19)](https://arxiv.org/abs/1812.01754)

- Domain generalization
    - [Deep Domain-Adversarial Image Generation for Domain Generalisation (AAAI'20)](https://arxiv.org/abs/2003.06054)
    - [Generalizing Across Domains via Cross-Gradient Training (ICLR'18)](https://arxiv.org/abs/1804.10745)

- Semi-supervised learning
    - [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)
    - [MixMatch: A Holistic Approach to Semi-Supervised Learning (NeurIPS'19)](https://arxiv.org/abs/1905.02249)
    - [Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results (NeurIPS'17)](https://arxiv.org/abs/1703.01780)
    - [Semi-supervised Learning by Entropy Minimization (NeurIPS'04)](http://papers.nips.cc/paper/2740-semi-supervised-learning-by-entropy-minimization.pdf)

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

Follow the instructions in [DATASETS.md](./DATASETS.md) to prepare the datasets.

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

`$DATA` denotes the path to the dataset folder. `--dataset-config-file` loads the common setting for the dataset such as image size and model architecture. `--config-file` loads the algorithm-specific setting such as hyper-parameters and optimization parameters.

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

### Write a new trainer

A good practice is to go through `dassl/engine/trainer.py` to get familar with the base trainer classes, which provide generic functions and training loops. To write a trainer class for domain adaptation or semi-supervised learning, the new class can subclass `TrainerXU`. For domain generalization, the new class can subclass `TrainerX`. In particular, `TrainerXU` and `TrainerX` mainly differ in whether using a data loader for unlabeled data. With the base classes, a new trainer may only need to implement the `forward_backward()` method, which performs loss computation and model update. See `dassl/enigne/da/source_only.py` for example.

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