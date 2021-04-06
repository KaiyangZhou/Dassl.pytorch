from yacs.config import CfgNode as CN

###########################
# Config definition
###########################

_C = CN()

_C.VERSION = 1

# Directory to save the output files
_C.OUTPUT_DIR = './output'
# Path to a directory where the files were saved
_C.RESUME = ''
# Set seed to negative value to randomize everything
# Set seed to positive value to use a fixed seed
_C.SEED = -1
_C.USE_CUDA = True
# Print detailed information (e.g. what trainer,
# dataset, backbone, etc.)
_C.VERBOSE = True

###########################
# Input
###########################
_C.INPUT = CN()
_C.INPUT.SIZE = (224, 224)
# For available choices please refer to transforms.py
_C.INPUT.TRANSFORMS = ()
# If True, tfm_train and tfm_test will be None
_C.INPUT.NO_TRANSFORM = False
# Default mean and std come from ImageNet
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Padding for random crop
_C.INPUT.CROP_PADDING = 4
# Cutout
_C.INPUT.CUTOUT_N = 1
_C.INPUT.CUTOUT_LEN = 16
# Gaussian noise
_C.INPUT.GN_MEAN = 0.
_C.INPUT.GN_STD = 0.15
# RandomAugment
_C.INPUT.RANDAUGMENT_N = 2
_C.INPUT.RANDAUGMENT_M = 10

###########################
# Dataset
###########################
_C.DATASET = CN()
# Directory where datasets are stored
_C.DATASET.ROOT = ''
_C.DATASET.NAME = ''
# List of names of source domains
_C.DATASET.SOURCE_DOMAINS = ()
# List of names of target domains
_C.DATASET.TARGET_DOMAINS = ()
# Number of labeled instances for the SSL setting
_C.DATASET.NUM_LABELED = 250
# Percentage of validation data (only used for SSL datasets)
# Set to 0 if do not want to use val data
# Using val data for hyperparameter tuning was done in Oliver et al. 2018
_C.DATASET.VAL_PERCENT = 0.1
# Fold index for STL-10 dataset (normal range is 0 - 9)
# Negative number means None
_C.DATASET.STL10_FOLD = -1
# CIFAR-10/100-C's corruption type and intensity level
_C.DATASET.CIFAR_C_TYPE = ''
_C.DATASET.CIFAR_C_LEVEL = 1
# Use all data in the unlabeled data set (e.g. FixMatch)
_C.DATASET.ALL_AS_UNLABELED = False

###########################
# Dataloader
###########################
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
# Apply transformations to an image K times (during training)
_C.DATALOADER.K_TRANSFORMS = 1
# Setting for train_x data-loader
_C.DATALOADER.TRAIN_X = CN()
_C.DATALOADER.TRAIN_X.SAMPLER = 'RandomSampler'
_C.DATALOADER.TRAIN_X.BATCH_SIZE = 32
# Parameter for RandomDomainSampler
# 0 or -1 means sampling from all domains
_C.DATALOADER.TRAIN_X.N_DOMAIN = 0

# Setting for train_u data-loader
_C.DATALOADER.TRAIN_U = CN()
# Set to false if you want to have unique
# data loader params for train_u
_C.DATALOADER.TRAIN_U.SAME_AS_X = True
_C.DATALOADER.TRAIN_U.SAMPLER = 'RandomSampler'
_C.DATALOADER.TRAIN_U.BATCH_SIZE = 32
_C.DATALOADER.TRAIN_U.N_DOMAIN = 0

# Setting for test data-loader
_C.DATALOADER.TEST = CN()
_C.DATALOADER.TEST.SAMPLER = 'SequentialSampler'
_C.DATALOADER.TEST.BATCH_SIZE = 32

###########################
# Model
###########################
_C.MODEL = CN()
# Path to model weights for initialization
_C.MODEL.INIT_WEIGHTS = ''
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = ''
_C.MODEL.BACKBONE.PRETRAINED = True
# Definition of embedding layer
_C.MODEL.HEAD = CN()
# If none, no embedding layer will be constructed
_C.MODEL.HEAD.NAME = ''
# Structure of hidden layers which is a list, e.g. [512, 512]
# If not defined, no embedding layer will be constructed
_C.MODEL.HEAD.HIDDEN_LAYERS = ()
_C.MODEL.HEAD.ACTIVATION = 'relu'
_C.MODEL.HEAD.BN = True
_C.MODEL.HEAD.DROPOUT = 0.

###########################
# Optimization
###########################
_C.OPTIM = CN()
_C.OPTIM.NAME = 'adam'
_C.OPTIM.LR = 0.0003
_C.OPTIM.WEIGHT_DECAY = 5e-4
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.SGD_DAMPNING = 0
_C.OPTIM.SGD_NESTEROV = False
_C.OPTIM.RMSPROP_ALPHA = 0.99
_C.OPTIM.ADAM_BETA1 = 0.9
_C.OPTIM.ADAM_BETA2 = 0.99
# STAGED_LR allows different layers to have
# different lr, e.g. pre-trained base layers
# can be assigned a smaller lr than the new
# classification layer
_C.OPTIM.STAGED_LR = False
_C.OPTIM.NEW_LAYERS = ()
_C.OPTIM.BASE_LR_MULT = 0.1
# Learning rate scheduler
_C.OPTIM.LR_SCHEDULER = 'single_step'
_C.OPTIM.STEPSIZE = (10, )
_C.OPTIM.GAMMA = 0.1
_C.OPTIM.MAX_EPOCH = 10

###########################
# Train
###########################
_C.TRAIN = CN()
# How often (epoch) to save model during training
# Set to 0 or negative value to disable
_C.TRAIN.CHECKPOINT_FREQ = 0
# How often (batch) to print training information
_C.TRAIN.PRINT_FREQ = 10
# Use 'train_x', 'train_u' or 'smaller_one' to count
# the number of iterations in an epoch (for DA and SSL)
_C.TRAIN.COUNT_ITER = 'train_x'

###########################
# Test
###########################
_C.TEST = CN()
_C.TEST.EVALUATOR = 'Classification'
_C.TEST.PER_CLASS_RESULT = False
# Compute confusion matrix, which will be saved
# to $OUTPUT_DIR/cmat.pt
_C.TEST.COMPUTE_CMAT = False
# If NO_TEST=True, no testing will be conducted
_C.TEST.NO_TEST = False
# How often (epoch) to do testing during training
# Set to 0 or negative value to disable
_C.TEST.EVAL_FREQ = 1
# Use 'test' set or 'val' set for evaluation
_C.TEST.SPLIT = 'test'

###########################
# Trainer specifics
###########################
_C.TRAINER = CN()
_C.TRAINER.NAME = ''

# MCD
_C.TRAINER.MCD = CN()
_C.TRAINER.MCD.N_STEP_F = 4
# MME
_C.TRAINER.MME = CN()
_C.TRAINER.MME.LMDA = 0.1
# SelfEnsembling
_C.TRAINER.SE = CN()
_C.TRAINER.SE.EMA_ALPHA = 0.999
_C.TRAINER.SE.CONF_THRE = 0.95
_C.TRAINER.SE.RAMPUP = 300

# M3SDA
_C.TRAINER.M3SDA = CN()
_C.TRAINER.M3SDA.LMDA = 0.5
_C.TRAINER.M3SDA.N_STEP_F = 4
# DAEL
_C.TRAINER.DAEL = CN()
_C.TRAINER.DAEL.WEIGHT_U = 0.5
_C.TRAINER.DAEL.CONF_THRE = 0.95
_C.TRAINER.DAEL.STRONG_TRANSFORMS = ()

# CrossGrad
_C.TRAINER.CG = CN()
_C.TRAINER.CG.EPS_F = 1.
_C.TRAINER.CG.EPS_D = 1.
_C.TRAINER.CG.ALPHA_F = 0.5
_C.TRAINER.CG.ALPHA_D = 0.5
# DDAIG
_C.TRAINER.DDAIG = CN()
_C.TRAINER.DDAIG.G_ARCH = ''
_C.TRAINER.DDAIG.LMDA = 0.3
_C.TRAINER.DDAIG.CLAMP = False
_C.TRAINER.DDAIG.CLAMP_MIN = -1.
_C.TRAINER.DDAIG.CLAMP_MAX = 1.
_C.TRAINER.DDAIG.WARMUP = 0
_C.TRAINER.DDAIG.ALPHA = 0.5

# EntMin
_C.TRAINER.ENTMIN = CN()
_C.TRAINER.ENTMIN.LMDA = 1e-3
# Mean Teacher
_C.TRAINER.MEANTEA = CN()
_C.TRAINER.MEANTEA.WEIGHT_U = 1.
_C.TRAINER.MEANTEA.EMA_ALPHA = 0.999
# MixMatch
_C.TRAINER.MIXMATCH = CN()
_C.TRAINER.MIXMATCH.WEIGHT_U = 100.
_C.TRAINER.MIXMATCH.TEMP = 2.
_C.TRAINER.MIXMATCH.MIXUP_BETA = 0.75
_C.TRAINER.MIXMATCH.RAMPUP = 20000
# FixMatch
_C.TRAINER.FIXMATCH = CN()
_C.TRAINER.FIXMATCH.WEIGHT_U = 1.
_C.TRAINER.FIXMATCH.CONF_THRE = 0.95
_C.TRAINER.FIXMATCH.STRONG_TRANSFORMS = ()
