"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import pickle
import shutil
import os.path as osp
import warnings
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn

from .tools import mkdir_if_missing

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "resume_from_checkpoint",
    "open_all_layers",
    "open_specified_layers",
    "count_num_param",
    "load_pretrained_weights",
    "init_network_weights",
]


def save_checkpoint(
    state,
    save_dir,
    is_best=False,
    remove_module_from_keys=True,
    model_name=""
):
    r"""Save checkpoint.

    Args:
        state (dict): dictionary.
        save_dir (str): directory to save checkpoint.
        is_best (bool, optional): if True, this checkpoint will be copied and named
            ``model-best.pth.tar``. Default is False.
        remove_module_from_keys (bool, optional): whether to remove "module."
            from layer names. Default is True.
        model_name (str, optional): model name to save.

    Examples::
        >>> state = {
        >>>     'state_dict': model.state_dict(),
        >>>     'epoch': 10,
        >>>     'optimizer': optimizer.state_dict()
        >>> }
        >>> save_checkpoint(state, 'log/my_model')
    """
    mkdir_if_missing(save_dir)

    if remove_module_from_keys:
        # remove 'module.' in state_dict's keys
        state_dict = state["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        state["state_dict"] = new_state_dict

    # save model
    epoch = state["epoch"]
    if not model_name:
        model_name = "model.pth.tar-" + str(epoch)
    fpath = osp.join(save_dir, model_name)
    torch.save(state, fpath)
    print('Checkpoint saved to "{}"'.format(fpath))

    # save current model name
    checkpoint_file = osp.join(save_dir, "checkpoint")
    checkpoint = open(checkpoint_file, "w+")
    checkpoint.write("{}\n".format(osp.basename(fpath)))
    checkpoint.close()

    if is_best:
        best_fpath = osp.join(osp.dirname(fpath), "model-best.pth.tar")
        shutil.copy(fpath, best_fpath)
        print('Best checkpoint saved to "{}"'.format(best_fpath))


def load_checkpoint(fpath):
    r"""Load checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")

    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))

    map_location = None if torch.cuda.is_available() else "cpu"

    try:
        checkpoint = torch.load(fpath, map_location=map_location)

    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )

    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise

    return checkpoint


def resume_from_checkpoint(fdir, model, optimizer=None, scheduler=None):
    r"""Resume training from a checkpoint.

    This will load (1) model weights and (2) ``state_dict``
    of optimizer if ``optimizer`` is not None.

    Args:
        fdir (str): directory where the model was saved.
        model (nn.Module): model.
        optimizer (Optimizer, optional): an Optimizer.
        scheduler (Scheduler, optional): an Scheduler.

    Returns:
        int: start_epoch.

    Examples::
        >>> fdir = 'log/my_model'
        >>> start_epoch = resume_from_checkpoint(fdir, model, optimizer, scheduler)
    """
    with open(osp.join(fdir, "checkpoint"), "r") as checkpoint:
        model_name = checkpoint.readlines()[0].strip("\n")
        fpath = osp.join(fdir, model_name)

    print('Loading checkpoint from "{}"'.format(fpath))
    checkpoint = load_checkpoint(fpath)
    model.load_state_dict(checkpoint["state_dict"])
    print("Loaded model weights")

    if optimizer is not None and "optimizer" in checkpoint.keys():
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Loaded optimizer")

    if scheduler is not None and "scheduler" in checkpoint.keys():
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded scheduler")

    start_epoch = checkpoint["epoch"]
    print("Previous epoch: {}".format(start_epoch))

    return start_epoch


def adjust_learning_rate(
    optimizer,
    base_lr,
    epoch,
    stepsize=20,
    gamma=0.1,
    linear_decay=False,
    final_lr=0,
    max_epoch=100,
):
    r"""Adjust learning rate.

    Deprecated.
    """
    if linear_decay:
        # linearly decay learning rate from base_lr to final_lr
        frac_done = epoch / max_epoch
        lr = frac_done*final_lr + (1.0-frac_done) * base_lr
    else:
        # decay learning rate by gamma for every stepsize
        lr = base_lr * (gamma**(epoch // stepsize))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def set_bn_to_eval(m):
    r"""Set BatchNorm layers to eval mode."""
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()


def open_all_layers(model):
    r"""Open all layers in model for training.

    Examples::
        >>> open_all_layers(model)
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True


def open_specified_layers(model, open_layers):
    r"""Open specified layers in model for training while keeping
    other layers frozen.

    Args:
        model (nn.Module): neural net model.
        open_layers (str or list): layers open for training.

    Examples::
        >>> # Only model.classifier will be updated.
        >>> open_layers = 'classifier'
        >>> open_specified_layers(model, open_layers)
        >>> # Only model.fc and model.classifier will be updated.
        >>> open_layers = ['fc', 'classifier']
        >>> open_specified_layers(model, open_layers)
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    if isinstance(open_layers, str):
        open_layers = [open_layers]

    for layer in open_layers:
        assert hasattr(
            model, layer
        ), '"{}" is not an attribute of the model, please provide the correct name'.format(
            layer
        )

    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def count_num_param(model):
    r"""Count number of parameters in a model.

    Args:
        model (nn.Module): network model.

    Examples::
        >>> model_size = count_num_param(model)
    """
    return sum(p.numel() for p in model.parameters())


def load_pretrained_weights(model, weight_path):
    r"""Load pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            "please check the key names manually "
            "(** ignored and continue **)".format(weight_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
            format(weight_path)
        )
        if len(discarded_layers) > 0:
            print(
                "** The following layers are discarded "
                "due to unmatched keys or layer size: {}".
                format(discarded_layers)
            )


def init_network_weights(model, init_type="normal", gain=0.02):

    def _init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method {} is not implemented".
                    format(init_type)
                )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find("BatchNorm") != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

        elif classname.find("InstanceNorm") != -1:
            if m.weight is not None and m.bias is not None:
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)

    model.apply(_init_func)
