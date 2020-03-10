from .defaults import _C as cfg_default


def get_cfg_default():
    return cfg_default.clone()
