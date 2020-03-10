import torch.nn as nn


class Sequential2(nn.Sequential):
    """An alternative sequential container to nn.Sequential,
    which accepts an arbitrary number of input arguments.
    """

    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
