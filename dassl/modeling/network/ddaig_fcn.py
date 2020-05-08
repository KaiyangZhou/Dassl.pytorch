"""
Credit to: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""
import functools
import torch
import torch.nn as nn
from torch.nn import functional as F

from .build import NETWORK_REGISTRY


def init_network_weights(model, init_type='normal', gain=0.02):

    def _init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (
            classname.find('Conv') != -1 or classname.find('Linear') != -1
        ):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method {} is not implemented'.
                    format(init_type)
                )
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('InstanceNorm2d') != -1:
            if m.weight is not None and m.bias is not None:
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)

    model.apply(_init_func)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % norm_type
        )
    return norm_layer


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(
        self, dim, padding_type, norm_layer, use_dropout, use_bias
    ):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type
            )

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type
            )
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class LocNet(nn.Module):
    """Localization network."""

    def __init__(
        self,
        input_nc,
        nc=32,
        n_blocks=3,
        use_dropout=False,
        padding_type='zero',
        image_size=32
    ):
        super().__init__()

        backbone = []
        backbone += [
            nn.Conv2d(
                input_nc, nc, kernel_size=3, stride=2, padding=1, bias=False
            )
        ]
        backbone += [nn.BatchNorm2d(nc)]
        backbone += [nn.ReLU(True)]
        for _ in range(n_blocks):
            backbone += [
                ResnetBlock(
                    nc,
                    padding_type=padding_type,
                    norm_layer=nn.BatchNorm2d,
                    use_dropout=use_dropout,
                    use_bias=False
                )
            ]
            backbone += [nn.MaxPool2d(2, stride=2)]
        self.backbone = nn.Sequential(*backbone)
        reduced_imsize = int(image_size * 0.5**(n_blocks + 1))
        self.fc_loc = nn.Linear(nc * reduced_imsize**2, 2 * 2)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc_loc(x)
        x = torch.tanh(x)
        x = x.view(-1, 2, 2)
        theta = x.data.new_zeros(x.size(0), 2, 3)
        theta[:, :, :2] = x
        return theta


class FCN(nn.Module):
    """Fully convolutional network."""

    def __init__(
        self,
        input_nc,
        output_nc,
        nc=32,
        n_blocks=3,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        padding_type='reflect',
        gctx=True,
        stn=False,
        image_size=32
    ):
        super().__init__()

        backbone = []

        p = 0
        if padding_type == 'reflect':
            backbone += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            backbone += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError
        backbone += [
            nn.Conv2d(
                input_nc, nc, kernel_size=3, stride=1, padding=p, bias=False
            )
        ]
        backbone += [norm_layer(nc)]
        backbone += [nn.ReLU(True)]

        for _ in range(n_blocks):
            backbone += [
                ResnetBlock(
                    nc,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=False
                )
            ]
        self.backbone = nn.Sequential(*backbone)

        # global context fusion layer
        self.gctx_fusion = None
        if gctx:
            self.gctx_fusion = nn.Sequential(
                nn.Conv2d(
                    2 * nc, nc, kernel_size=1, stride=1, padding=0, bias=False
                ), norm_layer(nc), nn.ReLU(True)
            )

        self.regress = nn.Sequential(
            nn.Conv2d(
                nc, output_nc, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.Tanh(),
        )

        self.locnet = None
        if stn:
            self.locnet = LocNet(
                input_nc, nc=nc, n_blocks=n_blocks, image_size=image_size
            )

    def init_loc_layer(self):
        """Initialize the weights/bias with identity transformation."""
        if self.locnet is not None:
            self.locnet.fc_loc.weight.data.zero_()
            self.locnet.fc_loc.bias.data.copy_(
                torch.tensor([1, 0, 0, 1], dtype=torch.float)
            )

    def stn(self, x):
        """Spatial transformer network."""
        theta = self.locnet(x)
        grid = F.affine_grid(theta, x.size())
        return F.grid_sample(x, grid), theta

    def forward(self, x, lmda=1., return_p=False, return_stn_output=False):
        """
        Args:
            x (torch.Tensor): input mini-batch.
            lmda (float): multiplier for perturbation.
            return_p (bool): return perturbation.
            return_stn_output (bool): return the output of stn.
        """
        theta = None
        if self.locnet is not None:
            x, theta = self.stn(x)
        input = x

        x = self.backbone(x)
        if self.gctx_fusion is not None:
            c = F.adaptive_avg_pool2d(x, (1, 1))
            c = c.expand_as(x)
            x = torch.cat([x, c], 1)
            x = self.gctx_fusion(x)

        p = self.regress(x)
        x_p = input + lmda*p

        if return_stn_output:
            return x_p, p, input

        if return_p:
            return x_p, p

        return x_p


@NETWORK_REGISTRY.register()
def fcn_3x32_gctx(**kwargs):
    norm_layer = get_norm_layer(norm_type='instance')
    net = FCN(3, 3, nc=32, n_blocks=3, norm_layer=norm_layer)
    init_network_weights(net, init_type='normal', gain=0.02)
    return net


@NETWORK_REGISTRY.register()
def fcn_3x64_gctx(**kwargs):
    norm_layer = get_norm_layer(norm_type='instance')
    net = FCN(3, 3, nc=64, n_blocks=3, norm_layer=norm_layer)
    init_network_weights(net, init_type='normal', gain=0.02)
    return net


@NETWORK_REGISTRY.register()
def fcn_3x32_gctx_stn(image_size=32, **kwargs):
    norm_layer = get_norm_layer(norm_type='instance')
    net = FCN(
        3,
        3,
        nc=32,
        n_blocks=3,
        norm_layer=norm_layer,
        stn=True,
        image_size=image_size
    )
    init_network_weights(net, init_type='normal', gain=0.02)
    net.init_loc_layer()
    return net


@NETWORK_REGISTRY.register()
def fcn_3x64_gctx_stn(image_size=224, **kwargs):
    norm_layer = get_norm_layer(norm_type='instance')
    net = FCN(
        3,
        3,
        nc=64,
        n_blocks=3,
        norm_layer=norm_layer,
        stn=True,
        image_size=image_size
    )
    init_network_weights(net, init_type='normal', gain=0.02)
    net.init_loc_layer()
    return net
