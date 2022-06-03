import torch.nn as nn

from .attention import Attention

__all__ = ["Conv2dDynamic"]


class Conv2dDynamic(nn.Module):
    """Conv2dDynamic from `"Dynamic Domain Generalization" <https://github.com/MetaVisionLab/DDG>`_.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool = True,
        squeeze: int = None,
        attention_in_channels: int = None
    ) -> None:
        super(Conv2dDynamic, self).__init__()

        if kernel_size // 2 != padding:
            # Only when this condition is met, we can ensure that different
            # kernel_size can obtain feature maps of consistent size.
            # Let I, K, S, P, O: O = (I + 2P - K) // S + 1, if P = K // 2, then O = (I - K % 2) // S + 1
            # This means that the output of two different Ks with the same parity can be made the same by adjusting P.
            raise ValueError("`padding` must be equal to `kernel_size // 2`.")
        if kernel_size % 2 == 0:
            raise ValueError(
                "Kernel_size must be odd now because the templates we used are odd (kernel_size=1)."
            )

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.kernel_templates = nn.ModuleDict()
        self.kernel_templates["conv_nn"] = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=min(in_channels, out_channels),
            bias=bias
        )
        self.kernel_templates["conv_11"] = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=bias
        )
        self.kernel_templates["conv_n1"] = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=stride,
            padding=(padding, 0),
            bias=bias
        )
        self.kernel_templates["conv_1n"] = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=stride,
            padding=(0, padding),
            bias=bias
        )
        self.attention = Attention(
            attention_in_channels if attention_in_channels else in_channels,
            4,
            squeeze,
            bias=bias
        )

    def forward(self, x, attention_x=None):
        attention_x = x if attention_x is None else attention_x
        y = self.attention(attention_x)

        out = self.conv(x)

        for i, template in enumerate(self.kernel_templates):
            out += self.kernel_templates[template](x) * y[:,
                                                          i].view(-1, 1, 1, 1)

        return out
