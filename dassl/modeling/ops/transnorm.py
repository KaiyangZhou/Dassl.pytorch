import torch
import torch.nn as nn


class _TransNorm(nn.Module):
    """Transferable normalization.

    Reference:
        - Wang et al. Transferable Normalization: Towards Improving
        Transferability of Deep Neural Networks. NeurIPS 2019.

    Args:
        num_features (int): number of features.
        eps (float): epsilon.
        momentum (float): value for updating running_mean and running_var.
        adaptive_alpha (bool): apply domain adaptive alpha.
    """

    def __init__(
        self, num_features, eps=1e-5, momentum=0.1, adaptive_alpha=True
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.adaptive_alpha = adaptive_alpha

        self.register_buffer("running_mean_s", torch.zeros(num_features))
        self.register_buffer("running_var_s", torch.ones(num_features))
        self.register_buffer("running_mean_t", torch.zeros(num_features))
        self.register_buffer("running_var_t", torch.ones(num_features))

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def resnet_running_stats(self):
        self.running_mean_s.zero_()
        self.running_var_s.fill_(1)
        self.running_mean_t.zero_()
        self.running_var_t.fill_(1)

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def _check_input(self, x):
        raise NotImplementedError

    def _compute_alpha(self, mean_s, var_s, mean_t, var_t):
        C = self.num_features
        ratio_s = mean_s / (var_s + self.eps).sqrt()
        ratio_t = mean_t / (var_t + self.eps).sqrt()
        dist = (ratio_s - ratio_t).abs()
        dist_inv = 1 / (1+dist)
        return C * dist_inv / dist_inv.sum()

    def forward(self, input):
        self._check_input(input)
        C = self.num_features
        if input.dim() == 2:
            new_shape = (1, C)
        elif input.dim() == 4:
            new_shape = (1, C, 1, 1)
        else:
            raise ValueError

        weight = self.weight.view(*new_shape)
        bias = self.bias.view(*new_shape)

        if not self.training:
            mean_t = self.running_mean_t.view(*new_shape)
            var_t = self.running_var_t.view(*new_shape)
            output = (input-mean_t) / (var_t + self.eps).sqrt()
            output = output*weight + bias

            if self.adaptive_alpha:
                mean_s = self.running_mean_s.view(*new_shape)
                var_s = self.running_var_s.view(*new_shape)
                alpha = self._compute_alpha(mean_s, var_s, mean_t, var_t)
                alpha = alpha.reshape(*new_shape)
                output = (1 + alpha.detach()) * output

            return output

        input_s, input_t = torch.split(input, input.shape[0] // 2, dim=0)

        x_s = input_s.transpose(0, 1).reshape(C, -1)
        mean_s = x_s.mean(1)
        var_s = x_s.var(1)
        self.running_mean_s.mul_(self.momentum)
        self.running_mean_s.add_((1 - self.momentum) * mean_s.data)
        self.running_var_s.mul_(self.momentum)
        self.running_var_s.add_((1 - self.momentum) * var_s.data)
        mean_s = mean_s.reshape(*new_shape)
        var_s = var_s.reshape(*new_shape)
        output_s = (input_s-mean_s) / (var_s + self.eps).sqrt()
        output_s = output_s*weight + bias

        x_t = input_t.transpose(0, 1).reshape(C, -1)
        mean_t = x_t.mean(1)
        var_t = x_t.var(1)
        self.running_mean_t.mul_(self.momentum)
        self.running_mean_t.add_((1 - self.momentum) * mean_t.data)
        self.running_var_t.mul_(self.momentum)
        self.running_var_t.add_((1 - self.momentum) * var_t.data)
        mean_t = mean_t.reshape(*new_shape)
        var_t = var_t.reshape(*new_shape)
        output_t = (input_t-mean_t) / (var_t + self.eps).sqrt()
        output_t = output_t*weight + bias

        output = torch.cat([output_s, output_t], 0)

        if self.adaptive_alpha:
            alpha = self._compute_alpha(mean_s, var_s, mean_t, var_t)
            alpha = alpha.reshape(*new_shape)
            output = (1 + alpha.detach()) * output

        return output


class TransNorm1d(_TransNorm):

    def _check_input(self, x):
        if x.dim() != 2:
            raise ValueError(
                "Expected the input to be 2-D, "
                "but got {}-D".format(x.dim())
            )


class TransNorm2d(_TransNorm):

    def _check_input(self, x):
        if x.dim() != 4:
            raise ValueError(
                "Expected the input to be 4-D, "
                "but got {}-D".format(x.dim())
            )
