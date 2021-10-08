import torch
import torch.nn as nn
from torch.nn import functional as F


class MaximumMeanDiscrepancy(nn.Module):

    def __init__(self, kernel_type="rbf", normalize=False):
        super().__init__()
        self.kernel_type = kernel_type
        self.normalize = normalize

    def forward(self, x, y):
        # x, y: two batches of data with shape (batch, dim)
        # MMD^2(x, y) = k(x, x') - 2k(x, y) + k(y, y')
        if self.normalize:
            x = F.normalize(x, dim=1)
            y = F.normalize(y, dim=1)
        if self.kernel_type == "linear":
            return self.linear_mmd(x, y)
        elif self.kernel_type == "poly":
            return self.poly_mmd(x, y)
        elif self.kernel_type == "rbf":
            return self.rbf_mmd(x, y)
        else:
            raise NotImplementedError

    def linear_mmd(self, x, y):
        # k(x, y) = x^T y
        k_xx = self.remove_self_distance(torch.mm(x, x.t()))
        k_yy = self.remove_self_distance(torch.mm(y, y.t()))
        k_xy = torch.mm(x, y.t())
        return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

    def poly_mmd(self, x, y, alpha=1.0, c=2.0, d=2):
        # k(x, y) = (alpha * x^T y + c)^d
        k_xx = self.remove_self_distance(torch.mm(x, x.t()))
        k_xx = (alpha*k_xx + c).pow(d)
        k_yy = self.remove_self_distance(torch.mm(y, y.t()))
        k_yy = (alpha*k_yy + c).pow(d)
        k_xy = torch.mm(x, y.t())
        k_xy = (alpha*k_xy + c).pow(d)
        return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

    def rbf_mmd(self, x, y):
        # k_xx
        d_xx = self.euclidean_squared_distance(x, x)
        d_xx = self.remove_self_distance(d_xx)
        k_xx = self.rbf_kernel_mixture(d_xx)
        # k_yy
        d_yy = self.euclidean_squared_distance(y, y)
        d_yy = self.remove_self_distance(d_yy)
        k_yy = self.rbf_kernel_mixture(d_yy)
        # k_xy
        d_xy = self.euclidean_squared_distance(x, y)
        k_xy = self.rbf_kernel_mixture(d_xy)
        return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

    @staticmethod
    def rbf_kernel_mixture(exponent, sigmas=[1, 5, 10]):
        K = 0
        for sigma in sigmas:
            gamma = 1.0 / (2.0 * sigma**2)
            K += torch.exp(-gamma * exponent)
        return K

    @staticmethod
    def remove_self_distance(distmat):
        tmp_list = []
        for i, row in enumerate(distmat):
            row1 = torch.cat([row[:i], row[i + 1:]])
            tmp_list.append(row1)
        return torch.stack(tmp_list)

    @staticmethod
    def euclidean_squared_distance(x, y):
        m, n = x.size(0), y.size(0)
        distmat = (
            torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) +
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        )
        # distmat.addmm_(1, -2, x, y.t())
        distmat.addmm_(x, y.t(), beta=1, alpha=-2)
        return distmat


if __name__ == "__main__":
    mmd = MaximumMeanDiscrepancy(kernel_type="rbf")
    input1, input2 = torch.rand(3, 100), torch.rand(3, 100)
    d = mmd(input1, input2)
    print(d.item())
