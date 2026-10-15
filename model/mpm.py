"""MPCNet-derived MPM: 5531 default, original 7753 compatibility.

Source: https://github.com/Wangtao-Bao/MPCNet/blob/5024ea51e8abdf7bc40e36489041143c468b1fdf/MPCNet.py
Authors: Wangtao-Bao/MPCNet contributors. See model/MPM_SOURCE.md.
The MPM kernel selection is adapted; helper classes retain upstream behavior.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class activation(nn.ReLU):
    def __init__(self, dim, act_num=3, deploy=False):
        super(activation, self).__init__()
        self.deploy = deploy
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num * 2 + 1, act_num * 2 + 1))
        self.bias = None
        self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        self.dim = dim
        self.act_num = act_num

    def forward(self, x):
        if self.deploy:
            return torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, self.bias, padding=(self.act_num * 2 + 1) // 2, groups=self.dim)
        else:
            return self.bn(torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, padding=(self.act_num * 2 + 1) // 2, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std

    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        self.bias.data = bias
        self.__delattr__('bn')
        self.deploy = True


class CBR(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = activation(c2, act_num=3)
        # self.act = self.default_act if ahaoct is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttiton(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttiton, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class MPM(nn.Module):
    def __init__(self, in_channel, kernel_variant="5531"):
        super(MPM, self).__init__()
        if kernel_variant not in ("5531", "7753"):
            raise ValueError("MPM kernel_variant must be 5531 or 7753")
        self.kernel_variant = kernel_variant
        small, medium, large = (1, 3, 5) if kernel_variant == "5531" else (3, 5, 7)
        # Attribute names retained for strict loading of original 7753 checkpoints.


        self.Conv_3 = nn.Sequential(
            nn.Conv2d(in_channel // 4, in_channel // 4, small, 1, padding=small // 2),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )

        self.Conv_5 = nn.Sequential(
            nn.Conv2d(in_channel // 4, in_channel // 4, medium, 1, padding=medium // 2),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )

        self.Conv_7 = nn.Sequential(
            nn.Conv2d(in_channel // 4, in_channel // 4, large, 1, padding=large // 2),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )


        self.Conv = CBR(in_channel * 2, in_channel, 1)
        self.ca = CoordAttiton(in_channel,in_channel)

    def forward(self, x):
        b, c, h, w = x.size()
        x_1 = x[:, :(c // 4), :, :]
        x_2 = x[:, (c // 4):(c // 4) * 2, :, :]
        x_3 = x[:, (c // 4) * 2:(c // 4) * 3, :, :]
        x_4 = x[:, (c // 4) * 3:, :, :]

        if self.kernel_variant == "5531":
            # 5x5, 5x5, 3x3, 1x1; two 5x5 branches share Conv and BN.
            branches = (self.Conv_7(x_1), self.Conv_7(x_2),
                        self.Conv_5(x_3), self.Conv_3(x_4))
        else:
            # Exact upstream call order, including shared BN updates.
            x_4_7 = self.Conv_7(x_4)
            x_3_5 = self.Conv_5(x_3)
            x_2_3 = self.Conv_3(x_2)
            x_1_1 = self.Conv_7(x_1)
            branches = (x_1_1, x_2_3, x_3_5, x_4_7)
        out = self.ca(self.Conv(torch.cat((*branches, x), 1)))

        return out
