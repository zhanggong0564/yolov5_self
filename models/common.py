import torch
import torch.nn as nn
import yaml
import math

import comment.nn_utils as nn_utils

# 制造一个整除的数字
def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def autopad(kernel, padding=None):
    # Pad to 'same'
    if padding is None:
        padding = kernel // 2
    return padding


class Conv(nn.Module):
    """
    标准卷积层
    """

    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=None, groups=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding=autopad(kernel_size, padding), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.LeakyReLU(0.1, inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    标准瓶颈层
    """

    def __init__(self, in_channel, out_channel, shortcut=True, groups=1, expansion=0.5):
        super().__init__()

        hidden_channel = int(out_channel * expansion)
        self.conv1 = Conv(in_channel, hidden_channel, 1, 1)
        self.conv2 = Conv(hidden_channel, out_channel, 3, 1, groups=groups)
        self.add = shortcut and in_channel == out_channel

    def forward(self, x):
        if self.add:
            return x + self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x))


class BottleneckCSP(nn.Module):
    def __init__(self, in_channel, out_channel, repeats=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__()

        hidden_channel = int(out_channel * expansion)
        self.conv1 = Conv(in_channel, hidden_channel, 1, 1)
        self.conv2 = nn.Conv2d(in_channel, hidden_channel, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(hidden_channel, hidden_channel, 1, 1, bias=False)
        self.conv4 = Conv(2 * hidden_channel, out_channel, 1, 1)
        self.bn = nn.BatchNorm2d(2 * hidden_channel)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.repeat_blocks = nn.Sequential(*[Bottleneck(hidden_channel, hidden_channel, shortcut, groups, expansion=1.0) for _ in range(repeats)])

    def forward(self, x):
        y1 = self.conv3(self.repeat_blocks(self.conv1(x)))
        y2 = self.conv2(x)
        ycat = torch.cat((y1, y2), dim=1)
        return self.conv4(self.act(self.bn(ycat)))


class SPP(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size_list=(5, 9, 13)):
        super().__init__()

        hidden_channel = in_channel // 2
        self.conv1 = Conv(in_channel, hidden_channel, 1, 1)
        self.conv2 = Conv(hidden_channel * (len(kernel_size_list) + 1), out_channel, 1, 1)

        self.spatial_pyramid_poolings = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2) for kernel_size in kernel_size_list]
        )

    def forward(self, x):
        x = self.conv1(x)
        spp = torch.cat([x] + [m(x) for m in self.spatial_pyramid_poolings], dim=1)
        return self.conv2(spp)


class Focus(nn.Module):
    """
    一种下采样方式,通过切片的形式将输入进行下采样2倍,但是不损失信息
    input[1x3x100x100]
    output[1x12x50x50]
    """

    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=None, groups=1, activation=True):
        super().__init__()
        self.conv = Conv(in_channel * 4, out_channel, kernel_size, stride, padding, groups, activation)

    def forward(self, x):
        #  block(y, x)
        #  a(0, 0)      b(1, 0)
        #  c(0, 1)      d(1, 1)
        a = x[..., ::2, ::2]
        b = x[..., 1::2, ::2]
        c = x[..., ::2, 1::2]
        d = x[..., 1::2, 1::2]
        return self.conv(torch.cat([a, b, c, d], dim=1))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.dimension = dimension

    def forward(self, x):
        return torch.cat(x, dim=self.dimension)


class Detect(nn.Module):
    def __init__(self, num_classes, num_anchor, reference_channels):
        super().__init__()

        self.num_anchor = num_anchor
        self.num_classes = num_classes
        self.num_output = self.num_classes + 5
        self.heads = nn.ModuleList([nn.Conv2d(input_channel, self.num_output * self.num_anchor, 1) for input_channel in reference_channels])

    def forward(self, x):
        if isinstance(x, tuple):
            x = list(x)
        for ilevel, head in enumerate(self.heads):
            x[ilevel] = head(x[ilevel])
        return x
