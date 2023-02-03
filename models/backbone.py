import torch
import torch.nn as nn
from models.common import Conv, make_divisible, BottleneckCSP, Focus, SPP


class CSPDarknet(nn.Module):
    def __init__(self, depth_multiple, width_multiple):
        super(CSPDarknet, self).__init__()
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        self.Focus = Focus(in_channel=3, out_channel=make_divisible(64 * width_multiple, 8), kernel_size=3)
        self.conv1 = Conv(make_divisible(64 * width_multiple, 8), make_divisible(128 * width_multiple, 8), kernel_size=3, stride=2)
        self.BottleneckCSP1 = BottleneckCSP(make_divisible(128 * width_multiple, 8), make_divisible(128 * width_multiple, 8))
        self.conv2 = Conv(make_divisible(128 * width_multiple, 8), out_channel=make_divisible(256 * width_multiple, 8), kernel_size=3, stride=2)
        self.BottleneckCSP2 = BottleneckCSP(
            make_divisible(256 * width_multiple, 8), make_divisible(256 * width_multiple, 8), repeats=max(round(9 * depth_multiple), 1)
        )
        self.conv3 = Conv(make_divisible(256 * width_multiple, 8), make_divisible(512 * width_multiple, 8), kernel_size=3, stride=2)
        self.BottleneckCSP3 = BottleneckCSP(
            make_divisible(512 * width_multiple, 8), make_divisible(512 * width_multiple, 8), repeats=max(round(9 * depth_multiple), 1)
        )
        self.conv4 = Conv(make_divisible(512 * width_multiple, 8), make_divisible(1024 * width_multiple, 8), kernel_size=3, stride=2)
        self.spp = SPP(make_divisible(1024 * width_multiple, 8), make_divisible(1024 * width_multiple, 8))
        self.BottleneckCSP4 = BottleneckCSP(
            make_divisible(1024 * width_multiple, 8), make_divisible(1024 * width_multiple, 8), repeats=max(round(3 * depth_multiple), 1), shortcut=False
        )

    def forward(self, x):
        x = self.Focus(x)
        x = self.conv1(x)
        x = self.BottleneckCSP1(x)
        x1 = self.conv2(x)
        x = self.BottleneckCSP2(x1)
        x2 = self.conv3(x)
        x = self.BottleneckCSP3(x2)
        x = self.conv4(x)
        x = self.spp(x)
        x3 = self.BottleneckCSP4(x)
        return x1, x2, x3


if __name__ == "__main__":
    x = torch.randn(3, 3, 640, 640)
    model = CSPDarknet(depth_multiple=0.33, width_multiple=0.50)
    y = model(x)
    for i in y:
        print(i.shape)
