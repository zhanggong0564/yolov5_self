import torch
import torch.nn as nn
from models.common import Conv,make_divisible,BottleneckCSP


class PANet(nn.Module):
    def __init__(self, depth_multiple, width_multiple):
        super(PANet, self).__init__()
        self.conv1 = Conv(make_divisible(1024 * width_multiple, 8), make_divisible(512 * width_multiple, 8), kernel_size=1)
        self.up1 = nn.Upsample(scale_factor=2)
        self.BottleneckCSP2_1 = BottleneckCSP(
            make_divisible(1024 * width_multiple, 8), make_divisible(512 * width_multiple, 8), repeats=max(round(3 * depth_multiple), 1), shortcut=False
        )

        self.conv2 = Conv(make_divisible(512 * width_multiple, 8), make_divisible(256 * width_multiple, 8), kernel_size=1)
        self.BottleneckCSP2_2 = BottleneckCSP(
            make_divisible(512 * width_multiple, 8), make_divisible(256 * width_multiple, 8), repeats=max(round(3 * depth_multiple), 1), shortcut=False
        )

        self.conv3 = Conv(make_divisible(256 * width_multiple, 8), make_divisible(256 * width_multiple, 8), kernel_size=3, stride=2)
        self.BottleneckCSP2_3 = BottleneckCSP(
            make_divisible(512 * width_multiple, 8), make_divisible(512 * width_multiple, 8), repeats=max(round(3 * depth_multiple), 1), shortcut=False
        )

        self.conv4 = Conv(make_divisible(512 * width_multiple, 8), make_divisible(512 * width_multiple, 8), kernel_size=3, stride=2)
        self.BottleneckCSP2_4 = BottleneckCSP(
            make_divisible(1024 * width_multiple, 8), make_divisible(1024 * width_multiple, 8), repeats=max(round(3 * depth_multiple), 1), shortcut=False
        )

    def forward(self, x):
        x1, x2, x3 = x
        x3 = self.conv1(x3)
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.BottleneckCSP2_1(x)

        temp = self.conv2(x)
        x = self.up1(temp)
        x = torch.cat([x, x1], dim=1)
        out3 = self.BottleneckCSP2_2(x)

        x = self.conv3(out3)
        x = torch.cat([x, temp], dim=1)
        out2 = self.BottleneckCSP2_3(x)

        x = self.conv4(out2)
        x = torch.cat([x, x3], dim=1)
        x = self.BottleneckCSP2_4(x)

        return out3, out2, x


if __name__ == "__main__":
    x = torch.randn(3, 3, 640, 640)
    from models.backbone import CSPDarknet

    model = CSPDarknet(depth_multiple=0.33, width_multiple=0.50)
    y = model(x)
    model2 = PANet(model.depth_multiple, model.width_multiple)
    y2 = model2(y)
    for i in y2:
        print(i.shape)
