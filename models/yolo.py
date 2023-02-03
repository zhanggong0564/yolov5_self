from models.backbone import CSPDarknet
from models.Neck import PANet
from models.common import Detect, make_divisible
import torch.nn as nn
import torch


class Yolo(nn.Module):
    def __init__(self, num_classes, num_anchor=3, depth_multiple=0.33, width_multiple=0.50):
        super(Yolo, self).__init__()
        self.backbone = CSPDarknet(depth_multiple, width_multiple)
        self.neck = PANet(depth_multiple, width_multiple)
        self.head = Detect(
            num_classes,
            num_anchor,
            [make_divisible(256 * width_multiple, 8), make_divisible(512 * width_multiple, 8), make_divisible(1024 * width_multiple, 8)],
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


def yolov5s(num_classes):
    models = Yolo(num_classes, depth_multiple=0.33, width_multiple=0.50)
    return models


def yolov5m(
    num_classes,
):
    models = Yolo(num_classes, depth_multiple=0.67, width_multiple=0.75)
    return models


def yolov5l(num_classes):
    models = Yolo(num_classes, depth_multiple=1.0, width_multiple=1.0)
    return models


def yolov5x(num_classes):
    models = Yolo(num_classes, depth_multiple=1.33, width_multiple=1.25)
    return models

def get_model(config):
    if config.model_name == 'yolov5s':
        return yolov5s(config.num_classes)

if __name__ == "__main__":
    x = torch.randn(3, 3, 640, 640)
    models = yolov5s(20)
    y = models(x)
    for i in y:
        print(i.shape)
