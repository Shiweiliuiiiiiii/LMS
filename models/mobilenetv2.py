"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018).
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""
# import sys, os
# sys.path.append('/home/shiweil/Projects/cutlass/examples/19_large_depthwise_conv2d_torch_extension')

import torch.nn as nn
import math
from timm.models.registry import register_model
from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM


"""mobilenetv2 in pytorch


[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, num_classes=100, kernel_size=(3,3)):
        super().__init__()
        # padding = kernel_size // 2
        self.pw1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True))

        self.dw_LoRA_1 = nn.Conv2d(in_channels * t, in_channels * t, (kernel_size[0], kernel_size[1]), stride=(stride, stride), padding=((kernel_size[0]-1)//2, (kernel_size[1]-1)//2), groups = in_channels * t)
        self.dw_BN = nn.BatchNorm2d(in_channels * t)
        self.dw_relu6 = nn.ReLU6(inplace=True)

        self.dw_LoRA_2 = nn.Conv2d(in_channels * t, in_channels * t, (kernel_size[1], kernel_size[0]),
                                   stride=(stride, stride),
                                   padding=((kernel_size[1] - 1) // 2, (kernel_size[0] - 1) // 2), groups = in_channels * t)


        self.pw2 = nn.Sequential(
            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels))

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels


    def forward(self, x):
        # one path
        output = self.pw1(x)
        residual = self.dw_LoRA_1(output) + self.dw_LoRA_2(output)
        residual = self.dw_BN(residual)
        residual = self.dw_relu6(residual)

        residual = self.pw2(residual)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual

class MobileNetV2_dual(nn.Module):

    def __init__(self, num_classes=1000, kernel_size=(7,3), width_factor=6):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1, kernel_size=kernel_size)
        self.stage2 = self._make_stage(2, 16, 24, 2, width_factor, kernel_size=kernel_size)
        self.stage3 = self._make_stage(3, 24, 32, 2, width_factor, kernel_size=kernel_size)
        self.stage4 = self._make_stage(4, 32, 64, 2, width_factor, kernel_size=kernel_size)
        self.stage5 = self._make_stage(3, 64, 96, 1, width_factor, kernel_size=kernel_size)
        self.stage6 = self._make_stage(3, 96, 160, 1, width_factor, kernel_size=kernel_size)
        self.stage7 = LinearBottleNeck(160, 320, 1, width_factor, kernel_size=kernel_size)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, num_classes, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t, kernel_size):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t, kernel_size=kernel_size))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t, kernel_size=kernel_size))
            repeat -= 1

        return nn.Sequential(*layers)

@register_model
def MobileNetV2(pretrained=False, **kwargs):
    model = MobileNetV2_dual(**kwargs)
    return model