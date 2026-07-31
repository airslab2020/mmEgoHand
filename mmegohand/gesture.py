"""ResNet classifiers for pose-sequence gesture recognition."""

from __future__ import annotations

from typing import Type

import torch
from torch import Tensor, nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        return self.relu(output + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        width_per_group: int = 64,
    ) -> None:
        super().__init__()
        width = int(out_channels * (width_per_group / 64.0)) * groups
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(
            width, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        return self.relu(output + identity)


class GestureResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock] | Type[Bottleneck],
        block_counts: list[int],
        num_classes: int = 8,
    ) -> None:
        super().__init__()
        self.in_channels = 64
        self.groups = 1
        self.width_per_group = 64
        self.conv1 = nn.Conv2d(
            30, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, block_counts[0])
        self.layer2 = self._make_layer(block, 128, block_counts[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_counts[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_counts[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )

    def _make_layer(
        self,
        block: Type[BasicBlock] | Type[Bottleneck],
        channels: int,
        count: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(channels * block.expansion),
            )
        layers = [
            block(
                self.in_channels,
                channels,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                width_per_group=self.width_per_group,
            )
        ]
        self.in_channels = channels * block.expansion
        for _ in range(1, count):
            layers.append(
                block(
                    self.in_channels,
                    channels,
                    groups=self.groups,
                    width_per_group=self.width_per_group,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, pose: Tensor) -> Tensor:
        if pose.ndim == 5 and pose.shape[1] == 1:
            pose = pose[:, 0]
        if pose.ndim != 4 or tuple(pose.shape[1:]) != (30, 42, 3):
            raise ValueError(
                f"pose must have shape [B,30,42,3], got {tuple(pose.shape)}"
            )
        output = self.maxpool(self.relu(self.bn1(self.conv1(pose))))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return self.fc(torch.flatten(self.avgpool(output), 1))


def build_gesture_model(
    architecture: str = "resnet50", num_classes: int = 8
) -> GestureResNet:
    variants = {
        "resnet18": (BasicBlock, [2, 2, 2, 2]),
        "resnet34": (BasicBlock, [3, 4, 6, 3]),
        "resnet50": (Bottleneck, [3, 4, 6, 3]),
        "resnet101": (Bottleneck, [3, 4, 23, 3]),
    }
    if architecture not in variants:
        raise ValueError(
            f"unsupported architecture {architecture!r}; "
            f"choose from {', '.join(variants)}"
        )
    block, counts = variants[architecture]
    return GestureResNet(block, counts, num_classes)
