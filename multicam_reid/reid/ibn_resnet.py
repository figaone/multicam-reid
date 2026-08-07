"""
Self-contained ResNet-50-IBN-a backbone for vehicle re-identification.

Ported from the AI City Challenge 2021 Track2 (DMT) / IBN-Net so it runs on a
modern PyTorch (2.x) with no external repo dependencies. IBN (Instance-Batch
Normalization) mixes InstanceNorm + BatchNorm in early stages, which makes the
learned features far more robust to camera/appearance domain shift -- exactly
what cross-camera intersection ReID needs.

Official ImageNet weights (resnet50_ibn_a-d9d0bb7b.pth) load directly into this
module; optionally, vehicle-ReID fine-tuned weights with the same key layout can
be supplied instead for stronger performance.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class IBN(nn.Module):
    """Half InstanceNorm + half BatchNorm (IBN-a)."""

    def __init__(self, planes: int):
        super().__init__()
        half1 = int(planes / 2)
        self._half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self._half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        return torch.cat((out1, out2), 1)


class BottleneckIBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = IBN(planes) if ibn else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResNetIBN(nn.Module):
    """ResNet-IBN-a feature extractor (returns the layer4 feature map)."""

    def __init__(self, last_stride: int = 1, layers=(3, 4, 6, 3), num_classes: int = 1000):
        scale = 64
        self.inplanes = scale
        super().__init__()
        self.conv1 = nn.Conv2d(3, scale, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(scale)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BottleneckIBN, scale, layers[0])
        self.layer2 = self._make_layer(BottleneckIBN, scale * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(BottleneckIBN, scale * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(BottleneckIBN, scale * 8, layers[3], stride=last_stride)
        # kept only so official ImageNet checkpoints load without key mismatches
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(scale * 8 * BottleneckIBN.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        # IBN is applied in all stages except the deepest (planes == 512).
        ibn = planes != 512
        layers = [block(self.inplanes, planes, ibn, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # (N, 2048, H', W')


def resnet50_ibn_a(last_stride: int = 1) -> ResNetIBN:
    return ResNetIBN(last_stride=last_stride, layers=(3, 4, 6, 3))


def load_backbone_weights(model: ResNetIBN, weights_path: str) -> tuple[int, int]:
    """
    Load a state dict into the backbone, ignoring classifier / missing keys.

    Returns (loaded, total) parameter-tensor counts for logging.
    """
    checkpoint = torch.load(weights_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model_state = model.state_dict()
    loaded = 0
    for key, value in state.items():
        key = key.replace("module.", "")
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded += 1
    model.load_state_dict(model_state)
    return loaded, len(model_state)
