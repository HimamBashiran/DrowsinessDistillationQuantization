"""Clean 3D DenseNet ported from video-classification-3d-cnn-pytorch-master."""

from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module("norm1", nn.BatchNorm3d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, bias=False))
        self.add_module("norm2", nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            self.add_module(
                f"denselayer{i + 1}",
                _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate),
            )


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module("norm", nn.BatchNorm3d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv3d(num_input_features, num_output_features, kernel_size=1, bias=False))
        self.add_module("pool", nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet3D(nn.Module):
    def __init__(self, sample_size=112, sample_duration=120, growth_rate=32,
                 block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4,
                 drop_rate=0.0, num_classes=2):
        super().__init__()
        self.sample_size = sample_size
        self.sample_duration = sample_duration

        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv3d(3, num_init_features, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)),
            ("norm0", nn.BatchNorm3d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module(f"transition{i + 1}", trans)
                num_features //= 2

        # Fixed from legacy BatchNorm2d -> BatchNorm3d for 5D (B, C, T, H, W) tensors
        self.features.add_module("norm5", nn.BatchNorm3d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward_features(self, x):
        return F.relu(self.features(x), inplace=True)

    def forward(self, x):
        feat_map = self.forward_features(x)
        pooled = F.adaptive_avg_pool3d(feat_map, (1, 1, 1)).flatten(1)
        return self.classifier(pooled)


def densenet121(**kwargs):
    return DenseNet3D(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
