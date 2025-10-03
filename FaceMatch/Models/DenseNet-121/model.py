"""
DenseNet-121 Implementation for Binary Gender Classification

This file implements the DenseNet-121 architecture. The implementation follows
PyTorch's official DenseNet source code model structure.

References:
    Liu, Z. (N.D.) DenseNet.
        https://github.com/liuzhuang13/DenseNet
    PyTorch (N.D.). Source code for torchvision.models.densenet.
        https://docs.pytorch.org/vision/0.8/_modules/torchvision/models/densenet.html
"""


# ============================================================================
# Imports and Dependencies
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from collections import OrderedDict


# ============================================================================
# Dense Layer Implementation
# ============================================================================

class _DenseLayer(nn.Module):
    """
    Dense Layer with Bottleneck (BN).

    Implements the core building block of the DenseNet model with:
    1. Bottleneck, consisting of a 1x1 convolution layer
    2. Composite function of a 3x3 convolution for feature extractions
    3. Dense connectivity that concatenates all the previous feature maps

    Args:
        num_input_features (int): Is the number of input channels
        growth_rate (int): Is the number of output channels
        bn_size (int): Bottleneck layers
        drop_rate (float): Dropout probability after convolution
        memory_efficient (bool): Save memory by using gradient checkpointing
    """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()

        # First batch norm, ReLU, and 1x1 convolutions (bottleneck)
        # Reduces `num_input_features` -> `bn_size` * `growth_rate` channels
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(
            num_input_features,
            bn_size * growth_rate,
            kernel_size=1,
            stride=1,
            bias=False
        )),

        # Second batch norm, ReLU, and 3x3 convolution (a composite function)
        # Produces the `growth_rate` output channels
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )),

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        """
        The bottleneck function concatenates inputs and then applies the
        first `conv` block.
        """
        # Concatenates previous feature maps along the channel's dimension
        concated_features = torch.cat(inputs, 1)

        # Applies the `BN`-`ReLU`-`Conv`(1x1) bottleneck
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def has_gradients(self, input):
        """
        Checks if the input tensor requires any gradient computations.
        """
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, input):
        """
        Uses gradient checkpointing for training.
        """
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    def forward(self, input):
        """
        Forward pass through the dense layer.
        """
        # Converts a single tensor to a list for consistent handling approach
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        # Uses a memory-efficient checkpoint if it's enabled and are gradients required
        if self.memory_efficient and self.has_gradients(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        # Applies the second convolution block (`BN`-`ReLU`-`Conv`(3x3))
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        # If specified, apply dropout to new feature maps
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)

        return new_features


# ============================================================================
# Dense Block Implementation
# ============================================================================

class _DenseBlock(nn.ModuleDict):

    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()

        for i in range(num_layers):
            layer_input_features = num_input_features + i * growth_rate

            layer = _DenseLayer(
                num_input_features=layer_input_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )

            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]

        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)

        return torch.cat(features, 1)


# ============================================================================
# Transition Layer Implementation
# ============================================================================

class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()

        self.add_module('norm', nn.BatchNorm2d(num_input_features))

        self.add_module('relu', nn.ReLU(inplace=True))

        self.add_module('conv', nn.Conv2d(
            num_input_features,
            num_output_features,
            kernel_size=1,
            stride=1,
            bias=False
        ))

        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


# ============================================================================
# DenseNet-121 Model Implementation
# ============================================================================

class DenseNet121Gender(nn.Module):

    def __init__(self,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0.2,
                 num_classes=2,
                 memory_efficient=False):

        super(DenseNet121Gender, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(
                3,
                num_init_features,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )),
        ]))

        num_features = num_init_features

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                transition_output_features = num_features // 2
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=transition_output_features
                )
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = transition_output_features

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)

        out = F.relu(features, inplace=True)

        out = F.adaptive_avg_pool2d(out, (1, 1))

        out = torch.flatten(out, 1)

        out = self.classifier(out)

        return out