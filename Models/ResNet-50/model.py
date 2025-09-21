"""
References:
    PyTorch Team. (n.d.). TorchVision Models. PyTorch. Retrieved September 01, 2025, from https://pytorch.org/vision/stable/models.html
    OpenGenus. (n.d.). Designing ResNet50 in PyTorch. OpenGenus IQ. Retrieved September 21, 2025, from https://iq.opengenus.org/designing-resnet50-in-pytorch/
    Nandi, A., & Karmakar, S. (2021). Age and Gender Prediction using Deep CNNs and Transfer Learning [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2110.12633
    Rabbi, M. R. (2022). Gender classification | PyTorch ResNet50. Kaggle. https://www.kaggle.com/code/rabbi2k3/gender-classification-pytorch-resnet-50
"""



# ============================================================================
# Imports and Dependencies
# ============================================================================

import torch
import torch.nn as nn
from collections import OrderedDict

# ============================================================================
# ResNet Bottleneck Block
# ============================================================================

class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)

        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride, padding=dilation,
            groups=groups, dilation=dilation, bias=False
        )
        self.bn2 = norm_layer(width)

        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ============================================================================
# ResNet-50 Model (Gender Classification)
# ============================================================================

class ResNet50Gender(nn.Module):
    """
    A custom ResNet-50 backbone with a simple classifier head:
    Stem: 7x7 conv -> BN -> ReLU -> 3x3 maxpool
    Stages: [3, 4, 6, 3] Bottleneck blocks
    Head: BN -> ReLU -> GAP -> Flatten -> Dropout -> Linear(num_classes)
    """

    def __init__(
            self,
            block = _Bottleneck,
            layers = (3, 4, 6, 3),
            num_classes: int = 2,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation=(False, False, False),
            norm_layer: nn.Module = None,
            classifier_dropout: float = 0.5,
            block_drop_rate: float = 0.0,
            zero_init_residual: bool = True,
            memory_efficient: bool = False,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.memory_efficient = memory_efficient
        self.block_drop_rate = block_drop_rate

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation must be 3 booleans (for layer2/3/4).")

        # Stem
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn0', norm_layer(self.inplanes)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilate=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        # Optional
        self.tail_norm = norm_layer(512 * block.expansion)
        self.tail_relu = nn.ReLU(inplace=True)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(512 * block.expansion, num_classes),
        )

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, _Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0.0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.inplanes, planes * block.expansion,
                                   kernel_size=1, stride=stride, bias=False)),
                ('bn', norm_layer(planes * block.expansion)),
            ]))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)     # stem
        x = self.layer1(x)       # 56x56 -> 56x56
        x = self.layer2(x)       # 56x56 -> 28x28
        x = self.layer3(x)       # 28x28 -> 14x14
        x = self.layer4(x)       # 14x14 -> 7x7

        x = self.tail_relu(self.tail_norm(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x