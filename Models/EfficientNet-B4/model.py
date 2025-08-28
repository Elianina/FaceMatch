# ============================================================================
# Imports and Dependencies
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict


# ============================================================================
# EfficientNet Building Blocks
# ============================================================================

class Swish(nn.Module):
    """
    Swish activation function (x * sigmoid(x))
    Used in EfficientNet for better performance than ReLU.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


class DropConnect(nn.Module):
    """
    Drop Connect implementation for regularization in EfficientNet.
    Randomly zeros entire channels during training.
    """
    def __init__(self, drop_rate):
        super(DropConnect, self).__init__()
        self.drop_rate = drop_rate
    
    def forward(self, x):
        if not self.training:
            return x
        
        batch_size = x.shape[0]
        random_tensor = (1 - self.drop_rate) + torch.rand(
            (batch_size, 1, 1, 1), dtype=x.dtype, device=x.device
        )
        random_tensor = random_tensor.floor()
        return x.div(1 - self.drop_rate) * random_tensor


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block used in EfficientNet.
    Adaptively recalibrates channel-wise feature responses.
    """
    def __init__(self, in_channels, reduced_channels):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            Swish(),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution Block (MBConv) used in EfficientNet.
    Combines depthwise separable convolutions with squeeze-and-excitation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 expand_ratio, se_ratio, drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self.expand_conv = None
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                Swish()
            )
        
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, 
                     stride, padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            Swish()
        )
        
        # Squeeze and Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = SqueezeExcitation(expanded_channels, se_channels)
        
        # Point-wise convolution
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Drop Connect
        if drop_connect_rate > 0:
            self.drop_connect = DropConnect(drop_connect_rate)
        else:
            self.drop_connect = None
    
    def forward(self, x):
        identity = x
        
        # Expansion
        if self.expand_conv is not None:
            x = self.expand_conv(x)
        
        # Depthwise
        x = self.depthwise_conv(x)
        
        # Squeeze and Excitation
        x = self.se(x)
        
        # Point-wise
        x = self.pointwise_conv(x)
        
        # Drop Connect
        if self.drop_connect is not None:
            x = self.drop_connect(x)
        
        # Residual connection
        if self.use_residual:
            x = x + identity
        
        return x


# ============================================================================
# EfficientNet-B4 Model Implementation
# ============================================================================

class EfficientNetB4Gender(nn.Module):
    """
    EfficientNet-B4 implementation for gender classification.
    
    References:
    - Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for 
      Convolutional Neural Networks. International Conference on Machine Learning (ICML).
    - Official EfficientNet paper: https://arxiv.org/abs/1905.11946
    - EfficientNet TensorFlow implementation: 
      https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
    - PyTorch EfficientNet implementation: 
      https://github.com/lukemelas/EfficientNet-PyTorch
    - Ross Wightman's timm library: 
      https://github.com/rwightman/pytorch-image-models
    """
    
    def __init__(self, 
                 num_classes=2,
                 classifier_dropout=0.4,
                 drop_connect_rate=0.2,
                 init_weights=True):
        
        super(EfficientNetB4Gender, self).__init__()
        
        # EfficientNet-B4 configuration
        # [expand_ratio, channels, num_layers, stride, kernel_size, se_ratio]
        block_configs = [
            [1, 24, 2, 1, 3, 0.25],   # Stage 1
            [6, 32, 4, 2, 3, 0.25],   # Stage 2
            [6, 56, 4, 2, 5, 0.25],   # Stage 3
            [6, 112, 6, 2, 3, 0.25],  # Stage 4
            [6, 160, 6, 1, 5, 0.25],  # Stage 5
            [6, 272, 8, 2, 5, 0.25],  # Stage 6
            [6, 448, 2, 1, 3, 0.25],  # Stage 7
        ]
        
        # Stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            Swish()
        )
        
        # Build MBConv blocks
        self.blocks = nn.ModuleList([])
        in_channels = 48
        total_blocks = sum(config[2] for config in block_configs)
        block_idx = 0
        
        for expand_ratio, out_channels, num_layers, stride, kernel_size, se_ratio in block_configs:
            for layer_idx in range(num_layers):
                # Drop connect rate increases linearly across blocks
                block_drop_rate = drop_connect_rate * block_idx / total_blocks
                
                self.blocks.append(MBConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride if layer_idx == 0 else 1,
                    expand_ratio=expand_ratio,
                    se_ratio=se_ratio,
                    drop_connect_rate=block_drop_rate
                ))
                
                in_channels = out_channels
                block_idx += 1
        
        # Head convolution
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels, 1792, kernel_size=1, bias=False),
            nn.BatchNorm2d(1792),
            Swish()
        )
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier (following DenseNet structure with dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(1792, num_classes)
        )
        
        # Initialize weights (similar to DenseNet implementation)
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        """
        Forward pass through the EfficientNet-B4 network.
        Following the same pattern as DenseNet implementation.
        """
        # Stem
        x = self.stem(x)
        
        # MBConv blocks
        for block in self.blocks:
            x = block(x)
        
        # Head
        x = self.head_conv(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        
        # Flatten for classifier (similar to DenseNet)
        x = torch.flatten(x, 1)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        """
        Initialize weights following the same pattern as DenseNet implementation.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming normal initialization (same as DenseNet)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Constant initialization (same as DenseNet)
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Normal initialization for linear layers (same as DenseNet)
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)