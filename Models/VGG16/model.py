# ============================================================================
# Imports and Dependencies
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ============================================================================
# VGG Block Implementation
# ============================================================================

class _VGGBlock(nn.Module):
    """
    VGG Block consisting of convolutional layers with batch normalization
    and ReLU activation, followed by optional max pooling.
    """
    
    def __init__(self, in_channels, out_channels, num_convs, pool=True):
        super(_VGGBlock, self).__init__()
        
        layers = []
        for i in range(num_convs):
            layers.extend([
                nn.Conv2d(in_channels if i == 0 else out_channels, 
                         out_channels, 
                         kernel_size=3, 
                         stride=1, 
                         padding=1, 
                         bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


# ============================================================================
# VGG16 Model Implementation  
# ============================================================================

class VGG16Gender(nn.Module):
    """
    VGG16 implementation for gender classification, following the structure
    of the DenseNet implementation but adapted for VGG architecture.
    
    References:
    - Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks 
      for large-scale image recognition. arXiv preprint arXiv:1409.1556.
    - Sebastian Raschka's VGG16-CelebA implementation:
      https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg16-celeba.ipynb
    - DigitalOcean VGG16 tutorial:
      https://www.digitalocean.com/community/tutorials/vgg-from-scratch-pytorch
    """
    
    def __init__(self, 
                 num_classes=2,
                 classifier_dropout=0.5,
                 init_weights=True):
        
        super(VGG16Gender, self).__init__()
        
        # Feature extraction layers (following VGG16 architecture)
        self.features = nn.Sequential(OrderedDict([
            # Block 1: 64 channels, 2 conv layers
            ('block1', _VGGBlock(3, 64, 2, pool=True)),
            
            # Block 2: 128 channels, 2 conv layers  
            ('block2', _VGGBlock(64, 128, 2, pool=True)),
            
            # Block 3: 256 channels, 3 conv layers
            ('block3', _VGGBlock(128, 256, 3, pool=True)),
            
            # Block 4: 512 channels, 3 conv layers
            ('block4', _VGGBlock(256, 512, 3, pool=True)),
            
            # Block 5: 512 channels, 3 conv layers
            ('block5', _VGGBlock(512, 512, 3, pool=True)),
        ]))
        
        # Adaptive average pooling (similar to DenseNet implementation)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classification layers (following DenseNet structure with dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        
        # Initialize weights (similar to DenseNet implementation)
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        """
        Forward pass through the VGG16 network.
        Following the same pattern as DenseNet implementation.
        """
        # Extract features
        features = self.features(x)
        
        # Apply adaptive average pooling
        out = self.avgpool(features)
        
        # Flatten for classifier (similar to DenseNet)
        out = torch.flatten(out, 1)
        
        # Apply classifier
        out = self.classifier(out)
        
        return out
    
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