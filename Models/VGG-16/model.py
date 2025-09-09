# ============================================================================
# Imports and Dependencies
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ============================================================================
# VGG Block Implementation (Regular VGG - NO Batch Normalization)
# ============================================================================

class _VGGBlock(nn.Module):
    """
    Regular VGG Block consisting of convolutional layers with ReLU activation,
    followed by optional max pooling. NO BATCH NORMALIZATION.
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
                         bias=True),  # bias=True for regular VGG (no batch norm)
                nn.ReLU(inplace=True)
            ])
        
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


# ============================================================================
# Regular VGG16 Model Implementation (NO Batch Normalization)
# ============================================================================

class VGG16Gender(nn.Module):
    """
    Regular VGG16 implementation for gender classification WITHOUT batch normalization.
    
    This follows the original VGG16 architecture from:
    Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks 
    for large-scale image recognition. arXiv preprint arXiv:1409.1556.
    
    Key differences from VGG16-BN:
    - No batch normalization layers
    - Conv2d layers have bias=True (since no batch norm)
    - Follows original VGG16 structure exactly
    """
    
    def __init__(self, 
                 num_classes=2,
                 classifier_dropout=0.5,
                 init_weights=True):
        
        super(VGG16Gender, self).__init__()
        
        # Feature extraction layers (following original VGG16 architecture)
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
        
        # Adaptive average pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classification layers (original VGG16 classifier structure)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(4096, num_classes)
        )
        
        # Initialize weights following original VGG16 initialization
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        """
        Forward pass through the regular VGG16 network.
        """
        # Extract features
        features = self.features(x)
        
        # Apply adaptive average pooling
        out = self.avgpool(features)
        
        # Flatten for classifier
        out = torch.flatten(out, 1)
        
        # Apply classifier
        out = self.classifier(out)
        
        return out
    
    def _initialize_weights(self):
        """
        Initialize weights following the original VGG16 initialization scheme.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Xavier normal initialization for conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Normal initialization for linear layers
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)