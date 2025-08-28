"""
EfficientNet-B4 Gender Classification - Main Entry Point

This implementation is based on the EfficientNet architecture for efficient 
convolutional neural networks with compound scaling.

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

Usage:
    `python main.py`
"""

import os
import sys
import warnings
import torch
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import EfficientNetB4Gender
from data_loader import create_data_loaders
from trainer import train_model

warnings.filterwarnings('ignore')


def main():

    # ========================================================================
    # Configuration Settings
    # ========================================================================

    CONFIG = {
        'img_dir': r"",  # add the path to the CelebA image file
        'attr_file': r"",  # add the path to the CelebA list_attr CSV file

        'batch_size': 16,  # Conservative batch size for EfficientNet-B4
        'num_epochs': 30,  # Number of training epochs
        'num_workers': 4,  # Parallel data loading processes
        'train_ratio': 0.8,  # 80% of data used for training
        'val_ratio': 0.1,  # 10% for validation, remaining 10% aside for testing

        'classifier_dropout': 0.4,  #