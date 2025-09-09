"""
VGG16 Gender Classification - Local Server Main Execution

This cell imports from the files created in previous cells and runs the complete training pipeline.
Updated for local server with correct paths.

References:
- Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks
  for large-scale image recognition. arXiv preprint arXiv:1409.1556.
- Sebastian Raschka's VGG16-CelebA implementation:
  https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg16-celeba.ipynb
- DigitalOcean VGG16 PyTorch tutorial:
  https://www.digitalocean.com/community/tutorials/vgg-from-scratch-pytorch
"""

import os
import sys
import warnings
import torch
import traceback

# Import from the files created in previous cells
from model import VGG16Gender
from data_loader import create_data_loaders
from trainer import train_model

warnings.filterwarnings('ignore')


def main():

    # ========================================================================
    # Configuration Settings - LOCAL SERVER PATHS
    # ========================================================================

    CONFIG = {
        'img_dir': r"",  # add the path to the CelebA image file
        'attr_file': r"",  # add the path to the CelebA list_attr CSV file

        'batch_size': 16,  # Conservative batch size to avoid memory errors
        'num_epochs': 40,  # 40 epochs as requested  
        'num_workers': 4,  # Local server can handle more workers
        'train_ratio': 0.8,  # 80% of data used for training
        'val_ratio': 0.1,  # 10% for validation, remaining 10% aside for testing

        'classifier_dropout': 0.5,  # Dropout rate in the final classification layer
        'num_classes': 2,  # Binary classification: Male (1) vs Female (0)

        # Output file paths for saving results (local server paths)
        'save_path': '/home/localadmin/jupyter/VGG16_best.pth',  # Best model checkpoint
        'log_dir': '/home/localadmin/jupyter/outputs/',  # Training logs directory
    }

    # ========================================================================
    # Project Information Display
    # ========================================================================

    print("\n" + "=" * 62)
    print("VGG16 Gender Classification")
    print("=" * 62 + "\n")
    print("Project: COSC595 Information Technology Project: Implementation")
    print("Model: VGG16")
    print("Task: Binary Gender Classification (Male/Female)")
    print("Platform: Local Server")
    print("\n" + "=" * 62 + "\n")

    # ========================================================================
    # Device Setup and Hardware Detection (NO CACHE CLEARING)
    # ========================================================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device Configuration:")
    print(f"\tUtilising Device: {str(device).upper()}")

    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024 ** 3
        print(f"\tGPU Name: {gpu_name}")
        print(f"\tGPU Memory: {gpu_memory} GB")

        # Skip cache clearing to avoid memory errors
        # torch.cuda.empty_cache()  # Commented out to avoid CUDA errors
    else:
        print("\tNote: Training on CPU will be significantly slower")

    print("\n")

    # ========================================================================
    # Dataset Loading and Preprocessing
    # ========================================================================

    print("---------------")
    print("Loading Dataset")
    print("---------------\n")

    try:
        # Updated call for original data_loader format
        train_loader, val_loader, test_loader = create_data_loaders(
            img_dir=CONFIG['img_dir'],
            attr_file=CONFIG['attr_file'],
            batch_size=CONFIG['batch_size'],
            num_workers=CONFIG['num_workers'],
            train_ratio=CONFIG['train_ratio'],
            val_ratio=CONFIG['val_ratio']
        )
        print("Dataset loaded successfully!\n")

    except FileNotFoundError as e:
        print(f"Dataset files not found: {e}")
        print("\nPlease update the file paths in CONFIG section:")
        print(f"   - img_dir: {CONFIG['img_dir']}")
        print(f"   - attr_file: {CONFIG['attr_file']}")
        print("\nEnsure the dataset is in the correct location.")
        print("Expected structure:")
        print("dataset_c/")
        print("├── 000001.jpg")
        print("├── 000002.jpg")
        print("├── ... (all images)")
        print("└── list_attr_celeba.csv")
        return None

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("This could be due to:")
        print("- Incorrect file paths")
        print("- Missing CSV file (list_attr_celeba.csv)")
        print("- CSV file format issues")
        print("- Insufficient memory")
        return None

    # ========================================================================
    # Model Initialisation and Configuration
    # ========================================================================

    print("=" * 62)
    print("Initialising VGG16 Model")
    print("=" * 62 + "\n")

    print("Model Configuration:")
    print(f"\tArchitecture: VGG16")
    print(f"\tClassifier Dropout: {CONFIG['classifier_dropout']}")
    print(f"\tNumber of Classes: {CONFIG['num_classes']} (Male/Female)")
    print()

    model = VGG16Gender(
        num_classes=CONFIG['num_classes'],
        classifier_dropout=CONFIG['classifier_dropout']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024 ** 2  # Assuming 32-bit floats

    print(f"Model Statistics:")
    print(f"\tTotal Parameters: {total_params:,}")
    print(f"\tTrainable Parameters: {trainable_params:,}")
    print(f"\tModel Size: {model_size_mb:.1f} MB")
    print(f"\tMemory Footprint: ~{model_size_mb * 2:.0f} MB