"""
VGG-16 Gender Classification - Main Execution

This implementation uses the classic VGG-16 architecture for gender classification
with batch normalization for improved training stability.

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
from log_utils import start_logging, stop_logging

warnings.filterwarnings('ignore')


def main():

    # Begin logging
    original_stdout, log_file, log_file_path = start_logging('VGG-16', '/home/localadmin/jupyter/outputs/')

    # ========================================================================
    # Configuration Settings
    # ========================================================================

    CONFIG = {
        'img_dir': '/home/localadmin/jupyter/dataset/dataset_c',  # Path to folder with all images
        'attr_file': '/home/localadmin/jupyter/dataset/dataset_c/list_attr_celeba.csv',  # Path to CSV file

        'batch_size': 16,  # Conservative batch size for VGG-16
        'num_epochs': 40,  # Number of training epochs
        'num_workers': 4,  # Parallel data loading processes
        'train_ratio': 0.8,  # 80% of data used for training
        'val_ratio': 0.1,  # 10% for validation, remaining 10% aside for testing

        'classifier_dropout': 0.5,  # Dropout rate in the final classification layer
        'num_classes': 2,  # Binary classification: Male (1) vs Female (0)

        # Output file paths for saving results (local server paths)
        'save_path': '/home/localadmin/jupyter/VGG-16_best.pth',  # Best model checkpoint
        'log_dir': '/home/localadmin/jupyter/outputs/',  # Training logs directory
        'plots_dir': '/home/localadmin/jupyter/outputs/',  # Plots and visualizations directory
    }

    # ========================================================================
    # Project Information Display
    # ========================================================================

    print("\n" + "=" * 62)
    print("VGG-16 Gender Classification")
    print("=" * 62 + "\n")
    print("Project: COSC595 Information Technology Project: Implementation")
    print("Model: VGG-16 (with Batch Normalization)")
    print("Task: Binary Gender Classification (Male/Female)")
    print("Platform: Local Server")
    print("\n" + "=" * 62 + "\n")

    # ========================================================================
    # Device Setup and Hardware Detection
    # ========================================================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device Configuration:")
    print(f"\tUtilising Device: {str(device).upper()}")

    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024 ** 3
        print(f"\tGPU Name: {gpu_name}")
        print(f"\tGPU Memory: {gpu_memory} GB")

        torch.cuda.empty_cache()
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
        print("\nEnsure the dataset is uploaded and paths are correct for local server.")
        print("Expected structure:")
        print("/home/localadmin/jupyter/dataset/dataset_c/")
        print("├── [all image files directly here]")
        print("└── list_attr_celeba.csv")
        return None

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("This could be due to:")
        print("- Incorrect file paths")
        print("- Missing CSV file")
        print("- CSV file format issues")
        print("- Insufficient memory")
        return None

    # ========================================================================
    # Model Initialisation and Configuration
    # ========================================================================

    print("=" * 62)
    print("Initialising VGG-16 Model")
    print("=" * 62 + "\n")

    print("Model Configuration:")
    print(f"\tArchitecture: VGG-16")
    print(f"\tClassifier Dropout: {CONFIG['classifier_dropout']}")
    print(f"\tNumber of Classes: {CONFIG['num_classes']} (Male/Female)")
    print(f"\tBatch Normalization: Yes (Improved VGG)")
    print()

    model = VGG16Gender(
        classifier_dropout=CONFIG['classifier_dropout'],
        num_classes=CONFIG['num_classes']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024 ** 2  # Assuming 32-bit floats

    print(f"Model Statistics:")
    print(f"\tTotal Parameters: {total_params:,}")
    print(f"\tTrainable Parameters: {trainable_params:,}")
    print(f"\tModel Size: {model_size_mb:.1f} MB")
    print(f"\tMemory Footprint: ~{model_size_mb * 2:.0f} MB (including gradients)")
    print()

    # ========================================================================
    # Training Execution
    # ========================================================================

    print("=" * 62)
    print("Starting Training Phase")
    print("=" * 62 + "\n")

    try:
        os.makedirs(os.path.dirname(CONFIG['save_path']), exist_ok=True)
        os.makedirs(CONFIG['log_dir'], exist_ok=True)
        print("Output directories verified/created successfully")
    except Exception as e:
        print(f"Warning: Could not create output directories: {e}")
        print("Model saving may fail")

    print()

    try:
        results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            config=CONFIG
        )

        # ====================================================================
        # Final Results Summary and Analysis
        # ====================================================================

        if results is not None:
            print("\n" + "=" * 62)
            print("FINAL RESULTS SUMMARY")
            print("=" * 62)

            best_val_acc = results['best_val_acc']
            best_epoch = results['best_epoch']
            test_acc = results['test_acc']
            test_loss = results['test_loss']

            print(f"Best Validation Accuracy: {best_val_acc:.2f}% (achieved at Epoch {best_epoch})")
            print(f"Final Test Accuracy: {test_acc:.2f}%")
            print(f"Final Test Loss: {test_loss:.4f}")
            print(f"Model saved to: {CONFIG['save_path']}")

            print("\nPerformance Analysis:")
            if test_acc >= 95.0:
                print("Excellent performance! VGG-16 achieved SOTA accuracy.")
            elif test_acc >= 90.0:
                print("Good performance! VGG-16 is working well for binary classification.")
            elif test_acc >= 85.0:
                print("Moderate performance. Consider further tuning.")
            elif test_acc >= 80.0:
                print("Reasonable performance for VGG-16 on this task.")
            else:
                print("Performance could be improved. Consider longer training or hyperparameter tuning.")

            print("=" * 62)

            print("VGG-16 training completed successfully!")
            print("Check the saved model at:", CONFIG['save_path'])
            print("=" * 62)

        # Stop logging
        stop_logging(original_stdout, log_file, log_file_path)

        return results

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        print("Any progress up to this point has been saved")

        stop_logging(original_stdout, log_file, log_file_path)  # Logging

        return None

    except RuntimeError as e:
        error_msg = str(e)
        print(f"Runtime error during training: {error_msg}")
        print("\nPossible solutions:")
        if "out of memory" in error_msg.lower():
            print("- Reduce batch_size in CONFIG (try 8 or 4)")
            print("- Restart runtime to clear GPU memory")
        else:
            print("- Reduce batch_size in CONFIG")
            print("- Check GPU allocation")

        stop_logging(original_stdout, log_file, log_file_path)  # Logging

        return None

    except Exception as e:
        print(f"Unexpected error during training: {e}")
        print("\nError traceback:")
        traceback.print_exc()

        stop_logging(original_stdout, log_file, log_file_path)  # Logging

        return None


# ========================================================================
# Main Execution
# ========================================================================

if __name__ == "__main__":
    print("Starting VGG-16 Gender Classification Training")
    print("=" * 62)

    results = main()

    if results is not None:
        print(f"\nTraining session completed with {results['test_acc']:.2f}% test accuracy")
    else:
        print("\nTraining session failed or was interrupted")
        print("Check the error messages above and try again.")

    print("=" * 62)
    print("Script execution finished.")