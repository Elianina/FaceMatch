"""
DenseNet-121 Gender Classification - Main Entry Point

Usage:
    `python main.py`

"""

import os
import sys
import warnings
import torch
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import DenseNet121Gender
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

        'batch_size': 64,  # Number of images processed simultaneously
        'num_epochs': 1,  # Total training epochs
        'num_workers': 4,  # Parallel data loading processes
        'train_ratio': 0.8,  # 80% of data used for training
        'val_ratio': 0.1,  # 10% for validation, remaining 10% aside for testing

        'growth_rate': 16,  # Number of feature maps added per dense layer
        'drop_rate': 0.2,  # Dropout rate within dense layers
        'classifier_dropout': 0.5,  # Dropout rate in the final classification layer
        'num_classes': 2,  # Binary classification: Male (1) vs Female (0)

        # Output file paths for saving results
        'save_path': '../../Results/models/DenseNet-121_best.pth',  # Best model checkpoint
        'log_dir': '../../Results/logs/',  # Training logs directory
    }

    # ========================================================================
    # Project Information Display
    # ========================================================================

    print("\n" + "=" * 62)
    print("DenseNet-121 Gender Classification")
    print("=" * 62 + "\n")
    print("Project: COSC595 Information Technology Project: Implementation")
    print("Model: DenseNet-121")
    print("Task: Binary Gender Classification (Male/Female)")
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
        print("\nEnsure the dataset is downloaded and extracted to the specified paths.")
        return None

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("This could be due to:")
        print("- Corrupted dataset files")
        print("- Insufficient memory")
        print("- Invalid CSV format")
        return None

    # ========================================================================
    # Model Initialisation and Configuration
    # ========================================================================

    print("=" * 62)
    print("Initialising DenseNet-121 Model")
    print("=" * 62 + "\n")

    print("Model Configuration:")
    print(f"\tGrowth Rate: {CONFIG['growth_rate']}")
    print(f"\tDense Layer Dropout: {CONFIG['drop_rate']}")
    print(f"\tClassifier Dropout: {CONFIG['classifier_dropout']}")
    print(f"\tNumber of Classes: {CONFIG['num_classes']} (Male/Female)")
    print()

    model = DenseNet121Gender(
        growth_rate=CONFIG['growth_rate'],
        drop_rate=CONFIG['drop_rate'],
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
                print("Excellent performance! Model achieved SOTA accuracy.")
            elif test_acc >= 90.0:
                print("Good performance! Model is working well for binary classification.")
            elif test_acc >= 85.0:
                print("Moderate performance. Consider further tuning.")
            else:
                print("Poor performance... Model may need some significant adjustments.")

            print("=" * 62)

            print("DenseNet-121 training completed successfully!")
            print("Check the Results directory for saved model and logs")
            print("=" * 62)

        return results

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        print("Any progress up to this point has been saved")
        return None

    except RuntimeError as e:
        print(f"Runtime error during training: {e}")
        print("\nPossible solutions:")
        print("- Reduce batch_size in CONFIG")
        print("- Reduce num_workers if using CPU")
        print("- Ensure sufficient GPU memory")
        return None

    except Exception as e:
        print(f"Unexpected error during training: {e}")
        print("\nError traceback:")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("Starting DenseNet-121 Gender Classification Training")
    print("=" * 62)

    results = main()

    if results is not None:
        print(f"\nTraining session completed with {results['test_acc']:.2f}% test accuracy")
    else:
        print("\nTraining session failed or was interrupted")

    print("=" * 62)
    print("Script execution finished.")