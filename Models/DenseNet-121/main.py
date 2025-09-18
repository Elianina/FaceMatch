# ============================================================================
# Imports and Dependencies
# ============================================================================

import os
import sys
import warnings
import torch
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import DenseNet121Gender
from data_loader import create_data_loaders
from trainer import train_model
from FaceMatch.utils.log_utils import start_logging, stop_logging

warnings.filterwarnings('ignore')


# ============================================================================
# Main Training Function
# ============================================================================

def main():

    # Begin logging
    original_stdout, log_file, log_file_path = start_logging('DenseNet-121', '../../Results/logs/')

    # ========================================================================
    # Configuration Settings
    # ========================================================================

    CONFIG = {
        'img_dir': r"C:\\Users\\carlf\\Desktop\\COSC595 - Models and Datasets\\CelebA\\img_align_celeba\\img_align_celeba",
        'attr_file': r"C:\\Users\\carlf\\Desktop\\COSC595 - Models and Datasets\\CelebA\\list_attr_celeba.csv",

        'batch_size': 64,
        'num_epochs': 10,
        'num_workers': 4,
        'train_ratio': 0.8,
        'val_ratio': 0.1,

        'growth_rate': 16,
        'drop_rate': 0.2,
        'classifier_dropout': 0.5,
        'num_classes': 2,

        'save_path': '../../Results/models/DenseNet-121_best.pth',
        'log_dir': '../../Results/logs/',
        'plots_dir': '../../Results/plots/',
    }

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
    model_size_mb = total_params * 4 / 1024 ** 2

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
        os.makedirs(CONFIG['plots_dir'], exist_ok=True)
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

        if results is not None:
            # ====================================================================
            # Final Results Summary and Analysis
            # ====================================================================

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
            print(f"Final Test Precision: {results['test_precision']:.4f}")
            print(f"Final Test Recall: {results['test_recall']:.4f}")
            print(f"Final Test F1-Score: {results['test_f1']:.4f}")
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
            print("Check the Results directory for saved model, logs, and plots")
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
        print(f"Runtime error during training: {e}")
        print("\nPossible solutions:")
        print("- Reduce batch_size in CONFIG")
        print("- Reduce num_workers if using CPU")
        print("- Ensure sufficient GPU memory")

        stop_logging(original_stdout, log_file, log_file_path)  # Logging

        return None

    except Exception as e:
        print(f"Unexpected error during training: {e}")
        print("\nError traceback:")
        traceback.print_exc()

        stop_logging(original_stdout, log_file, log_file_path)  # Logging

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