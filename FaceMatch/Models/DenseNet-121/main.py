"""
Main Training Script for DenseNet-121 Gendered/Facial Classification on CelebA Dataset.

This file handles the training pipeline for the binary gendered classification problem using
the DenseNet-121 architecture on the CelebA (Celebrity Faces Attributes) dataset. It is
the primary execution point for model training, evaluation, and results generation.

Authors: Carl Fokstuen, YuTing Lee, Mark Malady, Nayani Samaranayake, Vishal Cheroor Ravi
Course: COSC595 Information Technology Project - Implementation
Institution: The University of New England (UNE)
Date: September, 2025

Densenet-121 Model Reference:
    Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2018).
        Densely Connected Convolutional Networks. arXiv:1608.06993v5.
"""


# ============================================================================
# Imports and Dependencies
# ============================================================================

import os
import sys
import warnings
import torch
import traceback

# Adds the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.dirname(current_dir)
facematch_dir = os.path.dirname(models_dir)
project_root = os.path.dirname(facematch_dir)
sys.path.insert(0, project_root)

# Densenet-121 modules
from model import DenseNet121Gender
from trainer import train_model

# Utility modules
# from FaceMatch.utils.celeba_api import celeba_api_function

from FaceMatch.utils.data_loader import create_data_loaders
from FaceMatch.utils.image_eval_utils import image_eval_utils
from FaceMatch.utils.log_utils import start_logging, stop_logging

# Suppress any non-critical warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    """
    The main training function for the DenseNet-121 model on CelebA dataset.

    This function manages the training pipeline. It includes:
    - Dataset loading with stratified train/validation/test splits
    - Model initialisation with parameters
    - Training loop with validation and model checkpoints
    - Logging of training progress and results

    The function uses a centralised configuration dictionary to manage all
    parameters, file paths, and training settings.

    The function's configuration (`CONFIG`) includes:
    - Dataset paths and loading parameters
    - Model architecture and training parameters
    - Output directories for models, logs, and plots

    Returns:
        None: The model's results are saved to disk and logged to files

    Raises:
        `FileNotFoundError`: If the dataset files or directories do not exist
        `RuntimeError`: If the CUDA/GPU setup fails
        `Exception`: Or for any other training-related errors
    """
    # Initialise logging and save to `Results/logs/DenseNet-121_training_log.txt`
    original_stdout, log_file, log_file_path = start_logging('DenseNet-121', '../../Results/logs/')


    # ========================================================================
    # Configuration Settings
    # ========================================================================

    CONFIG = {                                           # Dataset configuration
        # Paths to CelebA dataset files and directories
        'img_dir': r"",
        'attr_file': r"",

       # Training parameters
        'batch_size': 64,                                # The balance between memory usage/gradient stability
        'num_epochs': 40,                                # Training epochs (40 epoches)
        'num_workers': 12,                                # CPU workers for data loading (adjust according to resources)
        'train_ratio': 0.8,                              # 80% of data for training
        'val_ratio': 0.1,                                # 10% for validation, 10% remaining for testing

        # Model parameters
        'growth_rate': 32,                               # Feature map growth rate (k)
        'drop_rate': 0.2,                                # Dropout rate in dense layers
        'num_classes': 2,                                # Binary classification: female (0) vs male (1)

        # Output paths
        'save_path': '../../Results/models/DenseNet-121_best.pth',  # Best model epoch
        'log_dir': '../../Results/logs/',                           # Training logs
        'plots_dir': '../../Results/plots/',                        # Plots
    }
    # Displays a formatted training session header
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

    # Configures a compute device (GPU if available, otherwise CPU) and then displays hardware information to the user
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device Configuration:")
    print(f"\tUtilising Device: {str(device).upper()}")

    # Displays the GPUs specifications if CUDA is available, otherwise show a CPU warning
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024 ** 3
        print(f"\tGPU Name: {gpu_name}")
        print(f"\tGPU Memory: {gpu_memory} GB")
        torch.cuda.empty_cache()  # Clears any cached GPU memory
    else:
        print("\tNote: Training on CPU will be significantly slower")

    print("\n")

    # ========================================================================
    # Dataset Loading and Preprocessing
    # ========================================================================

    # The Dataset Loading and Preprocessing section initialises the CelebA data loaders
    # with stratified splits, and creates the PyTorch DataLoader objects that handle
    # 1. Loading images from the disk in batches, 2. Applies any data transformations
    # (resize, normalise, augment), 3. Maintains a stratified gendered distribution
    # across the splits, and 4. Parallel data loading using multiple worker processes

    print("---------------")
    print("Loading Dataset")
    print("---------------\n")

    try:
        # Creates stratified data loaders using the custom CelebAGenderDataset class
        train_loader, val_loader, test_loader = create_data_loaders(
            img_dir=CONFIG['img_dir'],                     # Directory containing CelebA images
            attr_file=CONFIG['attr_file'],                 # .csv file with gendered attributes
            batch_size=CONFIG['batch_size'],               # Images per batch
            num_workers=CONFIG['num_workers'],             # Number of CPU processes
            train_ratio=CONFIG['train_ratio'],             # Proportion for training (80%)
            val_ratio=CONFIG['val_ratio'],                 # Proportion for validation (10%), the remainder for test
            image_size = 224
        )
        print("Dataset loaded successfully!\n")

    except FileNotFoundError as e:
        # Handles any missing dataset files with user feedback
        print(f"Dataset files not found: {e}")
        print("\nPlease update the file paths in CONFIG section:")
        print(f"   - img_dir: {CONFIG['img_dir']}")
        print(f"   - attr_file: {CONFIG['attr_file']}")
        print("\nEnsure the dataset is downloaded and extracted to the specified paths.")
        return None                                        # Graceful exit

    except Exception as e:
        # A catch-all for any other dataset loading issues with diagnostic user feedback
        print(f"Error loading dataset: {e}")
        print("This could be due to:")
        print("- Corrupted dataset files")
        print("- Insufficient memory")
        print("- Invalid CSV format")
        return None                                        # Graceful exit

    # ========================================================================
    # Model Initialisation and Configuration
    # ========================================================================

    # The Model Initialisation and Configuration section displays the model initialisation
    # header to the user with configuration details, and creates the DenseNet-121 architecture
    # reporting on key statistics for monitoring any computational requirements and memory usage
    print("=" * 62)
    print("Initialising DenseNet-121 Model")
    print("=" * 62 + "\n")

    # Prints a model configuration summary
    print("Model Configuration:")
    print(f"\tGrowth Rate: {CONFIG['growth_rate']}")                      # Feature map growth per layer
    print(f"\tDense Layer Dropout: {CONFIG['drop_rate']}")                # Regularisation within dense blocks
    print(f"\tNumber of Classes: {CONFIG['num_classes']} (Male/Female)")  # Binary classification output (M/F)
    print()

    # Instantiates the DenseNet-121 model with its hyperparameters, and
    # `.to(device)` moves model parameters to GPU/CPU as configured prior
    model = DenseNet121Gender(
        growth_rate=CONFIG['growth_rate'],                                # Controls the feature map expansion per layer
        drop_rate=CONFIG['drop_rate'],                                    # Dropout probability for dense layers (0.2)
        num_classes=CONFIG['num_classes']                                 # Binary output for M/F classification
    ).to(device)

    # Calculates and displays the model's complexity metrics
    # These statistics exist to help assess any computational and memory requirements
    total_params = sum(p.numel() for p in model.parameters())             # All parameters (weights + biases)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # Parameters that will be updated
    model_size_mb = total_params * 4 / 1024 ** 2                          # Size in MB (4 bytes per float32)

    print(f"Model Statistics:")
    print(f"\tTotal Parameters: {total_params:,}")                                   # Total DN-121 parameters
    print(f"\tTrainable Parameters: {trainable_params:,}")                           # Should match total
    print(f"\tModel Size: {model_size_mb:.1f} MB")                                   # Model's file size when saved
    print(f"\tMemory Footprint: ~{model_size_mb * 2:.0f} MB (including gradients)")  # Training memory estimate
    print()

    # ========================================================================
    # Training Execution
    # ========================================================================

    # Display the training phase header
    print("=" * 62)
    print("Starting Training Phase")
    print("=" * 62 + "\n")

    # Creates the required output directories with error handling. These directories
    # store model checkpoints, training logs, and plots.
    # Note: Using `exist_ok=True` prevents any errors if directories already exist.
    try:
        os.makedirs(os.path.dirname(CONFIG['save_path']), exist_ok=True)      # Model checkpoint directory
        os.makedirs(CONFIG['log_dir'], exist_ok=True)                         # Training logs directory
        os.makedirs(CONFIG['plots_dir'], exist_ok=True)                       # Plots directory
        print("Output directories verified/created successfully")
    except Exception as e:
        # Gracefully handles any filesystem permission issues or invalid paths
        print(f"Warning: Could not create output directories: {e}")
        print("Model saving may fail")

    print()

    # Executes the entire training pipeline with error handling and passes
    # initialised components to the training function
    try:
        results = train_model(
            model=model,                                                      # DenseNet-121 instance
            train_loader=train_loader,                                        # Training data with augmentation
            val_loader=val_loader,                                            # Validation data for model's selection
            test_loader=test_loader,                                          # Test data for the final evaluation
            device=device,                                                    # GPU/CPU device configuration
            config=CONFIG                                                     # Entire hyperparameter configuration
        )

        # Verify that training has completed successfully before proceeding to processing results
        if results is not None:

            # Final results summary and analysis
            print("\n" + "=" * 62)
            print("FINAL RESULTS SUMMARY")
            print("=" * 62)

            # Extracts key performance metrics from the training results dictionary
            best_val_acc = results['best_val_acc']                            # Best validation accuracy achieved
            best_epoch = results['best_epoch']                                # Epoch where the best validation occurred
            test_acc = results['test_acc']                                    # Final test set accuracy
            test_loss = results['test_loss']                                  # Final test set loss

            # Displays a performance summary to terminal and logs
            print(f"Best Validation Accuracy: {best_val_acc:.2f}% (achieved at Epoch {best_epoch})")
            print(f"Final Test Accuracy: {test_acc:.2f}%")
            print(f"Final Test Loss: {test_loss:.4f}")
            print(f"Final Test Precision: {results['test_precision']:.4f}")
            print(f"Final Test Recall: {results['test_recall']:.4f}")
            print(f"Final Test F1-Score: {results['test_f1']:.4f}")
            print(f"Model saved to: {CONFIG['save_path']}")

            # An automated performance assessment - purely for informational/feedback purposes
            # Note: Can be removed if not required
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

            # Training completion
            print("DenseNet-121 training completed successfully!")
            print("Check the Results directory for saved model, logs, and plots")
            print("=" * 62)

        # ====================================================================
        # Misclassification Analysis
        # ====================================================================

        # Get test dataset from the loader
        test_dataset = test_loader.dataset

        # Track misclassified images
        misclassified_images = image_eval_utils(
            model=model,
            test_loader=test_loader,
            device=device,
            test_dataset=test_dataset,
            save_dir='../../Results/misclassified',
            model_name = 'DenseNet-121'
        )

        print("=" * 62)

        # Stop logging
        stop_logging(original_stdout, log_file, log_file_path)

        return results

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        print("Any progress up to this point has been saved")

        stop_logging(original_stdout, log_file, log_file_path)  # Logging cleanup

        return None

    except RuntimeError as e:
        # Handles training errors with user feedback and logging
        print(f"Runtime error during training: {e}")
        print("\nPossible solutions:")
        print("- Reduce batch_size in CONFIG")
        print("- Reduce num_workers if using CPU")
        print("- Ensure sufficient GPU memory")

        stop_logging(original_stdout, log_file, log_file_path)  # Logging cleanup

        return None

    except Exception as e:
        # A catch-all for other errors with user feedback and logging
        print(f"Unexpected error during training: {e}")
        print("\nError traceback:")
        traceback.print_exc()

        stop_logging(original_stdout, log_file, log_file_path)  # Logging cleanup

        return None


if __name__ == "__main__":
    # The script's entry point
    print("Starting DenseNet-121 Gender Classification Training")
    print("=" * 62)

    # Executes the main training pipeline and captures results
    results = main()

    # Provides a final execution status summary for user feedback
    if results is not None:
        # Reports on final accuracy
        print(f"\nTraining session completed with {results['test_acc']:.2f}% test accuracy")
    else:
        # Informs the user of any failure
        print("\nTraining session failed or was interrupted")

    # A banner presenting an execution
    print("=" * 62)
    print("Script execution finished.")