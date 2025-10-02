"""
Image Misclassification for Trace for Gendered Classification Models

This module provides utility for analysing and tracing any misclassified images
within in gendered classification tasks. It evaluates the trained models on test datasets,
identifies any misclassified samples, and then generates detailed .csv reports with
confidence scores.

Authors: Carl Fokstuen, YuTing Lee, Mark Malady, Nayani Samaranayake, Vishal Cheroor Ravi
Course: COSC595 Information Technology Project - Implementation
Institution: The University of New England (UNE)
Date: September, 2025
"""


# ============================================================================
# Imports and Dependencies
# ============================================================================

import os
import csv
import torch
from tqdm import tqdm
from datetime import datetime


# ============================================================================
# Misclassification Analysis and Tracing Function
# ============================================================================

def image_eval_utils(model, test_loader, device, test_dataset,
                     save_dir='../../Results/misclassified', model_name='Model'):
    """
    Traces and saves misclassified images for a single model.

    Args:
        model: Trained model to evaluate
        test_loader: DataLoader for test set
        device: torch device (cuda/cpu)
        test_dataset: Dataset object with filenames attribute
        save_dir: Directory to save misclassification report
        model_name: Name of the model for CSV filename (e.g., 'DenseNet-121', 'EfficientNet-B4')

    Returns:
        list: List of misclassified image information
    """

    # Creates an output directory if one doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Sets the model to evaluation mode
    model.eval()

    # Initialises a list to store any misclassified information
    misclassified = []

    # Prints an analysis header
    print("=" * 62)
    print("ANALYZING MISCLASSIFIED IMAGES")
    print("=" * 62)
    print()

    # Disables the gradient computation evaluation
    with torch.no_grad():
        batch_idx = 0  # Traces the current batch for indexing the dataset

        # Creates a progress bar for user feedback during evaluation
        progress_bar = tqdm(test_loader, desc='Analyzing', colour='blue')

        # Iterates through all batches within the test set
        for images, labels in progress_bar:
            # Moves the data to the specified device (GPU/CPU)
            images, labels = images.to(device), labels.to(device)

            # Gets the model predictions via forward pass
            outputs = model(images)

            # Converts probabilities using softmax
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Retrieves the predicted class indices (0=Female, 1=Male)
            _, predicted = torch.max(outputs, 1)

            # Creates a boolean mask that identifies any misclassified samples within this batch
            misclassified_mask = predicted != labels

            # Processes each misclassified sample within the current batch
            for i, is_misclassified in enumerate(misclassified_mask):
                if is_misclassified:
                    # Calculates the index within the original dataset
                    dataset_idx = batch_idx * test_loader.batch_size + i

                    # Retrieves the filename from the dataset
                    filename = test_dataset.filenames[dataset_idx]

                    # Extracts the true and predicted labels
                    true_label = labels[i].item()  # Converts tensor to Python int
                    pred_label = predicted[i].item()

                    # Extracts the confidence scores for both classes
                    female_conf = probabilities[i][0].item()
                    male_conf = probabilities[i][1].item()

                    # Stores misclassification information
                    misclassified.append({
                        'filename': filename,
                        'true_label': 'Female' if true_label == 0 else 'Male',
                        'predicted_label': 'Female' if pred_label == 0 else 'Male',
                        'female_confidence': f'{female_conf:.4f}',  # Format to 4 decimal places
                        'male_confidence': f'{male_conf:.4f}'
                    })

            # Increments the batch counter for next iteration
            batch_idx += 1

        # Closes the progress bar
        progress_bar.close()

    # Generates a timestamped filename for .csv output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(save_dir, f'{model_name}_misclassified_{timestamp}.csv')

    # Saves the results and displays a summary
    if misclassified:
        # Writes the misclassification data to .csv file
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'true_label', 'predicted_label',
                          'female_confidence', 'male_confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Writes the header row
            writer.writeheader()

            # Writes all of the misclassification records
            writer.writerows(misclassified)

        # Displays the summary statistics
        print()
        print(f"Total misclassified images: {len(misclassified)}")
        print(f"Results saved to: {csv_filename}")
        print()

        # Calculates the breakdown by error type
        female_to_male = sum(1 for m in misclassified if m['true_label'] == 'Female')
        male_to_female = sum(1 for m in misclassified if m['true_label'] == 'Male')

        print("Misclassification Breakdown:")
        print(f"  Female → Male: {female_to_male} images")
        print(f"  Male → Female: {male_to_female} images")
        print()

        # Displays a sample of misclassifications for fast review
        print("Sample Misclassifications (first 5):")
        for i, item in enumerate(misclassified[:5]):
            print(f"  {i + 1}. {item['filename']}")
            print(f"     True: {item['true_label']}, Predicted: {item['predicted_label']}")
            print(f"     Confidence - Female: {item['female_confidence']}, Male: {item['male_confidence']}")

        # Indicates if there are any more misclassifications not previously shown
        if len(misclassified) > 5:
            print(f"  ... and {len(misclassified) - 5} more")
        print()
    else:
        print("No misclassifications found! Perfect accuracy!")
        print()

    # Prints a completion banner to the terminal
    print("=" * 62)
    print()

    return misclassified