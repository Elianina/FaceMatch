"""
The CelebrityFaces Attributes (CelebA) Dataset Loader file for Gendered Face Recognition.

This file handles the loading and preprocessing of the CelebA dataset for a binary gender classification task.
It provides stratified train/validation/test data splitting and batch loading for training Convolutional
Neural Network (CNN) models in this particular task.

The file's features include:
- Stratified train/validation/test data splitting to maintain class balance, and
- Data augmentation for training data:
    - This includes 1. horizontal flipping, 2. rotation, and 3. colour jitter

Required Files:
- "img_align_celeba": Directory containing CelebA's images (labelled: 000001.jpg, 000002.jpg, ...jpg)
- "list_attr_celeba": The attribute file with 'image_id' and 'Male' columns

Authors: Carl Fokstuen, YuTing Lee, Mark Malady, Nayani Samaranayake, Vishal Cheroor Ravi
Course: COSC595 Information Technology Project - Implementation
Institution: The University of New England (UNE)
Date: September, 2025

Dataset Attribution: Li, J. (2018). CelebFaces Attributes (CelebA) Dataset. Kaggle.
                     https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
"""


# ============================================================================
# Imports and Dependencies
# ============================================================================

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================================
# CelebA Dataset Implementation
# ============================================================================

class CelebAGenderDataset(Dataset):
    """
    CelebA dataset for gender classification.

    Loads CelebA's images and gender labels from the .csv file. Supports stratified
    train/validation/test splits that maintain an even gender distribution.

    Args:
        img_dir (str): The directory containing CelebA images (`img_align_celeba/`)
        attr_file (str): Path to the CSV attributes file (`list_attr_celeba.csv`)
        split (str): Dataset split - 'train', 'val', or 'test'. But does default to 'train'.
        transform (callable/optional): The image transformations to apply
        train_ratio (float): A proportion of data for training (80%)
        val_ratio (float): A proportion of data for validation (10%)

    Note:
        Gender encoding: CSV values are -1 (female) transformed to 0, and 1 (male) remains 1
    """
    def __init__(self, img_dir, attr_file, split='train', transform=None, train_ratio=0.8, val_ratio=0.1):
        self.img_dir = img_dir
        self.transform = transform

        # Loads the .csv file containing CelebA image attributes
        df = pd.read_csv(attr_file)

        # Converts the gender labels for the female class (see Note above)
        df['Male'] = df['Male'].replace({-1: 0})

        # Extracts the image's filenames and corresponding gender labels
        filenames = df['image_id'].values
        labels = df['Male'].values

        # Calculates the dataset's overall statistics for reporting purposes
        total_images = len(labels)
        female_count = (labels == 0).sum()
        male_count = (labels == 1).sum()
        female_pct = (female_count / total_images) * 100
        male_pct = (male_count / total_images) * 100

        # Creates an array of indices for stratified splitting
        indices = np.arange(len(filenames))

        # The first split separates training data from validation+testing data
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(val_ratio + (1 - train_ratio - val_ratio)),  # Combined val + test size
            stratify=labels,                                        # Maintain a gendered distribution across splits
            random_state=42                                         # Ensure for reproducible splits
        )

        # The second split separates validation and testing data from the remaining data
        temp_labels = labels[temp_idx]                              # Labels for the temporary (val+test) subset
        test_ratio = (1 - train_ratio - val_ratio) / (val_ratio + (1 - train_ratio - val_ratio))
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=test_ratio,
            stratify=temp_labels,                                   # Maintain the gendered distribution
            random_state=42
        )

        # Selects the appropriate indices based on the prior requested split
        if split == 'train':
            selected_idx = train_idx
        elif split == 'val':
            selected_idx = val_idx
        elif split == 'test':
            selected_idx = test_idx
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")

        # Stores the filenames and labels for this split
        self.filenames = filenames[selected_idx]
        self.labels = labels[selected_idx]

        # Calculates the statistics for the current split
        split_total = len(self.labels)
        split_female = (self.labels == 0).sum()
        split_male = (self.labels == 1).sum()
        split_female_pct = (split_female / split_total) * 100
        split_male_pct = (split_male / split_total) * 100

        # Prints the dataset's information, which is for the purpose of user feedback during training
        if split == 'train':
            print("Loading dataset attributes from:")
            print(f"\t{attr_file}" + "\n")
            print(f"Loaded Images from CSV file:")
            print(f"\t{total_images} images" + "\n")
            print(f"Overall Gender Statistics:")
            print(f"\tFemale: {female_count} ({female_pct:.1f}% Female),")
            print(f"\tMale: {male_count} ({male_pct:.1f}% Male)" + "\n")
            print("-" * 62)
            print(f"TRAINING Set: {split_total} images ({train_ratio * 100:.0f}%)")
            print("-" * 62 + "\n")
            print(f"Training Gender Distribution:")
            print(f"\tFemale: {split_female} ({split_female_pct:.1f}% Female),")
            print(f"\tMale: {split_male} ({split_male_pct:.1f}% Male)" + "\n")
        elif split == 'val':
            print("-" * 62)
            print(f"VALIDATION Set: {split_total} images ({val_ratio * 100:.0f}%)")
            print("-" * 62 + "\n")
            print(f"Validation Gender Distribution:")
            print(f"\tFemale: {split_female} ({split_female_pct:.1f}% Female),")
            print(f"\tMale: {split_male} ({split_male_pct:.1f}% Male)" + "\n")
        elif split == 'test':
            test_ratio_actual = (1 - train_ratio - val_ratio)
            print("-" * 62)
            print(f"TESTING Set: {split_total} images ({test_ratio_actual * 100:.0f}%)")
            print("-" * 62 + "\n")
            print(f"Testing Gender distribution:")
            print(f"\tFemale: {split_female} ({split_female_pct:.1f}% Female),")
            print(f"\tMale: {split_male} ({split_male_pct:.1f}% Male)" + "\n")

    def __len__(self):
        """Returns the total number of samples in this dataset split."""
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve

        Returns:
            tuple: (image, label) where an image is a PIL Image or tensor (if it is transformed)
                   and the label is an integer (0=female, 1=male)
        """
        # Constructs the full path to the image file
        img_path = os.path.join(self.img_dir, self.filenames[idx])

        try:
            # Loads an image and ensures it is in Red, Green, Blue (RGB) format
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Handles any corrupted or missing images with a black placeholder image
            print(f"Error loading image {self.filenames[idx]}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        # Gets the associated gender label for this image
        label = self.labels[idx]
        # Applies any specified transformations such as resize, normalise, or augment
        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================================
# Data Transforms
# ============================================================================

def get_transforms():
    """The function creates training and validation transforms on the CelebA dataset."""

    train_transform = transforms.Compose([       # Targets training set augmentation/s
        transforms.Resize((224, 224)),           # Resize image to 224x224 pixels
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of image flip to horizontal plane
        transforms.RandomRotation(10),           # Random rotation of +/- 10 degrees (simulate a small head tilt)
        transforms.ColorJitter(brightness=0.2,   # Introduce small variances in image lighting
                               contrast=0.2),
        transforms.ToTensor(),                   # Convert image to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean/std normalisation
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([         # Targets validation set augmentation/s
        transforms.Resize((224, 224)),           # Resize image to 224x224 pixels
        transforms.ToTensor(),                   # Convert image to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean/std normalisation
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


# ============================================================================
# Data Loader Creation
# ============================================================================

def create_data_loaders(img_dir, attr_file, batch_size=128, num_workers=4, train_ratio=0.8, val_ratio=0.1):
    """
    A function to create stratified data loaders for the CelebA gender classification problem.

        Args:
            img_dir (str): The path to the directory containing CelebA images
            attr_file (str): The path to the CSV file with gender attributes
            batch_size (int): Batch size for the data loaders (128 images per batch)
            num_workers (int): Number of worker processes for data loading (4 CPU cores)
            train_ratio (float): Proportion of data reserved for training (80%)
            val_ratio (float): Proportion of data reserved for validation (10%)

        Returns:
            tuple: (train_loader, val_loader, test_loader) - DataLoader objects for
                   training, validation, and testing with stratified gender splits.
        """

    # Gets the data augmentation transforms for the training batch, and standard transforms for validation/test batches
    train_transform, val_transform = get_transforms()

    # Creates stratified datasets for each train/val/test split
    train_dataset = CelebAGenderDataset(
        img_dir=img_dir,
        attr_file=attr_file,
        split='train',                    # Training batch - 80% of data with an augmentation
        transform=train_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    val_dataset = CelebAGenderDataset(
        img_dir=img_dir,
        attr_file=attr_file,
        split='val',                      # Validation batch - 10% of data - no augmentation
        transform=val_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    test_dataset = CelebAGenderDataset(
        img_dir=img_dir,
        attr_file=attr_file,
        split='test',                     # Remaining 10% of data forming a testing batch - no augmentation
        transform=val_transform,          # Use validation transforms (no augmentation)
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    # Creates the data loader for each stratified dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,                                # Randomise training order
                              num_workers=num_workers, pin_memory=True)    # Optimised for GPU memory transfer
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False,                                 # Non-randomised validation order
                            num_workers=num_workers, pin_memory=True)      # Optimised for GPU memory transfer
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False,                                # Non-randomised testing order
                             num_workers=num_workers, pin_memory=True)     # Optimised for GPU memory transfer

    # Returns DataLoader objects for training, validation, and testing batches
    return train_loader, val_loader, test_loader