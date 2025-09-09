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

    def __init__(self, img_dir, attr_file, split='train', transform=None, train_ratio=0.8, val_ratio=0.1):
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(attr_file)

        df['Male'] = df['Male'].replace({-1: 0})

        filenames = df['image_id'].values
        labels = df['Male'].values

        total_images = len(labels)
        female_count = (labels == 0).sum()
        male_count = (labels == 1).sum()
        female_pct = (female_count / total_images) * 100
        male_pct = (male_count / total_images) * 100

        indices = np.arange(len(filenames))

        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(val_ratio + (1 - train_ratio - val_ratio)),
            stratify=labels,
            random_state=42
        )

        temp_labels = labels[temp_idx]
        test_ratio = (1 - train_ratio - val_ratio) / (val_ratio + (1 - train_ratio - val_ratio))
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=test_ratio,
            stratify=temp_labels,
            random_state=42
        )

        if split == 'train':
            selected_idx = train_idx
        elif split == 'val':
            selected_idx = val_idx
        elif split == 'test':
            selected_idx = test_idx
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")

        self.filenames = filenames[selected_idx]
        self.labels = labels[selected_idx]

        split_total = len(self.labels)
        split_female = (self.labels == 0).sum()
        split_male = (self.labels == 1).sum()
        split_female_pct = (split_female / split_total) * 100
        split_male_pct = (split_male / split_total) * 100

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
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {self.filenames[idx]}: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================================
# Data Transforms
# ============================================================================

def get_transforms():

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2,
                               contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


# ============================================================================
# Data Loader Creation
# ============================================================================

def create_data_loaders(img_dir, attr_file, batch_size=128, num_workers=4, train_ratio=0.8, val_ratio=0.1):

    train_transform, val_transform = get_transforms()

    train_dataset = CelebAGenderDataset(
        img_dir=img_dir,
        attr_file=attr_file,
        split='train',
        transform=train_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    val_dataset = CelebAGenderDataset(
        img_dir=img_dir,
        attr_file=attr_file,
        split='val',
        transform=val_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    test_dataset = CelebAGenderDataset(
        img_dir=img_dir,
        attr_file=attr_file,
        split='test',
        transform=val_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader