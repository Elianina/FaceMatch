# ============================================================================
# Imports and Dependencies
# ============================================================================

import sys
import time
import torch
import torch.nn as nn
from tqdm import tqdm


# ============================================================================
# Training Setup (Optimized for Regular VGG16 without Batch Normalization)
# ============================================================================

def setup_training(model, device):
    """
    Setup training components optimized for regular VGG16 (without batch norm).
    Regular VGG requires more careful learning rate and weight decay settings.
    """

    # Lower learning rate for regular VGG16 (no batch norm stabilization)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4,  # Reduced from 1e-3 for stability without batch norm
        betas=(0.9, 0.999),
        weight_decay=5e-4  # Increased weight decay for regularization
    )

    # More conservative scheduler for regular VGG16
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=30,  # Reduce LR every 30 epochs
        gamma=0.1      # Multiply LR by 0.1
    )

    criterion = nn.CrossEntropyLoss()

    return optimizer, scheduler, criterion


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """
    Train epoch function optimized for regular VGG16.
    """

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} Training', leave=True, colour='red',
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]')

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping for stability (important for regular VGG)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix({
            'Loss': f'{running_loss / (progress_bar.n + 1):.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })

    progress_bar.colour = 'green'
    progress_bar.refresh()
    progress_bar.close()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Validation function - identical structure to original trainer.
    """

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation', leave=True, colour='red',
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]')

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'Loss': f'{running_loss / (progress_bar.n + 1):.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        progress_bar.colour = 'green'
        progress_bar.refresh()
        progress_bar.close()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluation function - identical structure to original trainer.
    """

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Testing', leave=True, colour='red',
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]')

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'Loss': f'{running_loss / (progress_bar.n + 1):.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        progress_bar.colour = 'green'
        progress_bar.refresh()
        progress_bar.close()

    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


# ============================================================================
# Main Training Pipeline
# ============================================================================

def train_model(model, train_loader, val_loader, test_loader, device, config):
    """
    Main training pipeline optimized for regular VGG16.
    """

    optimizer, scheduler, criterion = setup_training(model, device)

    print(f"Starting training for {config['num_epochs']} epochs")
    print("Note: Using optimized settings for regular VGG16 (no batch normalization)")
    print()
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, config['num_epochs'] + 1):
        print("-" * 62)
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print("-" * 62 + "\n")

        sys.stdout.flush()
        time.sleep(0.3)

        train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                            criterion, device, epoch)

        sys.stdout.flush()
        time.sleep(0.2)

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print()
        print()
        if epoch == 1:
            print(f"Epoch {epoch} Results:")
            print(f"\tTraining Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
            print(f"\tValidation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
            print(f"\tLearning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            print(f"Epoch {epoch} Results:")
            print(f"\tTraining Loss (TL): {train_loss:.4f}, Training Accuracy (TA): {train_acc:.2f}%")
            print(f"\tValidation Loss (VL): {val_loss:.4f}, Validation Accuracy (VA): {val_acc:.2f}%")
            print(f"\tLearning Rate (LR): {optimizer.param_groups[0]['lr']:.6f}")

        print()
        print()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, config['save_path'])
            print(f"Validation Accuracy: {val_acc:.2f}%   |   \033[92mThe new best model - Saved!\033[0m")
        else:
            print(f"Validation Accuracy: {val_acc:.2f}%   |   ")

        print()
        print()

        sys.stdout.flush()
        time.sleep(0.1)

    print("=" * 62)
    print("TRAINING COMPLETED")
    print("=" * 62 + "\n")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})" + "\n")

    print("=" * 62)
    print("Testing Regular VGG16 model...")
    print("=" * 62 + "\n")
    print("Evaluating on test set. Please wait." + "\n")

    model.load_state_dict(torch.load(config['save_path'])['model_state_dict'])

    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

    print()
    print("Evaluation Complete" "\n")
    print(f"Test Loss: {test_loss:.4f}   |   Test Acc: {test_acc:.2f}%" + "\n")

    print("=" * 61)
    print("=         REGULAR VGG16 MODEL CYCLE COMPLETE         =")
    print("=" * 61)

    return {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'test_loss': test_loss
    }