# ============================================================================
# Imports and Dependencies
# ============================================================================

import sys
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# ============================================================================
# Training Setup (Optimized for VGG16 with Batch Normalization)
# ============================================================================

def setup_training(model, device):
    """
    Setup training components optimized for VGG16 with batch normalization.
    """

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,  # Standard learning rate for VGG with batch norm
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    criterion = nn.CrossEntropyLoss()

    return optimizer, scheduler, criterion


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """
    Train epoch function for VGG16.
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


# ============================================================================
# Validation and Evaluation Functions with Comprehensive Metrics
# ============================================================================

def validate_with_metrics(model, val_loader, criterion, device):
    """
    Enhanced validation function with comprehensive metrics tracking.
    """
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation', leave=True, colour='red',
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}]')

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({
                'Loss': f'{running_loss / (progress_bar.n + 1):.4f}',
                'Acc': f'{100. * accuracy_score(all_labels, all_predictions):.2f}%'
            })

        progress_bar.colour = 'green'
        progress_bar.refresh()
        progress_bar.close()

    epoch_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')

    return epoch_loss, accuracy * 100, precision, recall, f1


def validate(model, val_loader, criterion, device):
    """
    Standard validation function for backward compatibility.
    """
    epoch_loss, epoch_acc, _, _, _ = validate_with_metrics(model, val_loader, criterion, device)
    return epoch_loss, epoch_acc


def evaluate_model(model, test_loader, criterion, device):
    """
    Standard evaluation function for backward compatibility.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Testing', leave=True, colour='red',
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}]')

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
    Enhanced training pipeline with comprehensive metrics tracking and evaluation.
    """
    optimizer, scheduler, criterion = setup_training(model, device)

    print(f"Starting training for {config['num_epochs']} epochs")
    print("Note: Using VGG16 with batch normalization")
    print()

    best_val_acc = 0.0
    best_epoch = 0

    # Enhanced history tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }

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

        val_loss, val_acc, val_precision, val_recall, val_f1 = validate_with_metrics(
            model, val_loader, criterion, device)

        # Store metrics in history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

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
                'config': config,
                'history': history
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

    # ========================================================================
    # Final Evaluation (Enhanced with Comprehensive Analysis)
    # ========================================================================

    print("=" * 62)
    print("Generating Training Visualisations")
    print("=" * 62 + "\n")

    from metrics_utils import evaluate_model_comprehensive

    model.load_state_dict(torch.load(config['save_path'])['model_state_dict'])

    results = evaluate_model_comprehensive(
        model=model,
        test_loader=test_loader,
        device=device,
        criterion=criterion,
        model_name='VGG-16',
        save_dir=config.get('plots_dir', '/home/localadmin/jupyter/outputs/'),
        history=history
    )

    print()
    print("All plots generated successfully!")
    print()

    print("=" * 62)
    print("Testing VGG-16 model...")
    print("=" * 62 + "\n")
    print("Evaluation Complete" + "\n")

    test_acc = results['accuracy'] * 100
    test_loss = results['loss']
    test_precision = results['precision']
    test_recall = results['recall']
    test_f1 = results['f1_score']

    print(f"Test Accuracy: {test_acc:.2f}%" + "\n")
    print(f"Test Precision: {test_precision:.4f}   |   Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}   |   Test ROC-AUC: {results['roc_auc']:.4f}" + "\n")

    print("=" * 62)
    print("=         VGG-16 MODEL CYCLE COMPLETE         =")
    print("=" * 62)

    return {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_roc_auc': results['roc_auc'],
        'history': history
    }