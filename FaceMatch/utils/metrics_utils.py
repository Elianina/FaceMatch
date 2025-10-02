# ============================================================================
# Imports and Dependencies
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_curve, auc, classification_report
)


# ============================================================================
# Classification Metrics
# ============================================================================

def calculate_classification_metrics(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    return accuracy, precision, recall, f1


def calculate_binary_metrics(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)

    metrics = {
        'accuracy': accuracy,
        'precision_female': precision[0],
        'precision_male': precision[1],
        'recall_female': recall[0],
        'recall_male': recall[1],
        'f1_female': f1[0],
        'f1_male': f1[1],
        'support_female': support[0],
        'support_male': support[1]
    }

    return metrics


# ============================================================================
# Training Curves Plotting
# ============================================================================

def plot_training_curves(history, save_dir, model_name):

    train_losses = history['train_loss']
    val_losses = history['val_loss']
    val_accuracies = history['val_accuracy']

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', linewidth=2.5, label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', linewidth=2.5, label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, len(train_losses))

    # Accuracy curves
    ax2.plot(epochs, val_accuracies, 'g-', linewidth=2.5, label='Validation Accuracy', marker='o', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'{model_name} Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(val_accuracies))
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_dir:
        plt.savefig(f'{save_dir}/{model_name}_training_curves.png', dpi=300, bbox_inches='tight', facecolor='white')

    plt.show()


# ============================================================================
# Metrics Summary Bar Chart
# ============================================================================

def plot_metrics_summary(metrics_dict, save_path, model_name):

    plt.figure(figsize=(10, 6))

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        metrics_dict['accuracy'],
        metrics_dict['precision'],
        metrics_dict['recall'],
        metrics_dict['f1_score']
    ]

    colours = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    bars = plt.bar(metrics, values, color=colours, alpha=0.8, edgecolor='black', linewidth=1.2, width=0.6)

    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.xlabel('Performance Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title(f'{model_name} Performance Summary', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')

    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    plt.show()


# ============================================================================
# Confusion Matrix Functions
# ============================================================================

def calculate_confusion_matrix(y_true, y_pred, class_names=None):

    if class_names is None:
        class_names = ['Female', 'Male']

    cm = confusion_matrix(y_true, y_pred)

    return cm, class_names


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, model_name='Model'):

    if class_names is None:
        class_names = ['Female', 'Male']

    cm, _ = calculate_confusion_matrix(y_true, y_pred, class_names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.title(f'{model_name} Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    accuracy = accuracy_score(y_true, y_pred)
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.3f}',
             transform=plt.gca().transAxes, ha='center', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    plt.show()


# ============================================================================
# ROC Curve Functions
# ============================================================================

def calculate_roc_curve(y_true, y_scores):

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def plot_roc_curve(y_true, y_scores, save_path=None, model_name='Model'):

    fpr, tpr, roc_auc = calculate_roc_curve(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name} ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    plt.show()

    return roc_auc


# ============================================================================
# Model Evaluation
# ============================================================================

def evaluate_model_comprehensive(model, test_loader, device, criterion=None, model_name='Model', save_dir=None,
                                 history=None):

    print("Starting comprehensive evaluation...")

    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    total_loss = 0.0
    num_batches = 0

    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1

            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            probabilities = F.softmax(outputs, dim=1)[:, 1]
            all_probabilities.extend(probabilities.cpu().numpy())

    print("Model evaluation completed, calculating metrics...")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    accuracy, precision, recall, f1 = calculate_classification_metrics(all_labels, all_predictions)
    binary_metrics = calculate_binary_metrics(all_labels, all_predictions)

    print("Metrics calculated, generating plots...")

    if save_dir:
        try:
            if history:
                print("Generating training curves...")
                plot_training_curves(history, save_dir, model_name)

            print("Generating confusion matrix...")
            confusion_path = f'{save_dir}/{model_name}_confusion_matrix.png'
            plot_confusion_matrix(all_labels, all_predictions, save_path=confusion_path, model_name=model_name)

            print("Generating ROC curve...")
            roc_path = f'{save_dir}/{model_name}_roc_curve.png'
            roc_auc = plot_roc_curve(all_labels, all_probabilities, save_path=roc_path, model_name=model_name)

            print("Generating metrics summary...")
            metrics_dict = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            metrics_path = f'{save_dir}/{model_name}_metrics_summary.png'
            plot_metrics_summary(metrics_dict, metrics_path, model_name)

            print("All plots generated successfully!")

        except Exception as e:
            print(f"Error generating plots: {e}")
            roc_auc = calculate_roc_curve(all_labels, all_probabilities)[2]
    else:
        roc_auc = calculate_roc_curve(all_labels, all_probabilities)[2]

    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'binary_metrics': binary_metrics,
        'predictions': all_predictions,
        'true_labels': all_labels,
        'probabilities': all_probabilities
    }

    return results


# ============================================================================
# Classification Report Functions
# ============================================================================

def generate_classification_report(y_true, y_pred, class_names=None, save_path=None):

    if class_names is None:
        class_names = ['Female', 'Male']

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    if save_path:
        with open(save_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
            f.write("\n\nConfusion Matrix\n")
            f.write("-" * 20 + "\n")
            cm, _ = calculate_confusion_matrix(y_true, y_pred, class_names)
            f.write(f"True\\Pred   {class_names[0]:<8} {class_names[1]:<8}\n")
            f.write(f"{class_names[0]:<10} {cm[0, 0]:<8} {cm[0, 1]:<8}\n")
            f.write(f"{class_names[1]:<10} {cm[1, 0]:<8} {cm[1, 1]:<8}\n")

    return report