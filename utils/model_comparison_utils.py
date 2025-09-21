# ============================================================================
# Model Comparisons Utility
# ============================================================================

# TODO: Implement the model comparisons functions
"""
This file will contain the functions for comparing models.
This will be achieved by reading the model logs and generating plots and metrics.
"""

import os
import re
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================================
# Metric Extraction Function
# ============================================================================

def extract_metrics_from_log(filepath):

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    model_name = os.path.basename(filepath).split('.')[0]
    epochs = re.findall(r"Epoch \d+ Results:(.*?)Validation Accuracy: .*?%", content, re.DOTALL)
    val_acc = [float(re.search(r"Validation Accuracy.*?: ([\d.]+)%", e).group(1)) for e in epochs if "Validation Accuracy" in e]
    train_acc = [float(re.search(r"Training Accuracy.*?: ([\d.]+)%", e).group(1)) for e in epochs if "Training Accuracy" in e]
    val_loss = [float(re.search(r"Validation Loss.*?: ([\d.]+)", e).group(1)) for e in epochs if "Validation Loss" in e]
    train_loss = [float(re.search(r"Training Loss.*?: ([\d.]+)", e).group(1)) for e in epochs if "Training Loss" in e]

    test_acc_match = re.search(r"Final Test Accuracy: ([\d.]+)%", content)
    test_loss_match = re.search(r"Final Test Loss: ([\d.]+)", content)
    test_acc = float(test_acc_match.group(1)) if test_acc_match else None
    test_loss = float(test_loss_match.group(1)) if test_loss_match else None

    return {
        "model": model_name,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_acc": max(val_acc) if val_acc else None,
        "final_test_acc": test_acc,
        "final_test_loss": test_loss
    }

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_validation_accuracies(models_data):
    plt.figure(figsize=(10, 6))
    for data in models_data:
        plt.plot(data["val_acc"], label=f"{data['model']}")
    plt.title("Validation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("../Results/comparison/", exist_ok=True)
    plt.savefig("../Results/comparison/comparison_validation_accuracy.png")
    plt.show()

# ============================================================================
# Summary Table Generator
# ============================================================================

def generate_summary_table(models_data):
    records = []
    for data in models_data:
        records.append({
            "Model": data["model"],
            "Best Val Accuracy (%)": data["best_val_acc"],
            "Test Accuracy (%)": data["final_test_acc"],
            "Test Loss": data["final_test_loss"]
        })
    df = pd.DataFrame(records)
    os.makedirs("../Results/comparison/", exist_ok=True)
    plt.savefig("../Results/comparison/comparison_validation_accuracy.png")
    print(df)

# ============================================================================
# Main Comparison Function
# ============================================================================

def compare_models(log_dir="../Results/logs/"):
    files = [f for f in os.listdir(log_dir) if f.endswith(".txt") or f.endswith(".rtf")]
    models_data = [extract_metrics_from_log(os.path.join(log_dir, file)) for file in files]
    generate_summary_table(models_data)
    plot_validation_accuracies(models_data)


if __name__ == "__main__":
    compare_models()