import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
import pandas as pd

# Output directory for plots
PLOT_DIR = "/home/xrad/KI_Piktogramm/results/EfficientNetB0/step2a"
os.makedirs(PLOT_DIR, exist_ok=True)

def pretty_labels(labels):
    
    # Replace underscores with spaces for plotting only
    
    return [lbl.replace("_", " ") for lbl in labels]


# -----------------------------------------------------------------------------
# Model comparison (bar plot)
# -----------------------------------------------------------------------------
def plot_model_comparison(results, save=True):
    df = pd.DataFrame(results)
    mean_acc = df.groupby('model')['val_acc'].mean()
    
    plt.figure(figsize=(8,6))
    mean_acc.plot(kind='bar', color='skyblue')
    plt.ylabel("Average Validation Accuracy (%)")
    plt.title("Model Comparison")
    plt.ylim(0, 100)
    plt.grid(axis='y')
    if save:
        plt.savefig(os.path.join(PLOT_DIR, "model_comparison.png"))
    plt.show()

# -----------------------------------------------------------------------------
# Training and validation process
# -----------------------------------------------------------------------------
def plot_training_curves(history, model_name, save=True):
    epochs = np.arange(1, len(history['train_acc'])+1)
    
    plt.figure(figsize=(12,5))
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_acc'], label="Train Acc")
    plt.plot(epochs, history['val_acc'], label="Val Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_name} Accuracy")
    plt.legend()
    plt.grid(True)
    
    # Loss
    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_loss'], label="Train Loss")
    plt.plot(epochs, history['val_loss'], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(PLOT_DIR, f"training_curves_{model_name}.png"))
    plt.show()

# -----------------------------------------------------------------------------
# Multi-class ROC for all models
# -----------------------------------------------------------------------------
def plot_roc_curves(y_true_dict, y_pred_proba_dict, class_names, save=True):
    n_classes = len(class_names)

    plt.figure(figsize=(10, 8))
    pretty = pretty_labels(class_names)




    for model_name, y_pred_proba in y_pred_proba_dict.items():
        y_true = y_true_dict[model_name]
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, 
                     label=f"{model_name} ({pretty[i]}) AUC={roc_auc:.2f}")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")

    if save:
        save_path = os.path.join(PLOT_DIR, f"ROC_curves_{model_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()

# -----------------------------------------------------------------------------
# Confusion matrices (classic + normalized)
# -----------------------------------------------------------------------------
def plot_confusion_matrices(y_true_dict, y_pred_dict, class_names, save=True):
    pretty = pretty_labels(class_names)

    

    for model_name, y_pred in y_pred_dict.items():
        y_true = y_true_dict[model_name]

        cm = confusion_matrix(y_true, y_pred)


        acc = (y_true == y_pred).mean()

        # =====================================================
        # 1. CLASSIC CONFUSION MATRIX
        # =====================================================
        fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=pretty
        )

        disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)

  
        ax.set_title(f"{model_name} - Confusion Matrix\nAccuracy: {acc:.3f}", fontsize=14)

       
        plt.xticks(rotation=30, ha="right", fontsize=11)
        plt.yticks(fontsize=11)

   
        if save:
            save_path = os.path.join(PLOT_DIR, f"confusion_matrix_{model_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.close() 

        # =====================================================
        # 2. Normalized
        # =====================================================
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=pretty,
            yticklabels=pretty,
            annot_kws={"size": 10},
            ax=ax
        )

  
        ax.set_title(f"{model_name} - Normalized Confusion Matrix\nAccuracy: {acc:.3f}", fontsize=14)

        ax.set_ylabel("True Label", fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)


        ax.set_xticklabels(pretty, rotation=30, ha="right")
        ax.set_yticklabels(pretty, rotation=0)

   
        if save:
            save_path = os.path.join(PLOT_DIR, f"confusion_matrix_heatmap_{model_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.close()
        
        
# -----------------------------------------------------------------------------
# Confusion matrices (classic + normalized)
# -----------------------------------------------------------------------------
        
def plot_mean_training_curves(all_fold_history, model_name, save=True):
    

    train_losses = all_fold_history[model_name]["train_loss"]
    val_losses = all_fold_history[model_name]["val_loss"]
    val_accs = all_fold_history[model_name]["val_acc"]
    train_accs = all_fold_history[model_name]["train_acc"]

    # unterschiedliche Längen durch EarlyStopping → kürzen
    min_len = min(len(l) for l in train_losses)

    train_losses = np.array([l[:min_len] for l in train_losses])
    val_losses = np.array([l[:min_len] for l in val_losses])
    val_accs = np.array([l[:min_len] for l in val_accs])
    train_accs = np.array([l[:min_len] for l in train_accs])



    mean_train_loss = train_losses.mean(axis=0)
    mean_val_loss = val_losses.mean(axis=0)
    mean_val_acc = val_accs.mean(axis=0)
    mean_train_acc = train_accs.mean(axis=0)

    epochs = range(1, min_len + 1)

    plt.figure(figsize=(10,5))

    # Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, mean_train_loss, label="Mean Train Loss")
    plt.plot(epochs, mean_val_loss, label="Mean Val Loss")
    plt.title(f"{model_name} - Mean Loss (All Folds)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, mean_train_acc, label="Mean Train Accuracy")
    plt.plot(epochs, mean_val_acc, label="Mean Val Accuracy")
    plt.title(f"{model_name} - Mean Accuracy (All Folds)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()

    if save:
        save_path = os.path.join(PLOT_DIR, f"mean_training_curves_{model_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()

    print(f"Saved mean training curves to {save_path}")