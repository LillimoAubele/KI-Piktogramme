import os
import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import shutil
from collections import Counter
from sklearn.metrics import roc_curve
import pickle





from preprocess_data_two_step_classification_step1_EfficientNetB0 import get_data_folds
from models import build_efficientnet_b0
from validation_two_step_classification_step1_EfficientNetB0 import evaluate_model, FocalLoss# , get_loss_function
from early_stopping import EarlyStopping
from visualize_results_two_step_classification_step1_EfficientNetB0 import (
    plot_roc_curves, 
    plot_confusion_matrices,
    plot_mean_training_curves
)

# -----------------------------------------------------------------------------
# Define parameters
# -----------------------------------------------------------------------------
data_dir = "/home/xrad/KI_Piktogramm/data/Tuebingen/Training/data_sorted_by_classes"

use_pretrained = True
fine_tune = True
num_classes = 2

num_folds = 5
batch_size = 15      
num_epochs = 40
learning_rate_head = 5e-4
learning_rate_backbone = 1e-4
weight_decay = 1e-3
label_smoothing = 0.1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


CLASS_NAMES = [
    "Flanke",
    "Oberbauch"
]

# -----------------------------------------------------------------------------
# Define models
# -----------------------------------------------------------------------------
models_to_train = {
    "EfficientNetB0": lambda num_classes: build_efficientnet_b0(num_classes, pretrained=use_pretrained, fine_tune=fine_tune),
}

# -----------------------------------------------------------------------------
# Save experiment config
# -----------------------------------------------------------------------------
config = {
    "data_dir": data_dir,
    "use_pretrained": use_pretrained,
    "fine_tune": fine_tune,
    "num_classes": num_classes,
    "num_folds": num_folds,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "learning_rate_head": learning_rate_head,
    "learning_rate_backbone": learning_rate_backbone,
    "device": str(device),
    "models": list(models_to_train.keys())
}

config_save_path = "/home/xrad/KI_Piktogramm/trained_models_two_step_classification/EfficientNetB0/step1/config.json"

# Create directory if none exists
os.makedirs(os.path.dirname(config_save_path), exist_ok=True)

with open(config_save_path, "w") as f:
    json.dump(config, f, indent=4)

print(f"Configuration saved to {config_save_path}")


# Output directory for plots
PLOT_DIR = "/home/xrad/KI_Piktogramm/results/EfficientNetB0/step1"
os.makedirs(PLOT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Load cross-validation folds
# -----------------------------------------------------------------------------
folds, dataset = get_data_folds(data_dir, batch_size, num_folds)



# -----------------------------------------------------------------------------
# Containers
# -----------------------------------------------------------------------------
history = {name: {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
           for name in models_to_train.keys()}

y_true_dict = {name: [] for name in models_to_train.keys()}
y_pred_dict = {name: [] for name in models_to_train.keys()}
y_pred_proba_dict = {name: [] for name in models_to_train.keys()}

all_fold_history = {
    name: {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    for name in models_to_train.keys()
}

results = []
fold_thresholds = {name: [] for name in models_to_train.keys()}


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
for model_name, build_model in models_to_train.items():
    
    print(f"\n==== Training {model_name} ====")
    
    for fold, (train_loader, val_loader, num_classes) in enumerate(folds, 1):
        print(f"\n--- Fold {fold}/{num_folds} ---")
        
        #------------------------------------------------------------------------------
        # Calculate class weights
        #------------------------------------------------------------------------------


        all_labels = []

        for _, labels in train_loader:
            all_labels.extend(labels.cpu().numpy())

        class_counts = Counter(all_labels)

        print(f"Fold {fold} class distribution:", class_counts)

        counts = np.array([class_counts[i] for i in range(num_classes)])

        weights = 1.0 / counts
        weights = weights / weights.sum() * num_classes

        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

        print("Class weights:", class_weights)
        
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        # Build model
        model = build_model(num_classes).to(device)
        
        # Freeze entire backbone first
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        
    
        # Optimizer
        optimizer = torch.optim.AdamW(
            [
                {"params": model.classifier.parameters(), "lr": learning_rate_head},
                {"params": [p for n, p in model.named_parameters() if "classifier" not in n], "lr": learning_rate_backbone},
            ],
            weight_decay=weight_decay
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
                

        # Loss + EarlyStopping
        criterion = FocalLoss(gamma=2.0, alpha=class_weights)

        early_stopping = EarlyStopping(
            patience=7,
            min_delta=0.002,
            min_epochs=20
        )
        
        
        # AMP
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

        
        # Track best model for this fold
        best_val_loss = float("inf")
        best_model_path = f"/home/xrad/KI_Piktogramm/trained_models_two_step_classification/EfficientNetB0/step1/{model_name}_fold{fold}_BEST.pth"


        for epoch in range(num_epochs):
        
         
            # Phase 2: unfreeze layer4
            if epoch == 3:
                print("Phase 2: Unfreezing last feature block")
            
                for param in model.features[-2].parameters():
                    param.requires_grad = True
            
                optimizer = torch.optim.AdamW(
                    [
                        {"params": model.classifier.parameters(), "lr": learning_rate_head},
                        {"params": model.features[-1].parameters(), "lr": 5e-5},
                    ],
                    weight_decay=weight_decay
                )
                        
            # Phase 3: unfreeze entire network
            if epoch == 7:
                for param in model.parameters():
                    param.requires_grad = True
                print("Phase 3: Unfreezing full backbone")
        
            # -------------------------------
            # Training
            # -------------------------------
            model.train()
            running_loss, correct, total = 0.0, 0, 0
        
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
        
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
        
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
                running_loss += loss.item() * images.size(0)
                
                probs1 = torch.softmax(outputs, dim=1)
                probs_class1 = probs1[:, 1]
                
                threshold = 0.43
                predicted = (probs_class1 > threshold).long()
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
            train_loss = running_loss / total
            train_acc = correct / total
        
            # -------------------------------
            # Validation
            # -------------------------------
            val_loss, val_acc, top2_acc, val_labels, val_preds, val_probs = evaluate_model(
                model, val_loader, criterion, device, return_preds=True)
        
            # -------------------------------
            # Logging
            # -------------------------------
            history[model_name]["train_loss"].append(train_loss)
            history[model_name]["train_acc"].append(train_acc)
            history[model_name]["val_loss"].append(val_loss)
            history[model_name]["val_acc"].append(val_acc)
            history[model_name].setdefault("top2_acc", []).append(top2_acc)
            
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
        
            y_true_dict[model_name].extend(val_labels)
            y_pred_dict[model_name].extend(val_preds)
            y_pred_proba_dict[model_name].extend(val_probs)
        
            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                f"Top-2 Acc: {top2_acc:.2f}"
            )
        
            # Best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model updated! Saved to {best_model_path}")
        

            # Scheduler + EarlyStopping
            scheduler.step()
            early_stopping(val_loss, epoch+1)
        
            if early_stopping.early_stop:
                print("Early stopping activated!")
                break


        # Store fold result
        results.append({
            "model": model_name,
            "fold": fold,
            "val_loss": val_loss,
            "val_acc": val_acc, 
            "train_loss": train_loss,
            "train_acc": train_acc, 
        })
        
        y_true_fold = np.array(val_labels)
        y_probs_fold = np.array(val_probs)[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_true_fold, y_probs_fold)
        
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        
        fold_thresholds[model_name].append(best_threshold)
        
        print(f"Fold {fold} best threshold: {best_threshold:.3f}")
        
        epochs = range(1, len(train_losses) + 1)
        
        # plt.figure(figsize=(10,5))
        
        # # Loss
        # plt.subplot(1,2,1)
        # plt.plot(epochs, train_losses, label="Train Loss")
        # plt.plot(epochs, val_losses, label="Val Loss")
        # plt.title(f"{model_name} - Fold {fold} Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.legend()
        
        # # Accuracy
        # plt.subplot(1,2,2)
        # plt.plot(epochs, val_accuracies, label="Val Accuracy")
        # plt.title(f"{model_name} - Fold {fold} Accuracy")
        # plt.xlabel("Epoch")
        # plt.ylabel("Accuracy")
        # plt.legend()
        
        # plt.tight_layout()
        # plt.savefig(f"/home/xrad/KI_Piktogramm/plots/ResNet50/{model_name}_fold{fold}.png")
        # plt.close()
            
        
        all_fold_history[model_name]["train_loss"].append(train_losses)
        all_fold_history[model_name]["train_acc"].append(train_accuracies)
        all_fold_history[model_name]["val_loss"].append(val_losses)
        all_fold_history[model_name]["val_acc"].append(val_accuracies)
    
       
        # Select best fold model as final model
    best_fold_file = None
    best_fold_loss = float("inf")
    
    for r in results:
        if r["model"] == model_name and r["val_loss"] < best_fold_loss:
            best_fold_loss = r["val_loss"]
            best_fold_file = f"/home/xrad/KI_Piktogramm/trained_models_two_step_classification/EfficientNetB0/step1/{model_name}_fold{r['fold']}_BEST.pth"
    
    # Save final best model
    final_model_save_path = f"/home/xrad/KI_Piktogramm/trained_models_two_step_classification/EfficientNetB0/step1/{model_name}.pth"

    shutil.copy(best_fold_file, final_model_save_path)

    
    print(f"\n Final {model_name} saved (best fold model) to '{final_model_save_path}'")


# Convert prediction lists to numpy arrays
for model_name in models_to_train.keys():
    y_true_dict[model_name] = np.array(y_true_dict[model_name])
    y_pred_dict[model_name] = np.array(y_pred_dict[model_name])
    y_pred_proba_dict[model_name] = np.array(y_pred_proba_dict[model_name])

# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------
df = pd.DataFrame(results)
csv_path = "/home/xrad/KI_Piktogramm/results/EfficientNetB0/step1/training_results_EfficientNetB0.csv"
df.to_csv(csv_path,index=False)
print(f"\nResults saved to '{csv_path}'.")

# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------

for model_name in all_fold_history.keys():

    plot_mean_training_curves(
        all_fold_history,
        model_name
        )
    

print("\n--- Mean Threshold per Model ---")

mean_thresholds = {}

for model_name in fold_thresholds:
    mean_t = np.mean(fold_thresholds[model_name])
    mean_thresholds[model_name] = mean_t

    print(f"{model_name}: {mean_t:.3f}")



#plot_roc_curves(y_true_dict, y_pred_proba_dict, CLASS_NAMES)
plot_confusion_matrices(y_true_dict, y_pred_dict, CLASS_NAMES)


with open(
    "/home/xrad/KI_Piktogramm/results/EfficientNetB0/step1/all_fold_history.pkl",
    "wb"
) as f:
    pickle.dump(all_fold_history, f)