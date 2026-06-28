import os
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score

from models import build_efficientnet_b0
from preprocess_data_5_classes import get_data_folds
from validation import evaluate_model
from visualization import create_visualizations


# =====================================================
# SETTINGS
# =====================================================

DATA_DIR = "/home/xrad/KI_Piktogramm/data/Tuebingen/Training/data_sorted_by_classes"

OUTPUT_DIR = "/home/xrad/KI_Piktogramm/results/EfficientNetB0/5_classes_one_step"


BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# =====================================================
# CLASS NAMES
# =====================================================

CLASS_NAMES = [
    "Hoher_Flankenschnitt_rechts",
    "Intercostalschnitt_rechts",
    "Oberbauchlaengsschnitt_MCL_rechts",
    "Oberbauchlaengsschnitt_links",
    "Subcostaler_Schraegschnitt_rechts"
]

DISPLAY_NAMES = [
    "Hoher Flankenschnitt",
    "Intercostal rechts",
    "MCL rechts",
    "Oberbauch links",
    "Subcostal rechts"
]

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)

# =====================================================
# FOCAL LOSS
# =====================================================

class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha=None,
        gamma=2.0
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits,
        targets
    ):

        ce_loss = F.cross_entropy(

            logits,
            targets,

            weight=self.alpha,

            reduction="none"
        )

        pt = torch.exp(
            -ce_loss
        )

        focal_loss = (
            (1 - pt) ** self.gamma
        ) * ce_loss

        return focal_loss.mean()

# =====================================================
# DATA
# =====================================================

folds, dataset = get_data_folds(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_folds=5
)

# =====================================================
# CLASS WEIGHTS
# =====================================================

targets = np.array([
    label
    for _, label
    in dataset.samples
])

num_classes = 5

class_counts = np.bincount(
    targets,
    minlength=num_classes
)

class_weights = (
    len(targets)
    /
    (
        num_classes
        * class_counts
    )
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)

print(
    "Class counts:",
    class_counts
)

print(
    "Class weights:",
    class_weights
)

# =====================================================
# TRAINING
# =====================================================

all_results = []

# ==========================================
# OVERALL RESULTS
# ==========================================

all_y_true = []
all_y_pred = []
all_y_probs = []

all_fold_accs = []
all_fold_top2 = []


for fold_idx, (
    train_loader,
    val_loader,
    num_classes
) in enumerate(
    folds,
    start=1
):

    print("\n====================")
    print(f"FOLD {fold_idx}")
    print("====================")

    model = build_efficientnet_b0(
        num_classes=num_classes,
        pretrained=True
    )

    model.to(device)

    criterion = FocalLoss(
        alpha=class_weights,
        gamma=2.0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    best_acc = 0.0

    best_weights = copy.deepcopy(
        model.state_dict()
    )

    history = []

    # =================================================
    # EPOCH LOOP
    # =================================================

    for epoch in range(NUM_EPOCHS):

        model.train()

        train_loss = 0.0

        train_true = []
        train_pred = []

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                * images.size(0)
            )

            preds = torch.argmax(
                outputs,
                dim=1
            )

            train_true.extend(
                labels.cpu().numpy()
            )

            train_pred.extend(
                preds.cpu().numpy()
            )

        train_loss /= len(
            train_loader.dataset
        )

        train_acc = accuracy_score(
            train_true,
            train_pred
        )

        # =============================================
        # VALIDATION
        # =============================================

        model.eval()

        val_loss = 0.0

        val_true = []
        val_pred = []

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss += (
                    loss.item()
                    * images.size(0)
                )

                preds = torch.argmax(
                    outputs,
                    dim=1
                )

                val_true.extend(
                    labels.cpu().numpy()
                )

                val_pred.extend(
                    preds.cpu().numpy()
                )

        val_loss /= len(
            val_loader.dataset
        )

        val_acc = accuracy_score(
            val_true,
            val_pred
        )

        scheduler.step(
            val_acc
        )

        if val_acc > best_acc:

            best_acc = val_acc

            best_weights = copy.deepcopy(
                model.state_dict()
            )

        history.append({

            "epoch": epoch + 1,

            "train_loss": train_loss,
            "val_loss": val_loss,

            "train_acc": train_acc,
            "val_acc": val_acc
        })

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Acc {val_acc:.4f}"
        )

    # =================================================
    # SAVE BEST MODEL
    # =================================================

    fold_dir = os.path.join(
        OUTPUT_DIR,
        f"fold_{fold_idx}"
    )

    os.makedirs(
        fold_dir,
        exist_ok=True
    )

    torch.save(

        best_weights,

        os.path.join(
            fold_dir,
            "EfficientNetB0_best.pth"
        )
    )

    pd.DataFrame(
        history
    ).to_csv(

        os.path.join(
            fold_dir,
            "history.csv"
        ),

        index=False
    )

    # =================================================
    # FINAL EVALUATION
    # =================================================

    model.load_state_dict(
        best_weights
    )

    val_loss, val_acc, top2_acc, y_true, y_pred, y_probs = evaluate_model(

        model,
        val_loader,
        criterion,
        device,

        return_preds=True
    )
    
    # ==========================================
    # COLLECT OVERALL RESULTS
    # ==========================================
    
    all_y_true.extend(
        y_true
    )
    
    all_y_pred.extend(
        y_pred
    )
    
    all_y_probs.extend(
        y_probs
    )
    
    all_fold_accs.append(
        val_acc
    )
    
    all_fold_top2.append(
        top2_acc
    )

    create_visualizations(

        y_true=y_true,

        y_pred=y_pred,

        y_probs=y_probs,

        class_names=DISPLAY_NAMES,

        output_dir=fold_dir,

        val_loss=val_loss,

        val_acc=val_acc,

        top2_acc=top2_acc
    )


    all_results.append({

        "fold": fold_idx,

        "best_val_acc": val_acc,

        "top2_acc": top2_acc
    })

# =====================================================
# SUMMARY
# =====================================================

summary_df = pd.DataFrame(
    all_results
)

summary_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "cross_validation_results.csv"
    ),

    index=False
)

print("\n====================")
print("FINAL RESULTS")
print("====================")

print(summary_df)

print(
    "\nMean Accuracy:",
    summary_df["best_val_acc"].mean()
)

print(
    "Std Accuracy:",
    summary_df["best_val_acc"].std()
)

print(
    "Mean Top2:",
    summary_df["top2_acc"].mean()
)


# =====================================================
# OVERALL RESULTS ACROSS ALL FOLDS
# =====================================================

overall_dir = os.path.join(
    OUTPUT_DIR,
    "overall_results"
)

os.makedirs(
    overall_dir,
    exist_ok=True
)

overall_acc = accuracy_score(
    all_y_true,
    all_y_pred
)

overall_top2 = np.mean(
    all_fold_top2
)

create_visualizations(

    y_true=all_y_true,

    y_pred=all_y_pred,

    y_probs=all_y_probs,

    class_names=DISPLAY_NAMES,

    output_dir=overall_dir,

    val_loss=np.nan,

    val_acc=overall_acc,

    top2_acc=overall_top2
)

# =====================================================
# CROSS VALIDATION SUMMARY
# =====================================================

cv_summary = pd.DataFrame({

    "metric": [

        "mean_accuracy",
        "std_accuracy",

        "mean_top2_accuracy",
        "std_top2_accuracy"
    ],

    "value": [

        np.mean(all_fold_accs),
        np.std(all_fold_accs),

        np.mean(all_fold_top2),
        np.std(all_fold_top2)
    ]
})

cv_summary.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "cross_validation_summary.csv"
    ),

    index=False
)

print("\n==========================")
print("OVERALL RESULTS")
print("==========================")

print(
    "Overall Accuracy:",
    round(overall_acc, 4)
)

print(
    "Mean Fold Accuracy:",
    round(np.mean(all_fold_accs), 4)
)

print(
    "Std Fold Accuracy:",
    round(np.std(all_fold_accs), 4)
)

print(
    "Mean Top2 Accuracy:",
    round(np.mean(all_fold_top2), 4)
)