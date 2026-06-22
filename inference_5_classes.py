import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from models import build_efficientnet_b0


# =====================================================
# Define parameters
# =====================================================

data_dir = "/home/xrad/KI_Piktogramm/data/Tuebingen/Training/data_sorted_by_classes"

save_dir = "/home/xrad/KI_Piktogramm/results/EfficientNetB0/5_classes"
os.makedirs(save_dir, exist_ok=True)

step1_model_path = "/home/xrad/KI_Piktogramm/trained_models_two_step_classification/EfficientNetB0/step1/EfficientNetB0.pth"

step2a_model_path = "/home/xrad/KI_Piktogramm/trained_models_two_step_classification/EfficientNetB0/step2a/EfficientNetB0.pth"

step2b_model_path = "/home/xrad/KI_Piktogramm/trained_models_two_step_classification/EfficientNetB0/step2b/EfficientNetB0.pth"

batch_size = 16

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# =====================================================
# FINAL CLASSES
# =====================================================

FINAL_CLASSES = [
    "Hoher Flankenschnitt rechts",
    "Intercostalschnitt rechts",
    "MCL rechts",
    "Oberbauch links",
    "Subcostal rechts"
]

# =====================================================
# TRANSFORMS
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# DATASET
# =====================================================

dataset = datasets.ImageFolder(
    root=data_dir,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False
)

# =====================================================
# LOAD STEP1 MODEL
# =====================================================

step1_model = build_efficientnet_b0(
    num_classes=2,
    pretrained=False
)

step1_model.load_state_dict(
    torch.load(
        step1_model_path,
        map_location=device
    )
)

step1_model.to(device)
step1_model.eval()

# =====================================================
# LOAD STEP2A MODEL
# =====================================================

step2a_model = build_efficientnet_b0(
    num_classes=2,
    pretrained=False
)

step2a_model.load_state_dict(
    torch.load(
        step2a_model_path,
        map_location=device
    )
)

step2a_model.to(device)
step2a_model.eval()

# =====================================================
# LOAD STEP2B MODEL
# =====================================================

step2b_model = build_efficientnet_b0(
    num_classes=3,
    pretrained=False
)

step2b_model.load_state_dict(
    torch.load(
        step2b_model_path,
        map_location=device
    )
)

step2b_model.to(device)
step2b_model.eval()

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_pipeline(images):

    with torch.no_grad():

        outputs1 = step1_model(images)

        probs1 = torch.softmax(
            outputs1,
            dim=1
        )
        
        threshold_step1 = 0.43
        
        pred_step1 = (
            probs1[:, 1] > threshold_step1
        ).long()

        final_preds = []

        for i in range(len(pred_step1)):

            single_img = images[i].unsqueeze(0)

            # -------------------
            # FLANKE
            # -------------------

            if pred_step1[i].item() == 0:

                outputs2a = step2a_model(single_img)

                pred2a = torch.argmax(
                    outputs2a,
                    dim=1
                ).item()

                mapping = {
                    0: 0,   # Hoher Flankenschnitt
                    1: 1    # Intercostal
                }

                final_preds.append(
                    mapping[pred2a]
                )

            # -------------------
            # OBERBAUCH
            # -------------------

            else:

                outputs2b = step2b_model(single_img)

                pred2b = torch.argmax(
                    outputs2b,
                    dim=1
                ).item()

                mapping = {
                    0: 2,   # MCL
                    1: 3,   # links
                    2: 4    # subcostal
                }

                final_preds.append(
                    mapping[pred2b]
                )

        return np.array(final_preds)

# =====================================================
# EVALUATION
# =====================================================

y_true = []
y_pred = []

for images, labels in loader:

    images = images.to(device)

    preds = predict_pipeline(images)

    y_pred.extend(preds)

    y_true.extend(
        labels.numpy()
    )

# =====================================================
# METRICS
# =====================================================

acc = accuracy_score(
    y_true,
    y_pred
)

metrics_df = pd.DataFrame({
    "accuracy": [acc]
})

metrics_csv = os.path.join(
    save_dir,
    "5_classes_metrics.csv"
)

metrics_df.to_csv(
    metrics_csv,
    index=False
)

print(f"Metrics saved to {metrics_csv}")

# ---------------------------------------------------------
# Save predictions
# ---------------------------------------------------------

pred_df = pd.DataFrame({
    "y_true": y_true,
    "y_pred": y_pred
})

pred_csv = os.path.join(
    save_dir,
    "5_classes_predictions.csv"
)

pred_df.to_csv(
    pred_csv,
    index=False
)

print(f"Predictions saved to {pred_csv}")
print(f"Accuracy: {acc:.4f}")

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=FINAL_CLASSES,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_csv = os.path.join(
    save_dir,
    "5_classes_classification_report.csv"
)

report_df.to_csv(report_csv)

print(f"Classification report saved to {report_csv}")

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm = np.nan_to_num(cm)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=FINAL_CLASSES,
    yticklabels=FINAL_CLASSES
)

plt.title(
    f"EfficientNetB0 5 Classes Normalized Confusion Matrix\nAccuracy = {acc:.3f}"
)

plt.xlabel("Prediction")
plt.ylabel("True")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(
    f"{save_dir}/5_classes_confusion_matrix_normalized.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
