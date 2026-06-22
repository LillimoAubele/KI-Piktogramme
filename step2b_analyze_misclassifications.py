import os
import shutil
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np

from models import build_efficientnet_b0


# =====================================================
# SETTINGS
# =====================================================

data_dir = "/home/xrad/KI_Piktogramm/data/Tuebingen/Training/data_sorted_by_classes"

model_path = "/home/xrad/KI_Piktogramm/trained_models_two_step_classification/EfficientNetB0/step2b/EfficientNetB0.pth"

save_dir = "/home/xrad/KI_Piktogramm/results/EfficientNetB0/step2b/misclassification_analysis"

os.makedirs(save_dir, exist_ok=True)

batch_size = 16

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =====================================================
# CLASS NAMES
# =====================================================

CLASS_NAMES = {
    0: "MCL",
    1: "Links",
    2: "Subcostal"
}


# =====================================================
# DATASET WITH PATHS
# =====================================================

class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):

        image, label = super().__getitem__(index)

        path = self.samples[index][0]

        return image, label, path


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

full_dataset = ImageFolderWithPaths(
    root=data_dir,
    transform=transform
)

# Nur Oberbauch-Klassen
subset_indices = [
    i
    for i, (_, label) in enumerate(full_dataset.samples)
    if label in [2, 3, 4]
]

samples = [
    full_dataset.samples[i]
    for i in subset_indices
]

print(f"Images in Step2b subset: {len(samples)}")


# =====================================================
# MODEL
# =====================================================

model = build_efficientnet_b0(
    num_classes=3,
    pretrained=False
)

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.to(device)
model.eval()


target_layers = [model.features[-1]]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

# =====================================================
# OUTPUT DIRECTORIES
# =====================================================

misclassified_root = os.path.join(
    save_dir,
    "misclassified"
)

gradcam_root = os.path.join(
    save_dir,
    "gradcam"
)

os.makedirs(
    gradcam_root,
    exist_ok=True
)

os.makedirs(
    misclassified_root,
    exist_ok=True
)

folders = [

    "true_MCL_pred_Links",
    "true_MCL_pred_Subcostal",

    "true_Links_pred_MCL",
    "true_Links_pred_Subcostal",

    "true_Subcostal_pred_MCL",
    "true_Subcostal_pred_Links"
]

for folder in folders:

    os.makedirs(
        os.path.join(
            misclassified_root,
            folder
        ),
        exist_ok=True
    )


# =====================================================
# GRADCAM
# =====================================================

def save_gradcam(
    image_path,
    target_dir,
    model,
    cam,
    transform,
    device
):

    pil_img = Image.open(
        image_path
    ).convert("RGB")

    rgb_img = np.array(
        pil_img.resize((224,224))
    ) / 255.0

    input_tensor = transform(
        pil_img
    ).unsqueeze(0).to(device)

    input_tensor.requires_grad_(True)

    with torch.enable_grad():

        grayscale_cam = cam(
            input_tensor=input_tensor
        )[0]

    overlay = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    # Heatmap separat erzeugen
    heatmap = plt.get_cmap("jet")(
        grayscale_cam
    )[:, :, :3]

    # -----------------------------
    # 3er-Darstellung
    # -----------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12,4)
    )

    axes[0].imshow(rgb_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(heatmap)
    axes[2].set_title("Heatmap")
    axes[2].axis("off")

    plt.tight_layout()

    save_path = os.path.join(
        target_dir,
        os.path.splitext(
            os.path.basename(image_path)
        )[0]
        + "_gradcam_triplet.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# EVALUATION
# =====================================================

results = []

n_correct = 0
n_total = 0

mapping = {
    2: 0,   # MCL
    3: 1,   # Links
    4: 2    # Subcostal
}



for filepath, original_label in samples:

    image, _, _ = full_dataset[
        full_dataset.samples.index(
            (filepath, original_label)
        )
    ]

    image = image.unsqueeze(0).to(device)

    outputs = model(image)

    probs = torch.softmax(
        outputs,
        dim=1
    )

    pred_label = torch.argmax(
        probs,
        dim=1
    ).item()

    true_label = mapping[
        original_label
    ]

    confidence = probs[
        0,
        pred_label
    ].item()

    n_total += 1

    if pred_label == true_label:
        n_correct += 1

    results.append({
        "filename": os.path.basename(filepath),
        "filepath": filepath,
        "true_label": CLASS_NAMES[true_label],
        "pred_label": CLASS_NAMES[pred_label],
        "confidence": confidence
    })

    # ---------------------------------
    # Copy misclassified images
    # ---------------------------------

    if pred_label != true_label:

        folder_name = (
            f"true_{CLASS_NAMES[true_label]}"
            f"_pred_{CLASS_NAMES[pred_label]}"
        )

        target_dir = os.path.join(
            misclassified_root,
            folder_name
        )

        shutil.copy(
            filepath,
            os.path.join(
                target_dir,
                os.path.basename(filepath)
            )
        )

        cam_dir = os.path.join(
            gradcam_root,
            folder_name
        )
        
        os.makedirs(
            cam_dir,
            exist_ok=True
        )
        
        save_gradcam(
            filepath,
            cam_dir,
            model,
            cam,
            transform,
            device
        )

# =====================================================
# SAVE PREDICTIONS
# =====================================================

results_df = pd.DataFrame(results)

csv_path = os.path.join(
    save_dir,
    "step2b_predictions.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)

print(f"\nPredictions saved to:\n{csv_path}")


# =====================================================
# SAVE MISCLASSIFICATIONS ONLY
# =====================================================

mis_df = results_df[
    results_df["true_label"] != results_df["pred_label"]
]

mis_df = mis_df.sort_values(
    "confidence",
    ascending=False
)

mis_df.to_csv(
    os.path.join(
        save_dir,
        "step2b_misclassifications_ranked.csv"
    ),
    index=False
)


# =====================================================
# SUMMARY
# =====================================================

accuracy = n_correct / n_total

summary_counts = {}

for folder in folders:

    summary_counts[folder] = len(
        os.listdir(
            os.path.join(
                misclassified_root,
                folder
            )
        )
    )

summary_df = pd.DataFrame({
    "metric": [
        "accuracy",
        "total_images"
    ] + list(summary_counts.keys()),
    "value": [
        accuracy,
        n_total
    ] + list(summary_counts.values())
})

summary_csv = os.path.join(
    save_dir,
    "step2b_misclassification_summary.csv"
)

summary_df.to_csv(
    summary_csv,
    index=False
)

print("\n==============================")
print("STEP 2B ANALYSIS")
print("==============================")
print(f"Accuracy: {accuracy:.4f}")
print(f"Total images: {n_total}")

for k, v in summary_counts.items():
    print(f"{k}: {v}")

print(
    f"\nSummary saved to:\n{summary_csv}"
)

print(
    f"\nMisclassified images saved to:\n"
    f"{misclassified_root}"
)