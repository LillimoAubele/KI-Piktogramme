import os
import shutil
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from models import build_efficientnet_b0

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import numpy as np


# =====================================================
# SETTINGS
# =====================================================

data_dir = "/home/xrad/KI_Piktogramm/data/Tuebingen/Training/data_sorted_by_classes"

model_path = "/home/xrad/KI_Piktogramm/trained_models_two_step_classification/EfficientNetB0/step2a/EfficientNetB0.pth"

save_dir = "/home/xrad/KI_Piktogramm/results/EfficientNetB0/step2a/misclassification_analysis"

os.makedirs(save_dir, exist_ok=True)

threshold = 0.543

batch_size = 16

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =====================================================
# CLASS NAMES
# =====================================================

CLASS_NAMES = {
    0: "HoherFlankenschnitt",
    1: "Intercostal"
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

# Nur Klassen 0 und 1
subset_indices = [
    i
    for i, (_, label) in enumerate(full_dataset.samples)
    if label in [0, 1]
]

dataset = Subset(
    full_dataset,
    subset_indices
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False
)

print(f"Images in Step2a subset: {len(dataset)}")


# =====================================================
# MODEL
# =====================================================

model = build_efficientnet_b0(
    num_classes=2,
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

for folder in [
    "true_HoherFlankenschnitt_pred_Intercostal",
    "true_Intercostal_pred_HoherFlankenschnitt"
]:
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

with torch.no_grad():

    for images, labels, paths in loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        probs_class1 = probs[:, 1]

        preds = (
            probs_class1 > threshold
        ).long()

        for i in range(len(labels)):

            true_label = labels[i].item()
            pred_label = preds[i].item()

            confidence = max(
                probs[i, 0].item(),
                probs[i, 1].item()
            )

            n_total += 1

            if pred_label == true_label:
                n_correct += 1

            sample_idx = dataset.indices[
                i + (
                    (n_total - 1)
                    // batch_size
                ) * 0
            ]

            filepath = full_dataset.samples[
                subset_indices[
                    (n_total - 1)
                ]
            ][0]

            results.append({
                "filename": os.path.basename(filepath),
                "filepath": filepath,
                "true_label": CLASS_NAMES[true_label],
                "pred_label": CLASS_NAMES[pred_label],
                "confidence": confidence
            })

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
    "step2a_predictions.csv"
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
        "step2a_misclassifications_ranked.csv"
    ),
    index=False
)


# =====================================================
# SUMMARY
# =====================================================

accuracy = n_correct / n_total

hoher_to_inter = len(
    os.listdir(
        os.path.join(
            misclassified_root,
            "true_HoherFlankenschnitt_pred_Intercostal"
        )
    )
)

inter_to_hoher = len(
    os.listdir(
        os.path.join(
            misclassified_root,
            "true_Intercostal_pred_HoherFlankenschnitt"
        )
    )
)

summary_df = pd.DataFrame({
    "metric": [
        "accuracy",
        "total_images",
        "HoherFlankenschnitt_to_Intercostal",
        "Intercostal_to_HoherFlankenschnitt"
    ],
    "value": [
        accuracy,
        n_total,
        hoher_to_inter,
        inter_to_hoher
    ]
})

summary_csv = os.path.join(
    save_dir,
    "step2a_misclassification_summary.csv"
)

summary_df.to_csv(
    summary_csv,
    index=False
)

print("\n==============================")
print("STEP 2A ANALYSIS")
print("==============================")
print(f"Accuracy: {accuracy:.4f}")
print(f"Total images: {n_total}")

print(
    f"\nHoher Flankenschnitt -> Intercostal: "
    f"{hoher_to_inter}"
)

print(
    f"Intercostal -> Hoher Flankenschnitt: "
    f"{inter_to_hoher}"
)

print(
    f"\nSummary saved to:\n{summary_csv}"
)

print(
    f"\nMisclassified images saved to:\n"
    f"{misclassified_root}"
)
