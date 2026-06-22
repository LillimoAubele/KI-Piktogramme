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

model_path = "/home/xrad/KI_Piktogramm/trained_models_two_step_classification/EfficientNetB0/step1/EfficientNetB0.pth"

save_dir = "/home/xrad/KI_Piktogramm/results/EfficientNetB0/step1/misclassification_analysis"

os.makedirs(save_dir, exist_ok=True)

threshold = 0.43
batch_size = 16

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =====================================================
# CLASS NAMES
# =====================================================

CLASS_NAMES = {
    0: "Flanke",
    1: "Oberbauch"
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

dataset = ImageFolderWithPaths(
    root=data_dir,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False
)

print(f"Images: {len(dataset)}")


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
    "true_Flanke_pred_Oberbauch",
    "true_Oberbauch_pred_Flanke"
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

        probs_oberbauch = probs[:, 1]

        preds = (
            probs_oberbauch > threshold
        ).long()

        for i in range(len(labels)):

            original_label = labels[i].item()

            # ---------------------------------
            # Map original 5 classes -> Step1
            # ---------------------------------

            if original_label in [0, 1]:
                true_step1 = 0
            else:
                true_step1 = 1

            pred_step1 = preds[i].item()

            confidence = max(
                probs[i, 0].item(),
                probs[i, 1].item()
            )

            n_total += 1

            if pred_step1 == true_step1:
                n_correct += 1

            # ---------------------------------
            # Store result
            # ---------------------------------

            results.append({
                "filename": os.path.basename(paths[i]),
                "filepath": paths[i],
                "true_label": CLASS_NAMES[true_step1],
                "pred_label": CLASS_NAMES[pred_step1],
                "confidence": confidence
            })

            # ---------------------------------
            # Misclassified?
            # ---------------------------------

            if pred_step1 != true_step1:

                folder_name = (
                    f"true_{CLASS_NAMES[true_step1]}"
                    f"_pred_{CLASS_NAMES[pred_step1]}"
                )

                target_dir = os.path.join(
                    misclassified_root,
                    folder_name
                )

                shutil.copy(
                    paths[i],
                    os.path.join(
                        target_dir,
                        os.path.basename(paths[i])
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
                    paths[i],
                    cam_dir,
                    model,
                    cam,
                    transform,
                    device
                )


# =====================================================
# SAVE CSV
# =====================================================

results_df = pd.DataFrame(results)

csv_path = os.path.join(
    save_dir,
    "step1_predictions.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)

print(f"\nPredictions saved to:\n{csv_path}")


# =====================================================
# SUMMARY
# =====================================================

accuracy = n_correct / n_total

print("\n==============================")
print("STEP 1 ANALYSIS")
print("==============================")
print(f"Accuracy: {accuracy:.4f}")
print(f"Total images: {n_total}")

flanke_to_oberbauch = len(
    os.listdir(
        os.path.join(
            misclassified_root,
            "true_Flanke_pred_Oberbauch"
        )
    )
)

oberbauch_to_flanke = len(
    os.listdir(
        os.path.join(
            misclassified_root,
            "true_Oberbauch_pred_Flanke"
        )
    )
)

print(
    f"\nFlanke -> Oberbauch: "
    f"{flanke_to_oberbauch}"
)

print(
    f"Oberbauch -> Flanke: "
    f"{oberbauch_to_flanke}"
)


# =====================================================
# SAVE SUMMARY
# =====================================================

summary_df = pd.DataFrame({
    "metric": [
        "accuracy",
        "total_images",
        "Flanke_to_Oberbauch",
        "Oberbauch_to_Flanke"
    ],
    "value": [
        accuracy,
        n_total,
        flanke_to_oberbauch,
        oberbauch_to_flanke
    ]
})

summary_csv = os.path.join(
    save_dir,
    "step1_misclassification_summary.csv"
)

summary_df.to_csv(
    summary_csv,
    index=False
)

print(
    f"\nSummary saved to:\n{summary_csv}"
)

print(
    f"\nMisclassified images saved to:\n"
    f"{misclassified_root}"
)

