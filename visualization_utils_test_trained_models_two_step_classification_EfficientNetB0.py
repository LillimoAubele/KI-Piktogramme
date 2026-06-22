import os
import numpy as np
import torch

from PIL import Image

import matplotlib.pyplot as plt


# =====================================================
# GRADCAM
# =====================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(
            self.save_activation
        )

        target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(
        self,
        module,
        inp,
        out
    ):

        self.activations = out.detach()

    def save_gradient(
        self,
        module,
        grad_in,
        grad_out
    ):

        self.gradients = grad_out[0].detach()

    def __call__(
        self,
        tensor,
        class_idx
    ):

        self.model.zero_grad()

        out = self.model(tensor)

        out[:, class_idx].backward()

        grads = self.gradients
        acts = self.activations

        weights = grads.mean(
            dim=(2,3),
            keepdim=True
        )

        cam = (
            weights * acts
        ).sum(dim=1).squeeze()

        cam = torch.relu(cam)

        cam = cam / (
            cam.max() + 1e-8
        )

        return cam.cpu().numpy()


# =====================================================
# TARGET LAYER
# =====================================================

def get_target_layer(model):

    return model.features[-1]


# =====================================================
# PICTOGRAMS
# =====================================================

PICTOGRAM_MAP = {

    0:
    "Hoher_Flankenschnitt_rechts.png",

    1:
    "Intercostalschnitt_rechts.png",

    2:
    "Oberbauchlaengsschnitt_MCL_rechts.png",

    3:
    "Oberbauchlaengsschnitt_links.png",

    4:
    "Subcostaler_Schraegschnitt_rechts.png"
}


# =====================================================
# OVERLAY PICTOGRAM
# =====================================================

def overlay_pictogram(
    image,
    class_idx,
    pictogram_dir
):

    icon_name = PICTOGRAM_MAP.get(
        class_idx
    )

    if icon_name is None:
        return image

    icon_path = os.path.join(
        pictogram_dir,
        icon_name
    )

    if not os.path.exists(
        icon_path
    ):
        return image

    icon = Image.open(
        icon_path
    ).convert("RGBA")

    scale = 0.15

    width = int(
        image.width * scale
    )

    height = int(
        icon.height *
        (
            width /
            icon.width
        )
    )

    icon = icon.resize(
        (width,height)
    )

    image = image.copy().convert(
        "RGBA"
    )

    image.paste(
        icon,
        (
            20,
            image.height - height - 20
        ),
        icon
    )

    return image.convert("RGB")


# =====================================================
# CREATE GRADCAM OVERLAY
# =====================================================

def create_gradcam_overlay(image, cam):

    image_np = (
        np.array(image)
        .astype(np.float32)
        / 255.0
    )

    cam = Image.fromarray(
        np.uint8(cam * 255)
    )

    cam = cam.resize(
        (image_np.shape[1],
         image_np.shape[0])
    )

    cam = (
        np.array(cam)
        .astype(np.float32)
        / 255.0
    )

    heatmap = plt.get_cmap("jet")(
        cam
    )[:, :, :3]

    overlay = (
        0.6 * image_np +
        0.4 * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    return overlay


# =====================================================
# SAVE TRIPLET
# =====================================================

def save_triplet(

    image,

    pred_class,

    cam,

    pictogram_dir,

    save_path
):

    pictogram_img = overlay_pictogram(

        image,
        pred_class,
        pictogram_dir

    )

    gradcam_overlay = create_gradcam_overlay(

        image,
        cam

    )

    fig, axes = plt.subplots(

        1,
        3,

        figsize=(12,4)

    )

    axes[0].imshow(image)

    axes[0].set_title(
        "Original"
    )

    axes[0].axis("off")

    axes[1].imshow(
        pictogram_img
    )

    axes[1].set_title(
        "Prediction"
    )

    axes[1].axis("off")

    axes[2].imshow(
        gradcam_overlay
    )

    axes[2].set_title(
        "GradCAM"
    )

    axes[2].axis("off")

    plt.tight_layout()

    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()


# =====================================================
# SAVE MISCLASSIFIED TRIPLET
# =====================================================

def save_misclassified_triplet(

    image,

    true_class,
    pred_class,

    cam,

    pictogram_dir,

    save_path
):

    true_img = overlay_pictogram(

        image,
        true_class,
        pictogram_dir

    )

    pred_img = overlay_pictogram(

        image,
        pred_class,
        pictogram_dir

    )

    gradcam_overlay = create_gradcam_overlay(

        image,
        cam

    )

    fig, axes = plt.subplots(

        1,
        4,

        figsize=(16,4)

    )

    axes[0].imshow(image)

    axes[0].set_title(
        "Original"
    )

    axes[0].axis("off")

    axes[1].imshow(true_img)

    axes[1].set_title(
        "True"
    )

    axes[1].axis("off")

    axes[2].imshow(pred_img)

    axes[2].set_title(
        "Prediction"
    )

    axes[2].axis("off")

    axes[3].imshow(
        gradcam_overlay
    )

    axes[3].set_title(
        "GradCAM"
    )

    axes[3].axis("off")

    plt.tight_layout()

    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()
