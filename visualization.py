"""
Visualization of the test set results.

Everything is saved to results/<model_name>/:
- classification_report.txt / .csv      per-class precision, recall, F1 + summary metrics
- confusion_matrix_normalized.png/.csv  row-normalized (= per-class recall), with raw counts
- confusion_matrix_counts.csv           raw counts
- roc_curves.png                        one-vs-rest ROC curves with AUC per class
- training_curves.png / history.csv     train loss and val accuracy per epoch
- misclassified.csv                     all wrong test predictions with file name and confidence
- gradcam_correct.png                   Grad-CAM grid: correctly classified examples per class
- gradcam_misclassified.png             Grad-CAM grid: the most confident errors
- gradcam/correct/, gradcam/misclassified/   individual Grad-CAM images (original | overlay)

Runs automatically at the end of main.py, or standalone on a saved checkpoint:
    python visualization.py --model efficientnet_b0
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")  # no display needed, write files only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)

from models import build_model
from preprocess_data import MEAN, STD, get_dataloaders

# Short class names for plot axes
SHORT_NAMES = {
    "1_upper_abdomen_longitudinal_left": "1: UA long. left",
    "2_upper_abdomen_longitudinal_MCL_right": "2: UA long. MCL right",
    "3_subcostal_oblique_right": "3: Subcostal oblique",
    "4_intercostal_right": "4: Intercostal",
    "5_high_flank_right": "5: High flank",
}

DPI = 300


def short_names(class_names):
    return [SHORT_NAMES.get(c, c) for c in class_names]


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device):
    """Return true labels, predicted labels and softmax probabilities."""
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        all_probs.append(F.softmax(model(x.to(device)), dim=1).cpu())
        all_labels.append(y)
    probs = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()
    return y_true, probs.argmax(1), probs


# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------
def save_classification_report(y_true, y_pred, probs, class_names, out_dir):
    """Save the per-class report and summary metrics as .txt and .csv."""
    report_txt = classification_report(y_true, y_pred, labels=range(len(class_names)),
                                       target_names=class_names, digits=3, zero_division=0)
    report_dict = classification_report(y_true, y_pred, labels=range(len(class_names)),
                                        target_names=class_names, output_dict=True, zero_division=0)

    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
    try:
        summary["macro_auc_ovr"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:  # e.g. a class is missing in the test set
        summary["macro_auc_ovr"] = float("nan")

    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(f"Test set: {len(y_true)} images\n\n")
        for k, v in summary.items():
            f.write(f"{k:<20s} {v:.4f}\n")
        f.write("\n" + report_txt)

    pd.DataFrame(report_dict).transpose().to_csv(os.path.join(out_dir, "classification_report.csv"))
    pd.Series(summary).to_csv(os.path.join(out_dir, "summary_metrics.csv"), header=["value"])

    print("\n" + report_txt)
    for k, v in summary.items():
        print(f"{k:<20s} {v:.4f}")
    return summary


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, out_dir,
                          title="Normalized confusion matrix (test set)"):
    """Row-normalized confusion matrix (diagonal = recall per class) with raw counts."""
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=range(n))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    labels = short_names(class_names)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(os.path.join(out_dir, "confusion_matrix_counts.csv"))
    pd.DataFrame(cm_norm, index=labels, columns=labels).round(4).to_csv(
        os.path.join(out_dir, "confusion_matrix_normalized.csv"))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=9,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted view")
    ax.set_ylabel("True view")
    acc = np.trace(cm) / cm.sum()
    ax.set_title(f"{title}\nTest acc = {acc:.3f} (n = {cm.sum()})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix_normalized.png"), dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------
def plot_roc_curves(y_true, probs, class_names, out_dir):
    """One-vs-rest ROC curve per class."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for i, name in enumerate(short_names(class_names)):
        if (y_true == i).sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true == i, probs[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves, one-vs-rest (test set)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def plot_training_curves(history, out_dir):
    """Train loss and validation accuracy per epoch."""
    hist = pd.DataFrame(history)
    hist.to_csv(os.path.join(out_dir, "history.csv"), index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(hist["epoch"], hist["train_loss"], marker="o", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")
    ax1.set_title("Training loss")

    ax2.plot(hist["epoch"], hist["val_acc"], marker="o", markersize=3, color="tab:green")
    best = hist.loc[hist["val_acc"].idxmax()]
    ax2.axvline(best["epoch"], color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation accuracy")
    ax2.set_title(f"Validation accuracy (best {best['val_acc']:.3f} at epoch {int(best['epoch'])})")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Misclassified images
# ---------------------------------------------------------------------------
def save_misclassified(dataset, y_true, y_pred, probs, class_names, out_dir):
    """List all wrong predictions with file name, true/predicted class and confidence."""
    rows = []
    for i in np.where(y_true != y_pred)[0]:
        rows.append({
            "file": os.path.basename(dataset.samples[i][0]),
            "true": class_names[y_true[i]],
            "predicted": class_names[y_pred[i]],
            "confidence": round(float(probs[i, y_pred[i]]), 4),
        })
    df = pd.DataFrame(rows, columns=["file", "true", "predicted", "confidence"])
    df.sort_values("confidence", ascending=False).to_csv(os.path.join(out_dir, "misclassified.csv"), index=False)
    print(f"{len(df)} misclassified test images")


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
class GradCAM:
    """
    Grad-CAM (Selvaraju et al., 2017): weights the feature maps of the last
    convolutional block by the mean gradient of the target class score.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def __call__(self, x, class_idx=None):
        self.model.eval()
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

    def remove(self):
        self.handle.remove()


def get_target_layer(model, model_name):
    """Last convolutional block of each supported backbone."""
    if model_name == "efficientnet_b0":
        return model.features[-1]
    if model_name == "resnet50":
        return model.layer4[-1]
    if model_name == "convnext_tiny":
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target layer defined for {model_name}")


def denormalize(x):
    """Undo ImageNet normalization and return a 2D grayscale image in [0, 1]."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img = (x.cpu() * std + mean).clamp(0, 1)
    return img[0].numpy()  # all 3 channels are identical (grayscale)


def _draw_overlay(ax, img, cam, title):
    ax.imshow(img, cmap="gray")
    ax.imshow(cam, cmap="jet", alpha=0.4)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def _save_single(img, cam, title, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6, 3.2))
    a1.imshow(img, cmap="gray")
    a1.set_title("Original", fontsize=8)
    a1.axis("off")
    _draw_overlay(a2, img, cam, "Grad-CAM")
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_gradcam(model, model_name, dataset, y_true, y_pred, probs, class_names, device, out_dir,
                n_per_class=4, n_misclassified=16, seed=42):
    """
    Grad-CAM for the predicted class:
    - n_per_class randomly chosen correctly classified images per class
    - the n_misclassified most confident errors
    """
    rng = np.random.default_rng(seed)
    labels = short_names(class_names)
    cam_dir_correct = os.path.join(out_dir, "gradcam", "correct")
    cam_dir_wrong = os.path.join(out_dir, "gradcam", "misclassified")
    os.makedirs(cam_dir_correct, exist_ok=True)
    os.makedirs(cam_dir_wrong, exist_ok=True)

    gradcam = GradCAM(model, get_target_layer(model, model_name))

    def compute(i):
        x, _ = dataset[i]  # eval transform, same as in the test loader
        cam, _ = gradcam(x.unsqueeze(0).to(device), class_idx=int(y_pred[i]))
        return denormalize(x), cam

    def file_stem(i):
        return os.path.splitext(os.path.basename(dataset.samples[i][0]))[0]

    # Correctly classified: rows = classes, columns = examples
    n_cls = len(class_names)
    fig, axes = plt.subplots(n_cls, n_per_class, figsize=(2.4 * n_per_class, 2.5 * n_cls), squeeze=False)
    for c in range(n_cls):
        candidates = np.where((y_true == c) & (y_pred == c))[0]
        chosen = rng.choice(candidates, size=min(n_per_class, len(candidates)), replace=False) \
            if len(candidates) else []
        for k in range(n_per_class):
            ax = axes[c, k]
            if k >= len(chosen):
                ax.axis("off")
                continue
            i = chosen[k]
            img, cam = compute(i)
            title = f"{labels[c]}\np = {probs[i, c]:.2f}"
            _draw_overlay(ax, img, cam, title)
            _save_single(img, cam, f"{file_stem(i)} | {title.replace(chr(10), ' | ')}",
                         os.path.join(cam_dir_correct, f"{file_stem(i)}.png"))
    fig.suptitle("Grad-CAM - correctly classified test images", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gradcam_correct.png"), dpi=200)
    plt.close(fig)

    # Misclassified: most confident errors first
    wrong = np.where(y_true != y_pred)[0]
    wrong = wrong[np.argsort(-probs[wrong, y_pred[wrong]])][:n_misclassified]
    if len(wrong):
        n_cols = 4
        n_rows = int(np.ceil(len(wrong) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.7 * n_rows), squeeze=False)
        for k, ax in enumerate(axes.flat):
            if k >= len(wrong):
                ax.axis("off")
                continue
            i = wrong[k]
            img, cam = compute(i)
            title = f"true: {labels[y_true[i]]}\npred: {labels[y_pred[i]]} (p = {probs[i, y_pred[i]]:.2f})"
            _draw_overlay(ax, img, cam, title)
            _save_single(img, cam, f"{file_stem(i)} | {title.replace(chr(10), ' | ')}",
                         os.path.join(cam_dir_wrong, f"{file_stem(i)}.png"))
        fig.suptitle("Grad-CAM - misclassified test images (most confident first)", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "gradcam_misclassified.png"), dpi=200)
        plt.close(fig)

    gradcam.remove()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_visualization(model, model_name, loaders, class_names, device, out_dir, history=None):
    """Create all plots and reports for the test set and save them to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    model = model.to(device)

    if history:
        plot_training_curves(history, out_dir)

    test_loader = loaders["test"]
    dataset = test_loader.dataset  # test loader is not shuffled -> order matches dataset.samples
    y_true, y_pred, probs = predict(model, test_loader, device)

    save_classification_report(y_true, y_pred, probs, class_names, out_dir)
    plot_confusion_matrix(y_true, y_pred, class_names, out_dir)
    plot_roc_curves(y_true, probs, class_names, out_dir)
    save_misclassified(dataset, y_true, y_pred, probs, class_names, out_dir)
    run_gradcam(model, model_name, dataset, y_true, y_pred, probs, class_names, device, out_dir)

    print(f"\nAll results saved to {out_dir}")


def main():
    # Imported here (not at module level) to avoid a circular import with main.py
    from main import BATCH_SIZE, DATA_DIR, IMG_SIZE, RESULTS_DIR

    parser = argparse.ArgumentParser(description="Visualize test set results of a trained model")
    parser.add_argument("--model", default="efficientnet_b0",
                        choices=["efficientnet_b0", "resnet50", "convnext_tiny"])
    parser.add_argument("--checkpoint", default=None,
                        help="Path to the model weights (default: results/<model>/best_model.pt)")
    args = parser.parse_args()

    out_dir = os.path.join(RESULTS_DIR, args.model)
    checkpoint = args.checkpoint or os.path.join(out_dir, "best_model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loaders, class_names = get_dataloaders(DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    model = build_model(args.model, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    history_path = os.path.join(out_dir, "history.csv")
    history = pd.read_csv(history_path).to_dict("records") if os.path.exists(history_path) else None

    run_visualization(model, args.model, loaders, class_names, device, out_dir, history)


if __name__ == "__main__":
    main()