"""
Shortcut checks: does the model recognise the view from the anatomy,
or from image geometry / acquisition artefacts?

1. Geometry analysis (all images)
   Image size, area / position / extent of the ultrasound sector and mean brightness, per class.
   -> geometry_features.csv, geometry_per_class.csv, image_sizes_per_class.csv, geometry_boxplots.png

2. Geometry-only baselines
   Random forests that see ONLY image size, sector geometry or a 16x16 sector silhouette
   (no anatomy), trained on train+val and tested on the same patient-level test set as the CNN.
   High accuracy = the view can be read from geometry alone -> potential shortcut.
   -> geometry_baselines.csv

3. Near-duplicate check
   Most similar image pairs from DIFFERENT patients (similarity of the fine image structure,
   independent of the sector outline). Identical images under different patient IDs would
   undermine the patient-level split.
   -> near_duplicates.csv, near_duplicates_top.png

4. Masking tests with the trained CNN (test set)
   - original               unchanged images
   - common_sector          everything outside the region shared by ~all sectors is blacked out
                            -> identical outline for all images, only the anatomy inside remains
   - outline_only           sector content replaced by random speckle-like noise
                            -> sector outline kept, anatomy removed
   The model relies on anatomy if common_sector stays close to original
   and outline_only drops towards chance level (0.20).
   -> masking_results.csv, masking_results.png, masking_examples.png

All results: results/<model>/shortcut_checks/

Usage:
    python shortcut_checks.py --model efficientnet_b0
    python shortcut_checks.py --geometry-only        # checks 1-3, no trained model needed
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from models import build_model
from preprocess_data import CLASSES, MEAN, STD, collect_images, split_by_patient
from visualization import predict, short_names

CHANCE = 1 / len(CLASSES)
GEOMETRY_COLS = ["sector_area", "x_min", "x_max", "y_min", "y_max", "cx", "cy"]
SIZE_COLS = ["width", "height", "aspect"]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def to_gray_tensor(img, img_size):
    """Same resizing as the evaluation pipeline, but single channel and not normalized: [1, H, W] in [0, 1]."""
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return tf(img)


def sector_mask(gray, threshold=0.03, close_kernel=15):
    """Binary mask of the ultrasound sector (non-black area) with small holes closed. gray: [H, W]."""
    x = F.avg_pool2d(gray[None, None], 5, stride=1, padding=2)
    m = (x > threshold).float()
    pad = close_kernel // 2
    m = F.max_pool2d(m, close_kernel, stride=1, padding=pad)      # dilation
    m = -F.max_pool2d(-m, close_kernel, stride=1, padding=pad)    # erosion
    return m[0, 0]


def highpass_signature(gray, size=64):
    """Fine image structure (speckle / anatomy) without the sector outline, z-normalized, flattened."""
    g = F.adaptive_avg_pool2d(gray[None, None], size)
    hp = g - F.avg_pool2d(g, 9, stride=1, padding=4, count_include_pad=False)
    hp = hp.flatten()
    return ((hp - hp.mean()) / (hp.std() + 1e-8)).numpy()


# ---------------------------------------------------------------------------
# 1. Geometry analysis
# ---------------------------------------------------------------------------
def compute_geometry(df, src_dir, img_size):
    """Per-image geometry features, 16x16 sector silhouettes and high-pass signatures."""
    rows, silhouettes, signatures = [], [], []
    for _, r in df.iterrows():
        img = Image.open(os.path.join(src_dir, r["file"]))
        w, h = img.size
        g = to_gray_tensor(img.convert("RGB"), img_size)[0]
        m = sector_mask(g)
        ys, xs = torch.nonzero(m, as_tuple=True)

        feats = {"file": r["file"], "patient": r["patient"], "label": r["label"],
                 "width": w, "height": h, "aspect": w / h, "sector_area": float(m.mean())}
        if len(xs):
            feats.update({
                "x_min": float(xs.min()) / img_size, "x_max": float(xs.max()) / img_size,
                "y_min": float(ys.min()) / img_size, "y_max": float(ys.max()) / img_size,
                "cx": float(xs.float().mean()) / img_size, "cy": float(ys.float().mean()) / img_size,
                "mean_intensity": float(g[m > 0].mean()),
            })
        rows.append(feats)
        silhouettes.append(F.adaptive_avg_pool2d(m[None, None], 16).flatten().numpy())
        signatures.append(highpass_signature(g))

    return pd.DataFrame(rows), np.stack(silhouettes), np.stack(signatures)


def save_geometry_analysis(feat_df, out_dir):
    feat_df.to_csv(os.path.join(out_dir, "geometry_features.csv"), index=False)

    cols = SIZE_COLS + GEOMETRY_COLS + ["mean_intensity"]
    per_class = feat_df.groupby("label")[cols].agg(["mean", "std"]).round(3)
    per_class.to_csv(os.path.join(out_dir, "geometry_per_class.csv"))

    sizes = pd.crosstab(feat_df["width"].astype(str) + "x" + feat_df["height"].astype(str), feat_df["label"])
    sizes.index.name = "image size (w x h)"
    sizes.to_csv(os.path.join(out_dir, "image_sizes_per_class.csv"))
    print("\nImage sizes per class (rows = size, columns = view):")
    print(sizes)

    plot_cols = ["width", "height", "sector_area", "cx", "cy", "y_max", "x_min", "x_max", "mean_intensity"]
    labels = sorted(feat_df["label"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for ax, col in zip(axes.flat, plot_cols):
        ax.boxplot([feat_df.loc[feat_df["label"] == l, col].dropna() for l in labels])
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels([str(l) for l in labels])
        ax.set_xlabel("View")
        ax.set_title(col)
    fig.suptitle("Image geometry per view (should look similar across views)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "geometry_boxplots.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Geometry-only baselines
# ---------------------------------------------------------------------------
def geometry_baselines(feat_df, silhouettes, out_dir, seed=42):
    """Train classifiers on geometry only and test them on the patient-level test split."""
    is_test = (feat_df["split"] == "test").values
    y = feat_df["label"].values
    scalars = feat_df[SIZE_COLS + GEOMETRY_COLS + ["mean_intensity"]].fillna(0).values

    feature_sets = {
        "image size (width, height, aspect)": feat_df[SIZE_COLS].values,
        "sector geometry (area, extent, centre)": feat_df[GEOMETRY_COLS].fillna(0).values,
        "sector silhouette (16x16 mask)": silhouettes,
        "all geometry + mean brightness": scalars,
    }

    rows = []
    for name, X in feature_sets.items():
        clf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
        clf.fit(X[~is_test], y[~is_test])
        pred = clf.predict(X[is_test])
        rows.append({"features": name,
                     "accuracy": accuracy_score(y[is_test], pred),
                     "balanced_accuracy": balanced_accuracy_score(y[is_test], pred)})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "geometry_baselines.csv"), index=False)
    print(f"\nGeometry-only baselines (test set, chance level = {CHANCE:.2f}):")
    print(res.to_string(index=False, float_format="%.3f"))
    return res


# ---------------------------------------------------------------------------
# 3. Near-duplicate check
# ---------------------------------------------------------------------------
def near_duplicates(feat_df, signatures, src_dir, out_dir, threshold=0.9, top_n=20):
    """Find the most similar image pairs belonging to different patients."""
    sim = signatures @ signatures.T / signatures.shape[1]  # correlation of high-pass signatures
    patients = feat_df["patient"].values
    iu, ju = np.triu_indices(len(feat_df), k=1)
    cross = patients[iu] != patients[ju]
    iu, ju, s = iu[cross], ju[cross], sim[iu[cross], ju[cross]]
    order = np.argsort(-s)

    keep = order[: max(top_n, int((s >= threshold).sum()))]
    pairs = pd.DataFrame({
        "file_a": feat_df["file"].values[iu[keep]], "patient_a": patients[iu[keep]],
        "split_a": feat_df["split"].values[iu[keep]], "view_a": feat_df["label"].values[iu[keep]],
        "file_b": feat_df["file"].values[ju[keep]], "patient_b": patients[ju[keep]],
        "split_b": feat_df["split"].values[ju[keep]], "view_b": feat_df["label"].values[ju[keep]],
        "similarity": s[keep].round(4),
    })
    pairs["suspicious"] = pairs["similarity"] >= threshold
    pairs.to_csv(os.path.join(out_dir, "near_duplicates.csv"), index=False)

    n_susp = int(pairs["suspicious"].sum())
    n_cross_split = int((pairs["suspicious"] & (pairs["split_a"] != pairs["split_b"])).sum())
    print(f"\nNear-duplicates between different patients (similarity >= {threshold}): {n_susp} pairs, "
          f"{n_cross_split} of them across different splits. Highest similarity: {s[order[0]]:.3f}")

    # Show the 8 most similar pairs for a visual check
    n_show = min(8, len(pairs))
    fig, axes = plt.subplots(n_show, 2, figsize=(5, 2.5 * n_show), squeeze=False)
    for r in range(n_show):
        for c, key in enumerate(["a", "b"]):
            img = Image.open(os.path.join(src_dir, pairs.loc[r, f"file_{key}"])).convert("L")
            axes[r, c].imshow(img, cmap="gray")
            axes[r, c].set_title(f"{pairs.loc[r, f'patient_{key}']} | view {pairs.loc[r, f'view_{key}']} | "
                                 f"{pairs.loc[r, f'split_{key}']}", fontsize=7)
            axes[r, c].axis("off")
        axes[r, 0].text(-0.05, 0.5, f"sim {pairs.loc[r, 'similarity']:.2f}", transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=8)
    fig.suptitle("Most similar pairs from different patients", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "near_duplicates_top.png"), dpi=150)
    plt.close(fig)
    return pairs


# ---------------------------------------------------------------------------
# 4. Masking tests
# ---------------------------------------------------------------------------
def compute_common_mask(image_paths, img_size, min_fraction=0.95):
    """Pixels that lie inside the sector in at least `min_fraction` of the images."""
    acc = torch.zeros(img_size, img_size)
    for p in image_paths:
        acc += sector_mask(to_gray_tensor(Image.open(p).convert("RGB"), img_size)[0])
    return (acc / len(image_paths) >= min_fraction).float()


class MaskedDataset(Dataset):
    """Test images with one of the masking conditions applied before normalization."""

    def __init__(self, samples, mode, img_size, common_mask=None):
        self.samples, self.mode, self.img_size, self.common_mask = samples, mode, img_size, common_mask
        self.normalize = transforms.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.samples)

    def masked_gray(self, idx):
        path, _ = self.samples[idx]
        g = to_gray_tensor(Image.open(path).convert("RGB"), self.img_size)  # [1, H, W]

        if self.mode == "common_sector":
            g = g * self.common_mask

        elif self.mode == "outline_only":
            m = sector_mask(g[0])
            inside = g[0][m > 0]
            mu, sd = (inside.mean(), inside.std()) if len(inside) else (0.3, 0.1)
            gen = torch.Generator().manual_seed(idx)
            noise = torch.randn(g.shape[1:], generator=gen)
            noise = F.avg_pool2d(noise[None, None], 3, stride=1, padding=1)[0, 0]  # speckle-like grain
            noise = (noise - noise.mean()) / (noise.std() + 1e-8)
            g = ((mu + sd * noise).clamp(0, 1) * m)[None]

        return g

    def __getitem__(self, idx):
        g = self.masked_gray(idx)
        return self.normalize(g.repeat(3, 1, 1)), self.samples[idx][1]


def masking_tests(model, test_dir, train_dir, class_names, device, img_size, batch_size, out_dir):
    test_samples = datasets.ImageFolder(test_dir).samples
    train_paths = [p for p, _ in datasets.ImageFolder(train_dir).samples]

    common_mask = compute_common_mask(train_paths, img_size)
    print(f"\nCommon sector region covers {common_mask.mean():.1%} of the image area")
    if common_mask.mean() < 0.15:
        print("WARNING: the common region is small - the common_sector test removes a lot of anatomy as well.")

    modes = ["original", "common_sector", "outline_only"]
    rows, datasets_ = [], {}
    for mode in modes:
        ds = MaskedDataset(test_samples, mode, img_size, common_mask)
        datasets_[mode] = ds
        y_true, y_pred, _ = predict(model, DataLoader(ds, batch_size=batch_size, num_workers=4), device)
        row = {"condition": mode,
               "accuracy": accuracy_score(y_true, y_pred),
               "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)}
        for i, name in enumerate(short_names(class_names)):
            sel = y_true == i
            row[f"recall {name}"] = (y_pred[sel] == i).mean() if sel.any() else np.nan
        rows.append(row)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "masking_results.csv"), index=False)
    print(f"\nMasking tests (test set, chance level = {CHANCE:.2f}):")
    print(res[["condition", "accuracy", "balanced_accuracy"]].to_string(index=False, float_format="%.3f"))

    # Bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(res["condition"], res["balanced_accuracy"], color=["tab:blue", "tab:green", "tab:red"])
    ax.axhline(CHANCE, color="gray", linestyle="--", linewidth=0.8, label="chance level")
    for i, v in enumerate(res["balanced_accuracy"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Balanced accuracy (test set)")
    ax.set_title("CNN performance under masking")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "masking_results.png"), dpi=200)
    plt.close(fig)

    # Example images: one per class, every condition
    labels = np.array([y for _, y in test_samples])
    examples = [int(np.where(labels == c)[0][0]) for c in range(len(class_names)) if (labels == c).any()]
    fig, axes = plt.subplots(len(examples), len(modes), figsize=(2.6 * len(modes), 2.6 * len(examples)),
                             squeeze=False)
    for r, idx in enumerate(examples):
        for c, mode in enumerate(modes):
            axes[r, c].imshow(datasets_[mode].masked_gray(idx)[0], cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(mode, fontsize=9)
        axes[r, 0].text(-0.05, 0.5, short_names(class_names)[labels[idx]], transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "masking_examples.png"), dpi=150)
    plt.close(fig)
    return res


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------
def interpret(baselines, masking, duplicates, out_dir):
    lines = ["Shortcut check summary", "======================", ""]

    best = baselines.loc[baselines["balanced_accuracy"].idxmax()]
    lines.append(f"Best geometry-only baseline: {best['features']} -> balanced accuracy "
                 f"{best['balanced_accuracy']:.3f} (chance {CHANCE:.2f})")
    if best["balanced_accuracy"] > 0.6:
        lines.append("  -> The view can largely be predicted from geometry alone. Check whether the CNN uses it "
                     "(see masking tests) and consider standardizing the sector/crop.")
    elif best["balanced_accuracy"] > 0.35:
        lines.append("  -> Geometry carries some information about the view (partly expected, e.g. depth settings). "
                     "The masking tests show whether the CNN depends on it.")
    else:
        lines.append("  -> Geometry carries little information about the view. Good.")

    n_susp = int(duplicates["suspicious"].sum())
    lines.append("")
    lines.append(f"Near-duplicate pairs between different patients: {n_susp}")
    lines.append("  -> Inspect near_duplicates_top.png / near_duplicates.csv." if n_susp else "  -> None found. Good.")

    if masking is not None:
        m = masking.set_index("condition")["balanced_accuracy"]
        lines += ["",
                  f"CNN balanced accuracy: original {m['original']:.3f} | common_sector {m['common_sector']:.3f} | "
                  f"outline_only {m['outline_only']:.3f}"]
        if m["original"] - m["common_sector"] <= 0.05:
            lines.append("  -> Identical outlines for all images barely change the result: "
                         "the CNN relies on the content inside the sector.")
        else:
            lines.append("  -> Performance drops when the outline is standardized: the CNN partly uses the "
                         "sector outline / periphery (or anatomy near the edges was removed).")
        if m["outline_only"] <= 0.35:
            lines.append("  -> Without anatomy the CNN is close to chance: it does not classify from the outline alone.")
        else:
            lines.append("  -> The CNN still performs above chance from the outline alone: "
                         "geometry is a shortcut it has learned.")

    text = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)


# ---------------------------------------------------------------------------
def main():
    from main import BATCH_SIZE, DATA_DIR, IMG_SIZE, RESULTS_DIR, SEED, SRC_DIR

    parser = argparse.ArgumentParser(description="Shortcut checks for the view classifier")
    parser.add_argument("--model", default="efficientnet_b0",
                        choices=["efficientnet_b0", "resnet50", "convnext_tiny"])
    parser.add_argument("--checkpoint", default=None,
                        help="Model weights (default: results/<model>/best_model.pt)")
    parser.add_argument("--geometry-only", action="store_true",
                        help="Only run checks 1-3 (no trained model needed)")
    args = parser.parse_args()

    out_dir = os.path.join(RESULTS_DIR, args.model, "shortcut_checks")
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use the same patient-level split as the CNN
    split_csv = os.path.join(DATA_DIR, "split_overview.csv")
    if os.path.exists(split_csv):
        df = pd.read_csv(split_csv)
    else:
        print(f"{split_csv} not found - creating the split with the same seed")
        df = split_by_patient(collect_images(SRC_DIR), seed=SEED)

    print("Computing image geometry ...")
    feat_df, silhouettes, signatures = compute_geometry(df, SRC_DIR, IMG_SIZE)
    feat_df["split"] = df["split"].values

    save_geometry_analysis(feat_df, out_dir)
    baselines = geometry_baselines(feat_df, silhouettes, out_dir, seed=SEED)
    duplicates = near_duplicates(feat_df, signatures, SRC_DIR, out_dir)

    masking = None
    if not args.geometry_only:
        checkpoint = args.checkpoint or os.path.join(RESULTS_DIR, args.model, "best_model.pt")
        class_names = [CLASSES[k] for k in sorted(CLASSES)]
        model = build_model(args.model, num_classes=len(class_names), pretrained=False)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model = model.to(device)
        masking = masking_tests(model, os.path.join(DATA_DIR, "test"), os.path.join(DATA_DIR, "train"),
                                class_names, device, IMG_SIZE, BATCH_SIZE, out_dir)

    interpret(baselines, masking, duplicates, out_dir)
    print(f"\nAll shortcut check results saved to {out_dir}")


if __name__ == "__main__":
    main()