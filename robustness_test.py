"""
Robustness test: how much image degradation can the trained model tolerate?

The trained model (no retraining) is evaluated on the held-out TEST set after each image
has been degraded with increasing severity. Perturbations relevant for ultrasound:

    gaussian_noise   additive noise on the whole image (strict test)      std in [0-1] intensity units
    gaussian_noise_sector  same noise, only inside the ultrasound sector  std in [0-1] intensity units
                     (more realistic: the black background stays black)
    speckle          multiplicative noise (stronger speckle)              relative std
    blur             Gaussian blur (poorer focus / lateral resolution)    sigma in pixels
    low_resolution   downsample and upsample again (older devices)        downsampling factor
    jpeg             JPEG compression (PACS export, screenshots)          JPEG quality (100 = none)
    gain             overall brightness / gain setting                    intensity factor (1 = none)
    contrast         gamma change (dynamic range / post-processing)       gamma (1 = none)
    rotation         probe tilt                                           degrees (random sign)
    shift            different probe position in the image                fraction of image size
    zoom             different depth setting                              scale (>1 zoom in, <1 zoom out)

All perturbations are applied to the 224x224 grayscale image the model sees
(after the same resizing as during evaluation), so the levels refer to that input.
Random parts (noise, rotation direction, shift direction) are seeded per image,
so every run gives identical results.

Output: results/<model>/robustness/
- robustness_results.csv     accuracy, balanced accuracy, macro F1, per-class recall,
                             mean confidence for every perturbation and level
- robustness_curves.png      balanced accuracy vs. severity, one panel per perturbation,
                             with the 95 % bootstrap interval of the clean test set
- robustness_per_class.png   recall per view vs. severity
- robustness_examples.png    one test image at every perturbation level (what the levels look like)
- summary.txt                per perturbation: up to which level the model stays within
                             5 / 10 percentage points of the clean result

Usage:
    python robustness_test.py --model efficientnet_b0
    python robustness_test.py --model efficientnet_b0 --noise-aug        # model trained with --noise-aug
    python robustness_test.py --model efficientnet_b0 --perturbations blur jpeg gain

Compare runs (reads the existing robustness_results.csv of each run, no model needed):
    python robustness_test.py --compare efficientnet_b0 efficientnet_b0_noiseaug
    -> results/robustness_comparison/ (comparison_curves.png, comparison.csv, comparison.txt)
"""
import argparse
import io
import os
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

CHANCE = 0.2
TOLERANCES = (0.05, 0.10)  # allowed drop in balanced accuracy for the summary


# ---------------------------------------------------------------------------
# Perturbations: PIL grayscale image ("L") -> PIL grayscale image, same size
# ---------------------------------------------------------------------------
def _to_float(img):
    return np.asarray(img, dtype=np.float32) / 255.0


def _to_img(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8), mode="L")


def gaussian_noise(img, std, rng):
    if std == 0:
        return img
    x = _to_float(img)
    return _to_img(x + rng.normal(0, std, x.shape))


def sector_mask(img):
    """Boolean mask of the ultrasound sector (non-black area, small holes closed)."""
    from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter
    m = gaussian_filter(_to_float(img), sigma=2) > 0.03
    m = binary_closing(m, structure=np.ones((7, 7)))
    return binary_fill_holes(m)


def gaussian_noise_sector(img, std, rng):
    if std == 0:
        return img
    x = _to_float(img)
    noise = rng.normal(0, std, x.shape) * sector_mask(img)
    return _to_img(x + noise)


def speckle(img, std, rng):
    if std == 0:
        return img
    from scipy.ndimage import gaussian_filter
    x = _to_float(img)
    # slightly smoothed noise gives the grain a speckle-like size of ~2 px
    noise = gaussian_filter(rng.normal(0, 1, x.shape), sigma=0.7)
    noise /= noise.std() + 1e-8
    return _to_img(x * (1 + std * noise))


def blur(img, sigma, rng):
    return img if sigma == 0 else img.filter(ImageFilter.GaussianBlur(sigma))


def low_resolution(img, factor, rng):
    if factor == 1:
        return img
    w, h = img.size
    small = img.resize((max(1, round(w / factor)), max(1, round(h / factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def jpeg(img, quality, rng):
    if quality >= 100:
        return img
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("L")


def gain(img, factor, rng):
    return img if factor == 1 else _to_img(_to_float(img) * factor)


def contrast(img, gamma, rng):
    return img if gamma == 1 else _to_img(_to_float(img) ** gamma)


def rotation(img, degrees, rng):
    if degrees == 0:
        return img
    sign = rng.choice([-1, 1])
    return img.rotate(sign * degrees, resample=Image.BILINEAR, fillcolor=0)


def shift(img, fraction, rng):
    if fraction == 0:
        return img
    w, h = img.size
    angle = rng.uniform(0, 2 * np.pi)  # random direction, fixed distance
    dx, dy = fraction * w * np.cos(angle), fraction * h * np.sin(angle)
    return img.rotate(0, translate=(dx, dy), resample=Image.BILINEAR, fillcolor=0)


def zoom(img, scale, rng):
    if scale == 1:
        return img
    w, h = img.size
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((nw, nh), Image.BILINEAR)
    if scale > 1:  # zoom in: centre crop
        left, top = (nw - w) // 2, (nh - h) // 2
        return resized.crop((left, top, left + w, top + h))
    canvas = Image.new("L", (w, h), 0)  # zoom out: pad with black
    canvas.paste(resized, ((w - nw) // 2, (h - nh) // 2))
    return canvas


# name: (function, levels, clean level, x-axis label)
PERTURBATIONS = {
    "gaussian_noise": (gaussian_noise, [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3], 0, "noise std"),
    "gaussian_noise_sector": (gaussian_noise_sector, [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3], 0,
                              "noise std (sector only)"),
    "speckle":        (speckle, [0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0], 0, "relative speckle std"),
    "blur":           (blur, [0, 0.5, 1, 1.5, 2, 3, 4, 6], 0, "Gaussian sigma (px)"),
    "low_resolution": (low_resolution, [1, 1.5, 2, 3, 4, 6, 8], 1, "downsampling factor"),
    "jpeg":           (jpeg, [100, 75, 50, 30, 20, 10, 5], 100, "JPEG quality"),
    "gain":           (gain, [0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0], 1.0, "intensity factor"),
    "contrast":       (contrast, [0.3, 0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 3.0], 1.0, "gamma"),
    "rotation":       (rotation, [0, 5, 10, 15, 20, 30, 45], 0, "rotation (degrees)"),
    "shift":          (shift, [0, 0.05, 0.1, 0.15, 0.2, 0.3], 0, "shift (fraction of image)"),
    "zoom":           (zoom, [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0], 1.0, "scale (>1 = zoom in)"),
}


def image_rng(perturbation, idx):
    """Reproducible random generator per perturbation and image (same for every level)."""
    return np.random.default_rng([zlib.crc32(perturbation.encode()), idx])


def load_gray(path, img_size):
    """Same resizing as the evaluation transform, kept in grayscale."""
    return Image.open(path).convert("L").resize((img_size, img_size), Image.BILINEAR)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred, probs, n_classes):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    row = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=range(n_classes), zero_division=0),
        "mean_confidence": float(probs.max(axis=1).mean()),
    }
    for c in range(n_classes):
        sel = y_true == c
        row[f"recall_view{c + 1}"] = float((y_pred[sel] == c).mean()) if sel.any() else np.nan
    return row


def bootstrap_ci(y_true, y_pred, n_boot=2000, seed=0):
    """95 % bootstrap interval of the balanced accuracy (resampling test images)."""
    from sklearn.metrics import balanced_accuracy_score
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
    return np.percentile(vals, [2.5, 97.5])


# ---------------------------------------------------------------------------
# Plots and summary
# ---------------------------------------------------------------------------
def plot_curves(res, clean, ci, out_dir):
    names = list(res["perturbation"].unique())
    n_cols = 4
    n_rows = int(np.ceil(len(names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.3 * n_rows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        d = res[res["perturbation"] == name].sort_values("level")
        clean_level = PERTURBATIONS[name][2]
        ax.axhspan(ci[0], ci[1], color="tab:blue", alpha=0.12, label="clean 95 % CI")
        ax.axhline(clean - TOLERANCES[0], color="tab:orange", linestyle=":", linewidth=1, label="-5 pp")
        ax.axhline(CHANCE, color="gray", linestyle="--", linewidth=0.8, label="chance")
        ax.plot(d["level"], d["balanced_accuracy"], marker="o", markersize=4, color="tab:blue")
        ax.axvline(clean_level, color="black", linewidth=0.6, alpha=0.5)
        if name == "jpeg":
            ax.invert_xaxis()  # left = no compression
        ax.set_ylim(0, 1.03)
        ax.set_title(name)
        ax.set_xlabel(PERTURBATIONS[name][3], fontsize=8)
        ax.set_ylabel("balanced accuracy", fontsize=8)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Robustness on the test set (clean balanced accuracy {clean:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "robustness_curves.png"), dpi=200)
    plt.close(fig)


def plot_per_class(res, class_labels, out_dir):
    names = list(res["perturbation"].unique())
    n_cols = 4
    n_rows = int(np.ceil(len(names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.3 * n_rows), squeeze=False)
    for ax, name in zip(axes.flat, names):
        d = res[res["perturbation"] == name].sort_values("level")
        for c, label in enumerate(class_labels):
            ax.plot(d["level"], d[f"recall_view{c + 1}"], marker=".", label=label)
        if name == "jpeg":
            ax.invert_xaxis()
        ax.set_ylim(0, 1.03)
        ax.set_title(name)
        ax.set_xlabel(PERTURBATIONS[name][3], fontsize=8)
        ax.set_ylabel("recall", fontsize=8)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6, loc="lower left")
    fig.suptitle("Recall per view under perturbation (test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "robustness_per_class.png"), dpi=200)
    plt.close(fig)


def plot_examples(path, img_size, names, out_dir):
    """One test image at every level of every perturbation."""
    base = load_gray(path, img_size)
    max_levels = max(len(PERTURBATIONS[n][1]) for n in names)
    fig, axes = plt.subplots(len(names), max_levels, figsize=(1.6 * max_levels, 1.75 * len(names)),
                             squeeze=False)
    for r, name in enumerate(names):
        fn, levels, _, _ = PERTURBATIONS[name]
        for c in range(max_levels):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(levels):
                continue
            ax.imshow(fn(base, levels[c], image_rng(name, 0)), cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{levels[c]:g}", fontsize=7)
        axes[r, 0].text(-0.08, 0.5, name, transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=8)
    fig.suptitle("What the perturbation levels look like (model input, 224 x 224)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(out_dir, "robustness_examples.png"), dpi=150)
    plt.close(fig)


def tolerated_range(d, clean, clean_level, tol):
    """Walk from the clean level outwards; return the furthest levels still within tolerance."""
    d = d.sort_values("level").reset_index(drop=True)
    i0 = int(np.argmin(np.abs(d["level"].values - clean_level)))
    lo = hi = i0
    while lo - 1 >= 0 and d.loc[lo - 1, "balanced_accuracy"] >= clean - tol:
        lo -= 1
    while hi + 1 < len(d) and d.loc[hi + 1, "balanced_accuracy"] >= clean - tol:
        hi += 1
    return d.loc[lo, "level"], d.loc[hi, "level"], lo == 0, hi == len(d) - 1


def tolerance_cells(d, clean, name):
    """Tolerated range of one perturbation as text, for 5 and 10 percentage points."""
    clean_level = PERTURBATIONS[name][2]
    cells = []
    for tol in TOLERANCES:
        lo, hi, lo_end, hi_end = tolerated_range(d, clean, clean_level, tol)
        if lo == hi == clean_level:
            cells.append("none")
            continue
        lo_s = f"{lo:g}{'+' if lo_end and lo != clean_level else ''}"
        hi_s = f"{hi:g}{'+' if hi_end and hi != clean_level else ''}"
        if name == "jpeg":  # clean level (100) is the highest value; tolerance extends downwards
            cells.append(f"quality >= {lo_s}")
        elif lo == clean_level:
            cells.append(f"up to {hi_s}")
        elif hi == clean_level:
            cells.append(f"down to {lo_s}")
        else:
            cells.append(f"{lo_s} to {hi_s}")
    return cells


def clean_score(res):
    """Balanced accuracy on the unperturbed test set, taken from the results table."""
    for name in res["perturbation"].unique():
        d = res[(res["perturbation"] == name) & (res["level"] == PERTURBATIONS[name][2])]
        if len(d):
            return float(d["balanced_accuracy"].iloc[0])
    raise ValueError("No clean level found in the results")


def write_summary(res, clean, ci, n_images, out_dir, noise_aug=False):
    lines = [
        "Robustness summary (test set)",
        "=============================",
        f"Clean balanced accuracy: {clean:.3f} (95 % bootstrap CI {ci[0]:.3f}-{ci[1]:.3f}, n = {n_images})",
        "Drops smaller than the width of this interval are within random variation.",
        "",
        "Range of each perturbation in which the balanced accuracy stays within",
        "5 / 10 percentage points of the clean result ('+' = not reached within the tested levels):",
        "",
        f"{'perturbation':<22s} {'within 5 pp':>20s} {'within 10 pp':>20s} {'worst tested':>14s}",
    ]
    for name in res["perturbation"].unique():
        d = res[res["perturbation"] == name]
        cells = tolerance_cells(d, clean, name)
        lines.append(f"{name:<22s} {cells[0]:>20s} {cells[1]:>20s} {d['balanced_accuracy'].min():>14.3f}")

    lines += ["",
              "Interpretation: compare the tolerated levels with robustness_examples.png. A model that",
              "only fails at levels where the image is clearly unusable is robust for practical purposes.",
              "Training used rotations up to 10 degrees, slight crops and brightness/contrast jitter of 20 %,",
              "so robustness up to roughly these levels is expected; beyond them it shows generalisation."]
    if noise_aug:
        lines += ["This model was trained WITH noise augmentation (Gaussian noise std up to 0.1, speckle up to",
                  "0.4, blur sigma up to 2 px). Robustness within these levels shows that the augmentation worked;",
                  "only levels beyond them test generalisation."]
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)


# ---------------------------------------------------------------------------
# Comparison of several runs (e.g. without vs. with noise augmentation)
# ---------------------------------------------------------------------------
def compare_runs(runs, results_dir):
    tables = {}
    for run in runs:
        path = os.path.join(results_dir, run, "robustness", "robustness_results.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found - run the robustness test for '{run}' first")
        tables[run] = pd.read_csv(path)

    out_dir = os.path.join(results_dir, "robustness_comparison")
    os.makedirs(out_dir, exist_ok=True)

    # perturbations present in all runs, in the usual order
    common = [p for p in PERTURBATIONS if all(p in t["perturbation"].values for t in tables.values())]
    missing = sorted({p for t in tables.values() for p in t["perturbation"].unique()} - set(common))
    if missing:
        print(f"Not in every run (skipped, rerun the older run to include them): {missing}")

    long = pd.concat([t.assign(run=r) for r, t in tables.items()], ignore_index=True)
    long[long["perturbation"].isin(common)].round(4).to_csv(os.path.join(out_dir, "comparison.csv"), index=False)

    # curves
    n_cols = 4
    n_rows = int(np.ceil(len(common) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.3 * n_rows), squeeze=False)
    for ax, name in zip(axes.flat, common):
        for run, t in tables.items():
            d = t[t["perturbation"] == name].sort_values("level")
            ax.plot(d["level"], d["balanced_accuracy"], marker="o", markersize=3.5, label=run)
        ax.axhline(CHANCE, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(PERTURBATIONS[name][2], color="black", linewidth=0.6, alpha=0.5)
        if name == "jpeg":
            ax.invert_xaxis()
        ax.set_ylim(0, 1.03)
        ax.set_title(name)
        ax.set_xlabel(PERTURBATIONS[name][3], fontsize=8)
        ax.set_ylabel("balanced accuracy", fontsize=8)
    for ax in list(axes.flat)[len(common):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Robustness comparison (test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_curves.png"), dpi=200)
    plt.close(fig)

    # text table: clean score and tolerated range (within 5 pp of each run's own clean score)
    cleans = {r: clean_score(t) for r, t in tables.items()}
    width = max(20, max(len(r) for r in runs) + 2)
    lines = ["Robustness comparison (test set)", "================================", ""]
    lines += [f"Clean balanced accuracy  {r}: {c:.3f}" for r, c in cleans.items()]
    lines += ["", "Range within 5 percentage points of each run's own clean result:", "",
              f"{'perturbation':<22s}" + "".join(f"{r:>{width}s}" for r in runs)]
    for name in common:
        row = f"{name:<22s}"
        for r, t in tables.items():
            row += f"{tolerance_cells(t[t['perturbation'] == name], cleans[r], name)[0]:>{width}s}"
        lines.append(row)
    lines += ["", "Worst balanced accuracy over all tested levels:", "",
              f"{'perturbation':<22s}" + "".join(f"{r:>{width}s}" for r in runs)]
    for name in common:
        row = f"{name:<22s}"
        for r, t in tables.items():
            row += f"{t.loc[t['perturbation'] == name, 'balanced_accuracy'].min():>{width}.3f}"
        lines.append(row)
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "comparison.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nComparison saved to {out_dir}")


# ---------------------------------------------------------------------------
def main():
    from main import BATCH_SIZE, DATA_DIR, IMG_SIZE, RESULTS_DIR

    parser = argparse.ArgumentParser(description="Robustness of the trained model to image degradation")
    parser.add_argument("--model", default="efficientnet_b0",
                        choices=["efficientnet_b0", "resnet50", "convnext_tiny"])
    parser.add_argument("--noise-aug", action="store_true",
                        help="Test the model trained with --noise-aug (results/<model>_noiseaug/)")
    parser.add_argument("--checkpoint", default=None,
                        help="Model weights (default: results/<run>/best_model.pt)")
    parser.add_argument("--perturbations", nargs="+", default=list(PERTURBATIONS),
                        choices=list(PERTURBATIONS), help="Subset of perturbations to run")
    parser.add_argument("--compare", nargs="+", metavar="RUN",
                        help="Compare existing results of these runs (folder names in results/), "
                             "e.g. --compare efficientnet_b0 efficientnet_b0_noiseaug")
    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare, RESULTS_DIR)
        return

    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import datasets

    from models import build_model
    from preprocess_data import CLASSES, get_transforms
    from visualization import predict, short_names

    class PerturbedDataset(Dataset):
        def __init__(self, samples, name, level, transform):
            self.samples, self.name, self.level, self.transform = samples, name, level, transform
            self.fn = PERTURBATIONS[name][0]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            img = self.fn(load_gray(path, IMG_SIZE), self.level, image_rng(self.name, idx))
            return self.transform(img), label

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.model + ("_noiseaug" if args.noise_aug else "")
    out_dir = os.path.join(RESULTS_DIR, run_name, "robustness")
    os.makedirs(out_dir, exist_ok=True)

    class_names = [CLASSES[k] for k in sorted(CLASSES)]
    n_classes = len(class_names)
    checkpoint = args.checkpoint or os.path.join(RESULTS_DIR, run_name, "best_model.pt")
    model = build_model(args.model, num_classes=n_classes, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device).eval()

    _, eval_tf = get_transforms(IMG_SIZE)  # evaluation never uses augmentation
    samples = datasets.ImageFolder(os.path.join(DATA_DIR, "test")).samples
    print(f"Device: {device} | test images: {len(samples)} | checkpoint: {checkpoint}")

    def run(name, level):
        ds = PerturbedDataset(samples, name, level, eval_tf)
        return predict(model, DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4), device)

    # Clean reference (should reproduce the test accuracy of main.py)
    y_true, y_pred, probs = run(args.perturbations[0], PERTURBATIONS[args.perturbations[0]][2])
    clean_metrics = metrics(y_true, y_pred, probs, n_classes)
    clean = clean_metrics["balanced_accuracy"]
    ci = bootstrap_ci(y_true, y_pred)
    print(f"Clean: accuracy {clean_metrics['accuracy']:.3f} | balanced accuracy {clean:.3f} "
          f"(95 % CI {ci[0]:.3f}-{ci[1]:.3f})")

    rows = []
    for name in args.perturbations:
        _, levels, clean_level, _ = PERTURBATIONS[name]
        for level in levels:
            if level == clean_level:
                m = clean_metrics
            else:
                yt, yp, pr = run(name, level)
                m = metrics(yt, yp, pr, n_classes)
            rows.append({"perturbation": name, "level": level, **m})
            print(f"{name:<22s} level {level:>6g} | balanced acc {m['balanced_accuracy']:.3f} "
                  f"| mean confidence {m['mean_confidence']:.3f}")

    res = pd.DataFrame(rows)
    res.round(4).to_csv(os.path.join(out_dir, "robustness_results.csv"), index=False)

    plot_curves(res, clean, ci, out_dir)
    plot_per_class(res, short_names(class_names), out_dir)
    plot_examples(samples[0][0], IMG_SIZE, args.perturbations, out_dir)
    write_summary(res, clean, ci, len(samples), out_dir, noise_aug=args.noise_aug)
    print(f"\nAll robustness results saved to {out_dir}")


if __name__ == "__main__":
    main()