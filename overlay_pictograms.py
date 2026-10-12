"""
Overlay the matching probe-position pictogram on classified ultrasound images.

For every image of the chosen split, the trained model predicts the view (1-5).
The pictogram template of the PREDICTED view is pasted into the bottom-right
corner of the original-resolution image.

Output: results/<model>/pictogram_overlay/<split>/
- <predicted class>/<image>.png     images with pictogram
- misclassified/true<X>_pred<Y>_<image>.png
                                    wrong predictions only, with red frame and label
- predictions.csv                   file, true view, predicted view, confidence, correct

Pictogram templates (PNG) are looked up in PICTOGRAM_DIR. Each file name must
contain the view number 1-5 as a separate number, e.g. "1_left.png", "pictogram_3.png".
Alternatively set the file names explicitly in PICTOGRAM_FILES below.

Usage:
    python overlay_pictograms.py --model efficientnet_b0              # test set
    python overlay_pictograms.py --model efficientnet_b0 --split all  # train + val + test
"""
import argparse
import os
import re

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets

from models import build_model
from preprocess_data import CLASSES, get_transforms


# ---------------- Configuration ----------------
PICTOGRAM_DIR = "/home/xrad/KI_Piktogramm/github/KI-Piktogramme/Pictogram_templates"


PICTOGRAM_FILES = {1: "Oberbauchlaengsschnitt_links.png",
                   2: "Oberbauchlaengsschnitt_MCL_rechts.png",
                   3: "Subcostaler_Schraegschnitt_rechts.png",
                   4: "Intercostalschnitt_rechts.png",
                   5: "Hoher_Flankenschnitt_rechts.png"}

PICTOGRAM_WIDTH = 0.20   # pictogram width as fraction of the image width
MARGIN = 0.02            # distance to the right / bottom edge as fraction of the image width

# -----------------------------------------------


def load_pictograms(pictogram_dir, explicit_files=None):
    """Return {view number: RGBA image} for views 1-5."""
    files = {}
    if explicit_files:
        files = {k: os.path.join(pictogram_dir, v) for k, v in explicit_files.items()}
    else:
        for fn in sorted(os.listdir(pictogram_dir)):
            if not fn.lower().endswith(".png"):
                continue
            # view number as a separate number in the file name (not part of e.g. "12")
            numbers = [int(n) for n in re.findall(r"(?<!\d)(\d)(?!\d)", os.path.splitext(fn)[0])]
            views = [n for n in numbers if n in CLASSES]
            if len(views) != 1:
                print(f"WARNING: ignoring {fn} (no unique view number 1-5 in the file name)")
                continue
            if views[0] in files:
                raise ValueError(f"Two pictograms for view {views[0]}: "
                                 f"{os.path.basename(files[views[0]])} and {fn}. "
                                 f"Rename one or set PICTOGRAM_FILES.")
            files[views[0]] = os.path.join(pictogram_dir, fn)

    missing = [k for k in CLASSES if k not in files]
    if missing:
        raise FileNotFoundError(f"No pictogram found for view(s) {missing} in {pictogram_dir}")

    # Check that every file exists before loading, and report all problems at once
    not_found = {k: os.path.basename(p) for k, p in files.items() if not os.path.isfile(p)}
    if not_found:
        available = sorted(f for f in os.listdir(pictogram_dir) if f.lower().endswith(".png"))
        msg = [f"Pictogram file(s) not found in {pictogram_dir}:"]
        msg += [f"  view {k}: '{name}'" for k, name in sorted(not_found.items())]
        msg += ["Available PNG files (copy the exact names into PICTOGRAM_FILES):"]
        msg += [f"  '{name}'" for name in available]
        raise FileNotFoundError("\n".join(msg))

    print("Pictogram templates:")
    for k in sorted(files):
        print(f"  view {k}: {os.path.basename(files[k])}")
    return {k: Image.open(p).convert("RGBA") for k, p in files.items()}


def overlay(image, pictogram, width_frac=PICTOGRAM_WIDTH, margin_frac=MARGIN):
    """Paste the pictogram (keeping its transparency) into the bottom-right corner."""
    img = image.convert("RGBA")
    w, h = img.size

    target_w = max(1, int(w * width_frac))
    target_h = max(1, int(pictogram.height * target_w / pictogram.width))
    if target_h > h // 2:  # very wide images: limit the height
        target_h = h // 2
        target_w = max(1, int(pictogram.width * target_h / pictogram.height))
    picto = pictogram.resize((target_w, target_h), Image.LANCZOS)

    margin = int(w * margin_frac)
    x, y = w - target_w - margin, h - target_h - margin
    img.alpha_composite(picto, dest=(max(0, x), max(0, y)))
    return img.convert("RGB")


def mark_misclassified(image, true_view, pred_view):
    """Red frame plus a label with true and predicted view (top-left corner)."""
    img = image.convert("RGB").copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)

    border = max(3, int(w * 0.012))
    draw.rectangle([0, 0, w - 1, h - 1], outline=(220, 0, 0), width=border)

    text = f"MISCLASSIFIED  true: {true_view}  predicted: {pred_view}"
    font_size = max(12, int(w * 0.035))
    try:
        font = ImageFont.load_default(size=font_size)   # Pillow >= 10.1
    except TypeError:
        font = ImageFont.load_default()
    x0, y0 = border + 6, border + 6
    left, top, right, bottom = draw.textbbox((x0, y0), text, font=font)
    draw.rectangle([left - 4, top - 4, right + 4, bottom + 4], fill=(220, 0, 0))
    draw.text((x0, y0), text, fill=(255, 255, 255), font=font)
    return img


@torch.no_grad()
def predict_image(model, image, transform, device):
    """Return (predicted view 1-5, confidence) for one PIL image."""
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    return idx + 1, float(probs[idx])


def main():
    from main import DATA_DIR, IMG_SIZE, RESULTS_DIR

    parser = argparse.ArgumentParser(description="Overlay the predicted pictogram on ultrasound images")
    parser.add_argument("--model", default="efficientnet_b0",
                        choices=["efficientnet_b0", "resnet50", "convnext_tiny"])
    parser.add_argument("--checkpoint", default=None,
                        help="Model weights (default: results/<model>/best_model.pt)")
    parser.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    parser.add_argument("--pictogram-dir", default=PICTOGRAM_DIR)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pictograms = load_pictograms(args.pictogram_dir, PICTOGRAM_FILES)

    class_names = [CLASSES[k] for k in sorted(CLASSES)]
    checkpoint = args.checkpoint or os.path.join(RESULTS_DIR, args.model, "best_model.pt")
    model = build_model(args.model, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device).eval()
    _, eval_tf = get_transforms(IMG_SIZE)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    out_root = os.path.join(RESULTS_DIR, args.model, "pictogram_overlay", args.split)

    rows = []
    for split in splits:
        samples = datasets.ImageFolder(os.path.join(DATA_DIR, split)).samples
        print(f"\n{split}: {len(samples)} images")
        for path, true_idx in samples:
            image = Image.open(path)
            pred_view, conf = predict_image(model, image, eval_tf, device)

            out_dir = os.path.join(out_root, CLASSES[pred_view])
            os.makedirs(out_dir, exist_ok=True)
            out_name = os.path.splitext(os.path.basename(path))[0] + ".png"
            result = overlay(image, pictograms[pred_view])
            result.save(os.path.join(out_dir, out_name))

            # Wrong predictions: additional marked copy in misclassified/
            true_view = true_idx + 1
            if pred_view != true_view:
                wrong_dir = os.path.join(out_root, "misclassified")
                os.makedirs(wrong_dir, exist_ok=True)
                wrong_name = f"true{true_view}_pred{pred_view}_{out_name}"
                mark_misclassified(result, true_view, pred_view).save(os.path.join(wrong_dir, wrong_name))

            rows.append({"file": os.path.basename(path), "split": split,
                         "true_view": true_idx + 1, "predicted_view": pred_view,
                         "confidence": round(conf, 4), "correct": true_idx + 1 == pred_view})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_root, "predictions.csv"), index=False)
    print(f"\n{len(df)} images saved to {out_root}")
    print(f"Accuracy on these images: {df['correct'].mean():.3f} "
          f"({(~df['correct']).sum()} with a wrong pictogram, see predictions.csv)")


if __name__ == "__main__":
    main()