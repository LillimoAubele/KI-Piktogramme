"""
Liver ultrasound view classification (5 probe positions).

Usage:
    python main.py                          # split data, train, evaluate on test set
    python main.py --skip-split             # reuse an existing split
    python main.py --model resnet50         # try another backbone

All outputs (checkpoint, reports, plots, Grad-CAM) are written to results/<model>/.
"""
import argparse
import os
import shutil

import torch

from evaluation import report
from models import build_model
from preprocess_data import get_dataloaders, prepare_dataset
from training import train_model
from visualization import run_visualization

# ---------------- Configuration ----------------
SRC_DIR = "/home/xrad/KI_Piktogramm/github/KI-Piktogramme/data/anonymisiert/Pat_mit_Piktogramm_cut"
DATA_DIR = "/home/xrad/KI_Piktogramm/github/KI-Piktogramme/data/split_views"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40
LR = 3e-4
PATIENCE = 8
# -----------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Liver ultrasound view classification")
    parser.add_argument("--model", default="efficientnet_b0",
                        choices=["efficientnet_b0", "resnet50", "convnext_tiny"])
    parser.add_argument("--skip-split", action="store_true",
                        help="Do not re-create the train/val/test split")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = os.path.join(RESULTS_DIR, args.model)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Preprocessing + patient-level split
    if not args.skip_split:
        if os.path.exists(DATA_DIR):
            # Safety check: never delete the folder with the original images
            if os.path.abspath(DATA_DIR) == os.path.abspath(SRC_DIR):
                raise ValueError("DATA_DIR must not be the same folder as SRC_DIR!")
            print(f"Removing existing split in {DATA_DIR} and creating a new one")
            shutil.rmtree(DATA_DIR)
        prepare_dataset(SRC_DIR, DATA_DIR, seed=SEED)

    loaders, class_names = get_dataloaders(DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    print(f"Classes: {class_names}")

    # 2) Model
    model = build_model(args.model, num_classes=len(class_names), pretrained=True)

    # 3) Training (model selection on the validation set only)
    model, history = train_model(model, loaders, device, epochs=EPOCHS, lr=LR, patience=PATIENCE,
                                 checkpoint_path=os.path.join(out_dir, "best_model.pt"))

    # 4) Final evaluation on the held-out test set
    print("\n=== Test set ===")
    report(model, loaders["test"], device, class_names)

    # 5) Reports, plots and Grad-CAM -> results/<model>/
    run_visualization(model, args.model, loaders, class_names, device, out_dir, history=history)


if __name__ == "__main__":
    main()