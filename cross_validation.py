"""
Patient-level, class-stratified k-fold cross-validation (default: 5 folds).

Every patient is in the test fold exactly once, so every image gets exactly one
out-of-fold prediction. Within each fold, a patient-level validation set is split
off the remaining patients for early stopping / model selection.

    fold k:  test = fold k (~20 % of patients)
             val  = ~1/6 of the remaining patients (~13 %)
             train = the rest (~67 %)

Output: results/cv_<model>/
- fold_<k>/                         checkpoint, report, confusion matrix, training curves per fold
- cv_metrics_per_fold.csv           accuracy, balanced accuracy, macro F1, macro AUC per fold
- cv_per_class.csv                  per-class precision / recall / F1 (mean and std over folds)
- cv_summary.txt                    mean +/- std over folds (the numbers to report)
- oof_predictions.csv               out-of-fold prediction and probabilities for every image
- classification_report.txt/.csv    pooled over all out-of-fold predictions
- confusion_matrix_normalized.png   pooled over all out-of-fold predictions

Usage:
    python cross_validation.py --model efficientnet_b0
    python cross_validation.py --model resnet50 --folds 5

Note: this trains one model per fold, so it takes about k times as long as main.py.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold

from models import build_model
from preprocess_data import CLASSES, collect_images, get_dataloaders_from_df
from training import train_model
from visualization import plot_confusion_matrix, plot_training_curves, predict, save_classification_report

SUMMARY_METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc_ovr"]


def make_folds(df, n_folds=5, n_val_folds=6, seed=42):
    """Yield (fold index, train_df, val_df, test_df) with strict patient separation."""
    outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (dev_idx, test_idx) in enumerate(outer.split(df, df["label"], groups=df["patient"])):
        dev, test = df.iloc[dev_idx], df.iloc[test_idx]

        inner = StratifiedGroupKFold(n_splits=n_val_folds, shuffle=True, random_state=seed + k)
        train_idx, val_idx = next(inner.split(dev, dev["label"], groups=dev["patient"]))
        train, val = dev.iloc[train_idx], dev.iloc[val_idx]

        # Sanity check: no patient may appear in more than one split
        p_train, p_val, p_test = set(train["patient"]), set(val["patient"]), set(test["patient"])
        assert not (p_train & p_val or p_train & p_test or p_val & p_test), "Patient overlap between splits!"

        yield k, train, val, test


def write_summary(metrics, per_class_agg, n_images, n_patients, model_name, out_dir):
    """Write mean +/- std over folds to cv_summary.txt and print it."""
    lines = [
        f"{len(metrics)}-fold patient-level cross-validation | model: {model_name}",
        f"{n_images} images from {n_patients} patients (each image tested exactly once)",
        "",
        f"{'metric':<20s} {'mean':>7s} {'std':>7s} {'min':>7s} {'max':>7s}",
    ]
    for m in SUMMARY_METRICS:
        v = metrics[m]
        lines.append(f"{m:<20s} {v.mean():7.4f} {v.std():7.4f} {v.min():7.4f} {v.max():7.4f}")

    lines += ["", "Per class (mean +/- std over folds):",
              f"{'class':<42s} {'precision':>17s} {'recall':>17s} {'f1-score':>17s}"]
    for cls, row in per_class_agg.iterrows():
        cells = [f"{row[(m, 'mean')]:.3f} +/- {row[(m, 'std')]:.3f}" for m in ["precision", "recall", "f1-score"]]
        lines.append(f"{cls:<42s} " + " ".join(f"{c:>17s}" for c in cells))

    text = "\n".join(lines)
    with open(os.path.join(out_dir, "cv_summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)


def main():
    # Imported here (not at module level) to keep main.py the single place for configuration
    from main import BATCH_SIZE, EPOCHS, IMG_SIZE, LR, PATIENCE, RESULTS_DIR, SEED, SRC_DIR

    parser = argparse.ArgumentParser(description="Patient-level k-fold cross-validation")
    parser.add_argument("--model", default="efficientnet_b0",
                        choices=["efficientnet_b0", "resnet50", "convnext_tiny"])
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = os.path.join(RESULTS_DIR, f"cv_{args.model}")
    os.makedirs(out_dir, exist_ok=True)

    df = collect_images(SRC_DIR)
    class_names = [CLASSES[k] for k in sorted(CLASSES)]
    n_classes = len(class_names)

    fold_metrics, per_class_rows, oof_parts, oof_probs = [], [], [], []

    for k, train, val, test in make_folds(df, n_folds=args.folds, seed=SEED):
        print(f"\n===== Fold {k + 1}/{args.folds} | images train/val/test: "
              f"{len(train)}/{len(val)}/{len(test)} | patients: "
              f"{train['patient'].nunique()}/{val['patient'].nunique()}/{test['patient'].nunique()} =====")

        fold_dir = os.path.join(out_dir, f"fold_{k + 1}")
        os.makedirs(fold_dir, exist_ok=True)

        loaders, _ = get_dataloaders_from_df({"train": train, "val": val, "test": test},
                                             SRC_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

        torch.manual_seed(SEED + k)
        model = build_model(args.model, num_classes=n_classes, pretrained=True)
        model, history = train_model(model, loaders, device, epochs=EPOCHS, lr=LR, patience=PATIENCE,
                                     checkpoint_path=os.path.join(fold_dir, "best_model.pt"))
        plot_training_curves(history, fold_dir)

        # Test fold (loader is not shuffled -> predictions are in the order of `test`)
        y_true, y_pred, probs = predict(model, loaders["test"], device)
        summary = save_classification_report(y_true, y_pred, probs, class_names, fold_dir)
        plot_confusion_matrix(y_true, y_pred, class_names, fold_dir,
                              title=f"Normalized confusion matrix (fold {k + 1}, test set)")
        fold_metrics.append({"fold": k + 1, **summary})

        report = classification_report(y_true, y_pred, labels=range(n_classes), target_names=class_names,
                                       output_dict=True, zero_division=0)
        for cls in class_names:
            per_class_rows.append({"fold": k + 1, "class": cls,
                                   **{m: report[cls][m] for m in ["precision", "recall", "f1-score"]}})

        part = pd.DataFrame({
            "file": test["file"].values,
            "patient": test["patient"].values,
            "fold": k + 1,
            "true": y_true,
            "pred": y_pred,
            "correct": y_true == y_pred,
            "confidence": probs.max(axis=1),
        })
        for i in range(n_classes):
            part[f"p_class{i + 1}"] = probs[:, i]
        oof_parts.append(part)
        oof_probs.append(probs)

    # ---- Aggregate over folds ----
    metrics = pd.DataFrame(fold_metrics)
    metrics.to_csv(os.path.join(out_dir, "cv_metrics_per_fold.csv"), index=False)

    per_class = pd.DataFrame(per_class_rows)
    per_class_agg = per_class.groupby("class")[["precision", "recall", "f1-score"]].agg(["mean", "std"])
    per_class_agg = per_class_agg.reindex(class_names)
    per_class_agg.to_csv(os.path.join(out_dir, "cv_per_class.csv"))

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)

    # Pooled report and confusion matrix over all out-of-fold predictions
    print("\n===== Pooled out-of-fold results =====")
    save_classification_report(oof["true"].values, oof["pred"].values, np.concatenate(oof_probs),
                               class_names, out_dir)
    plot_confusion_matrix(oof["true"].values, oof["pred"].values, class_names, out_dir,
                          title=f"Normalized confusion matrix ({args.folds}-fold CV, all test folds pooled)")

    write_summary(metrics, per_class_agg, len(df), df["patient"].nunique(), args.model, out_dir)
    print(f"\nAll cross-validation results saved to {out_dir}")


if __name__ == "__main__":
    main()