"""
Label noise test: how many wrong labels can the training data contain before
the classification quality really suffers?

A fraction of the TRAINING and VALIDATION labels is deliberately corrupted, a new
model is trained on these noisy labels, and it is evaluated on the CLEAN test set
(same patient-level split as main.py). Validation labels are corrupted as well,
because in practice the validation data would contain the same annotation errors.

Two kinds of label errors:
    symmetric   wrong label = any other view, chosen at random
    adjacent    wrong label = an anatomically neighbouring view (realistic confusions):
                1 -> 2 | 2 -> 1 or 3 | 3 -> 2 or 4 | 4 -> 3 or 5 | 5 -> 4

Noise rates (default): 0, 5, 10, 20, 30, 40, 50 % of the images, each with several
seeds (default 3: different selection of corrupted images and different training run).
Rate 0 % is the clean reference and is trained once per seed, shared by both noise types.

Additionally measured for every run ("label error recovery"): for the training images
whose label was corrupted, how often does the trained model predict the TRUE view
instead of the wrong label it was trained with? A high value means the model is not
simply memorising errors - and that its disagreements with the labels can be used to
find mislabelled images.

Output: results/label_noise[_noiseaug]/
- label_noise_results.csv   one row per run (written after every run)
- label_noise_summary.csv   mean / std over seeds per noise type and rate
- label_noise_curves.png    test balanced accuracy vs. noise rate (+ recovery)
- label_noise_per_class.png test recall per view vs. noise rate
- summary.txt               up to which noise rate the result stays within 5 / 10 pp

The run can be interrupted: on restart, runs already in label_noise_results.csv are skipped.

Usage:
    python label_noise_test.py --noise-aug                       # like the main model (recommended)
    python label_noise_test.py --noise-aug --seeds 1             # quick overview, ~1/3 of the time
    python label_noise_test.py --noise-aug --rates 0 10 30 50 --types adjacent
    python label_noise_test.py --noise-aug --summary-only        # only re-create plots / summary

Runtime: one run takes about as long as main.py. Default settings = 3 + 2 x 6 x 3 = 39 runs.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_RATES = [0, 5, 10, 20, 30, 40, 50]   # percent
NOISE_TYPES = ["symmetric", "adjacent"]
TOLERANCES = (0.05, 0.10)
CHANCE = 0.2

# Anatomically neighbouring views (used for "adjacent" noise)
NEIGHBOURS = {1: [2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]}
VIEWS = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Label corruption
# ---------------------------------------------------------------------------
def corrupt_labels(labels, rate, noise_type, rng):
    """
    Return (noisy labels, boolean mask of corrupted images).
    Exactly round(rate * n) images get a wrong label; the label always changes.
    """
    labels = np.asarray(labels)
    noisy = labels.copy()
    n_flip = int(round(rate * len(labels)))
    flip = np.zeros(len(labels), dtype=bool)
    if n_flip == 0:
        return noisy, flip
    idx = rng.choice(len(labels), size=n_flip, replace=False)
    flip[idx] = True
    for i in idx:
        options = NEIGHBOURS[labels[i]] if noise_type == "adjacent" else [v for v in VIEWS if v != labels[i]]
        noisy[i] = rng.choice(options)
    return noisy, flip


def run_rng(seed, rate_pct, noise_type):
    """Reproducible random generator per run."""
    return np.random.default_rng([seed, int(rate_pct), NOISE_TYPES.index(noise_type) if rate_pct else 0])


def run_key(noise_type, rate_pct, seed):
    # rate 0 is identical for both noise types -> one shared reference run per seed
    return ("clean" if rate_pct == 0 else noise_type, int(rate_pct), int(seed))


# ---------------------------------------------------------------------------
# Summary, plots
# ---------------------------------------------------------------------------
def expand_clean(res):
    """Copy the shared 0 % runs into both noise types (for plotting / summary)."""
    clean = res[res["noise_type"] == "clean"]
    parts = [res[res["noise_type"] != "clean"]]
    for t in NOISE_TYPES:
        parts.append(clean.assign(noise_type=t))
    return pd.concat(parts, ignore_index=True)


def summarise(res):
    cols = ["test_accuracy", "test_balanced_accuracy", "test_macro_f1", "recovery_rate", "best_val_acc"] + \
           [c for c in res.columns if c.startswith("recall_view")]
    agg = expand_clean(res).groupby(["noise_type", "rate_pct"])[cols].agg(["mean", "std", "count"])
    return agg


def plot_curves(res, out_dir):
    d = expand_clean(res)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    clean = d[d["rate_pct"] == 0]["test_balanced_accuracy"].mean()
    for t, color in zip(NOISE_TYPES, ["tab:blue", "tab:orange"]):
        g = d[d["noise_type"] == t].groupby("rate_pct")
        m, s = g["test_balanced_accuracy"].mean(), g["test_balanced_accuracy"].std().fillna(0)
        ax1.errorbar(m.index, m.values, yerr=s.values, marker="o", capsize=3, color=color, label=t)
        rec = d[(d["noise_type"] == t) & (d["rate_pct"] > 0)].groupby("rate_pct")["recovery_rate"]
        ax2.errorbar(rec.mean().index, rec.mean().values, yerr=rec.std().fillna(0).values,
                     marker="o", capsize=3, color=color, label=t)
    ax1.axhline(clean - TOLERANCES[0], color="gray", linestyle=":", linewidth=1, label="clean - 5 pp")
    ax1.axhline(CHANCE, color="gray", linestyle="--", linewidth=0.8, label="chance")
    ax1.set_ylim(0, 1.03)
    ax1.set_xlabel("corrupted training / validation labels (%)")
    ax1.set_ylabel("balanced accuracy on the clean test set")
    ax1.set_title("Test performance vs. label noise (mean +/- std over seeds)")
    ax1.legend(fontsize=8, loc="lower left")

    ax2.set_ylim(0, 1.03)
    ax2.set_xlabel("corrupted training / validation labels (%)")
    ax2.set_ylabel("fraction predicted as TRUE view")
    ax2.set_title("Label error recovery on corrupted training images")
    ax2.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "label_noise_curves.png"), dpi=200)
    plt.close(fig)


def plot_per_class(res, class_labels, out_dir):
    d = expand_clean(res)
    fig, axes = plt.subplots(1, len(NOISE_TYPES), figsize=(6 * len(NOISE_TYPES), 4.5), squeeze=False)
    for ax, t in zip(axes[0], NOISE_TYPES):
        g = d[d["noise_type"] == t].groupby("rate_pct")
        for c, label in enumerate(class_labels):
            m = g[f"recall_view{c + 1}"].mean()
            ax.plot(m.index, m.values, marker="o", markersize=4, label=label)
        ax.set_ylim(0, 1.03)
        ax.set_title(f"Recall per view - {t} noise")
        ax.set_xlabel("corrupted training / validation labels (%)")
        ax.set_ylabel("recall on the clean test set")
        ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "label_noise_per_class.png"), dpi=200)
    plt.close(fig)


def write_summary(res, out_dir):
    d = expand_clean(res)
    clean_runs = d[(d["noise_type"] == NOISE_TYPES[0]) & (d["rate_pct"] == 0)]
    clean = clean_runs["test_balanced_accuracy"].mean()
    lines = [
        "Label noise summary (clean test set, patient-level split of main.py)",
        "====================================================================",
        f"Clean reference (0 % noise): balanced accuracy {clean:.3f} "
        f"(std {clean_runs['test_balanced_accuracy'].std():.3f}, {len(clean_runs)} seed(s))",
        "",
        f"{'noise type':<11s} {'rate':>5s} {'bal. acc':>9s} {'std':>6s} {'drop':>7s} {'recovery':>9s} {'seeds':>6s}",
    ]
    for t in NOISE_TYPES:
        g = d[d["noise_type"] == t].groupby("rate_pct")
        for rate, grp in g:
            m, s = grp["test_balanced_accuracy"].mean(), grp["test_balanced_accuracy"].std()
            rec = grp["recovery_rate"].mean() if rate > 0 else float("nan")
            lines.append(f"{t:<11s} {rate:>4d}% {m:>9.3f} {0 if np.isnan(s) else s:>6.3f} "
                         f"{(m - clean) * 100:>+6.1f}pp {rec:>9.3f} {len(grp):>6d}")
        lines.append("")

    lines.append("Highest tested noise rate up to which the mean balanced accuracy stays within ...")
    for t in NOISE_TYPES:
        means = d[d["noise_type"] == t].groupby("rate_pct")["test_balanced_accuracy"].mean().sort_index()
        cells = []
        for tol in TOLERANCES:
            ok = 0
            for rate, m in means.items():
                if m >= clean - tol:
                    ok = rate
                else:
                    break
            plus = "+" if ok == means.index.max() else ""
            cells.append(f"{int(tol * 100)} pp: {ok}%{plus}")
        lines.append(f"  {t:<10s} " + " | ".join(cells))
    lines += ["  ('+' = not exceeded within the tested rates)",
              "",
              "Interpretation:",
              "- 'recovery' = share of corrupted TRAINING images that the final model still assigns to",
              "  their true view. Values near 1 mean the model does not memorise label errors; its",
              "  disagreements with the labels can then be used to find mislabelled images.",
              "- Adjacent noise is the harder, more realistic case: at 50 % the labels of neighbouring",
              "  views (e.g. view 1 vs. 2) become indistinguishable, so a collapse there is expected.",
              "- Differences smaller than the std over seeds are within random variation."]
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)


def make_reports(res, out_dir, class_labels):
    summarise(res).round(4).to_csv(os.path.join(out_dir, "label_noise_summary.csv"))
    plot_curves(res, out_dir)
    plot_per_class(res, class_labels, out_dir)
    write_summary(res, out_dir)


# ---------------------------------------------------------------------------
def main():
    from main import BATCH_SIZE, DATA_DIR, EPOCHS, IMG_SIZE, LR, PATIENCE, RESULTS_DIR, SRC_DIR

    parser = argparse.ArgumentParser(description="Robustness of training to wrong labels")
    parser.add_argument("--model", default="efficientnet_b0",
                        choices=["efficientnet_b0", "resnet50", "convnext_tiny"])
    parser.add_argument("--noise-aug", action="store_true",
                        help="Train with the image noise augmentation, like the main model")
    parser.add_argument("--rates", nargs="+", type=int, default=DEFAULT_RATES,
                        help="Noise rates in percent (0 = clean reference)")
    parser.add_argument("--types", nargs="+", default=NOISE_TYPES, choices=NOISE_TYPES)
    parser.add_argument("--seeds", type=int, default=3, help="Repetitions per setting")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only re-create plots and summary from existing results")
    args = parser.parse_args()

    out_dir = os.path.join(RESULTS_DIR, f"label_noise{'_noiseaug' if args.noise_aug else ''}")
    os.makedirs(out_dir, exist_ok=True)
    results_csv = os.path.join(out_dir, "label_noise_results.csv")

    from preprocess_data import CLASSES
    from visualization import short_names
    class_labels = short_names([CLASSES[k] for k in sorted(CLASSES)])

    if args.summary_only:
        make_reports(pd.read_csv(results_csv), out_dir, class_labels)
        return

    import torch
    from models import build_model
    from preprocess_data import get_dataloaders_from_df
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from training import train_model
    from visualization import predict

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = pd.read_csv(os.path.join(DATA_DIR, "split_overview.csv"))
    train_df = split[split["split"] == "train"].reset_index(drop=True)
    val_df = split[split["split"] == "val"].reset_index(drop=True)
    test_df = split[split["split"] == "test"].reset_index(drop=True)
    print(f"Device: {device} | train/val/test images: {len(train_df)}/{len(val_df)}/{len(test_df)} "
          f"| noise augmentation: {'on' if args.noise_aug else 'off'}")

    done = pd.read_csv(results_csv) if os.path.exists(results_csv) else pd.DataFrame()
    done_keys = {(r.noise_type, int(r.rate_pct), int(r.seed)) for r in done.itertuples()} if len(done) else set()

    plan = []
    for seed in range(args.seeds):
        for rate in sorted(set(args.rates)):
            for t in (args.types if rate > 0 else [args.types[0]]):
                key = run_key(t, rate, seed)
                if key not in done_keys and key not in plan:
                    plan.append(key)
    print(f"{len(plan)} runs to do ({len(done_keys)} already done)")

    tmp_ckpt = os.path.join(out_dir, "_tmp_model.pt")
    for i, (t, rate, seed) in enumerate(plan, 1):
        print(f"\n===== Run {i}/{len(plan)}: {t} noise, {rate} %, seed {seed} =====")
        noise_type = NOISE_TYPES[0] if t == "clean" else t
        rng = run_rng(seed, rate, noise_type)
        tr, va = train_df.copy(), val_df.copy()
        tr["label"], tr_flip = corrupt_labels(train_df["label"].values, rate / 100, noise_type, rng)
        va["label"], _ = corrupt_labels(val_df["label"].values, rate / 100, noise_type, rng)
        print(f"corrupted labels: train {tr_flip.sum()}/{len(tr)}, val {(va['label'] != val_df['label']).sum()}/{len(va)}")

        loaders, _ = get_dataloaders_from_df({"train": tr, "val": va, "test": test_df}, SRC_DIR,
                                             img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                             noise_aug=args.noise_aug)
        torch.manual_seed(1000 * seed + rate)
        model = build_model(args.model, num_classes=len(CLASSES), pretrained=True)
        model, history = train_model(model, loaders, device, epochs=EPOCHS, lr=LR, patience=PATIENCE,
                                     checkpoint_path=tmp_ckpt)

        # clean test set
        y_true, y_pred, _ = predict(model, loaders["test"], device)
        row = {"noise_type": t, "rate_pct": rate, "seed": seed,
               "n_corrupted_train": int(tr_flip.sum()),
               "test_accuracy": accuracy_score(y_true, y_pred),
               "test_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
               "test_macro_f1": f1_score(y_true, y_pred, average="macro"),
               "best_val_acc": max(h["val_acc"] for h in history),
               "epochs": len(history)}
        for c in range(len(CLASSES)):
            sel = y_true == c
            row[f"recall_view{c + 1}"] = float((y_pred[sel] == c).mean()) if sel.any() else np.nan

        # recovery: corrupted training images, evaluated without augmentation, vs. their TRUE label
        if tr_flip.any():
            eval_loaders, _ = get_dataloaders_from_df({"x": train_df[tr_flip]}, SRC_DIR,
                                                      img_size=IMG_SIZE, batch_size=BATCH_SIZE)
            yt, yp, _ = predict(model, eval_loaders["x"], device)  # yt = true labels (0-4)
            row["recovery_rate"] = float((yp == yt).mean())
            row["memorised_rate"] = float((yp == tr.loc[tr_flip, "label"].values - 1).mean())
        else:
            row["recovery_rate"] = row["memorised_rate"] = np.nan

        print(f"test balanced acc {row['test_balanced_accuracy']:.3f} | recovery {row['recovery_rate']:.3f}")
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.round(4).to_csv(results_csv, index=False)  # save after every run

    if os.path.exists(tmp_ckpt):
        os.remove(tmp_ckpt)
    make_reports(done, out_dir, class_labels)
    print(f"\nAll label noise results saved to {out_dir}")


if __name__ == "__main__":
    main()