"""
Data preprocessing for the liver ultrasound view classification.

- Parses patient ID and view label (last digit before the file extension) from file names
- Splits the data into train / val / test at PATIENT level
  (all images of one patient end up in the same split -> no data leakage)
- Stratifies by view (1-5) so every class is evenly represented in each split
- Copies the images into an ImageFolder structure: DST/<split>/<class>/<image>
- Provides the image transforms and DataLoaders used for training and evaluation
"""
import os
import re
import shutil

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Label digit -> class folder name
CLASSES = {
    1: "Oberbauchlängsschnitt links",
    2: "Oberbauchlängsschnitt MCL rechts",
    3: "Subcostaler Schrägschnitt",
    4: "Intercostalschnitt",
    5: "Hoher Flankenschnit",
}

# ImageNet statistics (the pretrained backbones expect these)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def collect_images(src_dir):
    """Read all image files and extract patient ID and view label."""
    rows, skipped = [], []
    for fn in sorted(os.listdir(src_dir)):
        if not fn.lower().endswith(IMAGE_EXTENSIONS):
            continue
        stem = os.path.splitext(fn)[0]
        patient = re.search(r"Pat\d+", stem)
        label = re.search(r"(\d)$", stem)  # last digit before the extension
        if not patient or not label or int(label.group(1)) not in CLASSES:
            skipped.append(fn)
            continue
        rows.append({"file": fn, "patient": patient.group(), "label": int(label.group(1))})

    df = pd.DataFrame(rows)
    print(f"Found {len(df)} images from {df['patient'].nunique()} patients")
    if skipped:
        print(f"WARNING: skipped {len(skipped)} files (no patient ID / label found), e.g. {skipped[:5]}")
    return df


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------
def split_by_patient(df, seed=42, n_folds=20, test_folds=(0, 1, 2), val_folds=(3, 4, 5)):
    """
    Patient-level, class-stratified split.
    With 20 folds of 5 % each: 3 folds test (15 %), 3 folds val (15 %), rest train (70 %).
    """
    df = df.copy()
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df["split"] = "train"
    for fold, (_, idx) in enumerate(sgkf.split(df, df["label"], groups=df["patient"])):
        if fold in test_folds:
            df.loc[df.index[idx], "split"] = "test"
        elif fold in val_folds:
            df.loc[df.index[idx], "split"] = "val"

    # Sanity check: no patient may appear in more than one split
    assert (df.groupby("patient")["split"].nunique() == 1).all(), "Patient found in multiple splits!"

    print("\nImages per split and class:")
    print(pd.crosstab(df["split"], df["label"], margins=True))
    print("\nPatients per split:")
    print(df.groupby("split")["patient"].nunique())
    return df


def copy_to_imagefolder(df, src_dir, dst_dir):
    """Copy images into DST/<split>/<class>/ and save the split overview as CSV."""
    for _, row in df.iterrows():
        target = os.path.join(dst_dir, row["split"], CLASSES[row["label"]])
        os.makedirs(target, exist_ok=True)
        shutil.copy2(os.path.join(src_dir, row["file"]), os.path.join(target, row["file"]))

    df.to_csv(os.path.join(dst_dir, "split_overview.csv"), index=False)
    print(f"\nData written to {dst_dir} (assignment in split_overview.csv)")


def prepare_dataset(src_dir, dst_dir, seed=42):
    """Full preprocessing pipeline: parse -> split -> copy."""
    df = collect_images(src_dir)
    df = split_by_patient(df, seed=seed)
    copy_to_imagefolder(df, src_dir, dst_dir)
    return df


# ---------------------------------------------------------------------------
# Transforms and DataLoaders
# ---------------------------------------------------------------------------
def get_transforms(img_size=224):
    """
    Train transforms with light augmentation, eval transforms without.
    IMPORTANT: no horizontal flipping - left/right is class-relevant here!
    """
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # ultrasound is grayscale, backbones expect 3 channels
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def get_dataloaders(data_dir, img_size=224, batch_size=32, num_workers=4):
    """Create ImageFolder datasets and DataLoaders for train / val / test."""
    train_tf, eval_tf = get_transforms(img_size)
    datasets_ = {
        split: datasets.ImageFolder(os.path.join(data_dir, split), train_tf if split == "train" else eval_tf)
        for split in ["train", "val", "test"]
    }
    loaders = {
        split: DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers)
        for split, ds in datasets_.items()
    }
    class_names = datasets_["train"].classes
    return loaders, class_names


# ---------------------------------------------------------------------------
# DataLoaders directly from a DataFrame (used by cross_validation.py)
# ---------------------------------------------------------------------------
class DataFrameDataset(Dataset):
    """
    Dataset that reads images listed in a DataFrame (columns: file, label) from src_dir.
    Labels 1-5 are mapped to class indices 0-4, which matches the alphabetical
    ImageFolder order of the CLASSES folder names.
    """

    def __init__(self, df, src_dir, transform):
        self.samples = [(os.path.join(src_dir, f), int(label) - 1)
                        for f, label in zip(df["file"], df["label"])]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def get_dataloaders_from_df(split_dfs, src_dir, img_size=224, batch_size=32, num_workers=4):
    """
    split_dfs: {"train": df, "val": df, "test": df}
    Returns DataLoaders (train shuffled, val/test in DataFrame order) and the class names.
    """
    train_tf, eval_tf = get_transforms(img_size)
    loaders = {}
    for split, df in split_dfs.items():
        ds = DataFrameDataset(df, src_dir, train_tf if split == "train" else eval_tf)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                    num_workers=num_workers)
    class_names = [CLASSES[k] for k in sorted(CLASSES)]
    return loaders, class_names