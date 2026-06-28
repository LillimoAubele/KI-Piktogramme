import numpy as np

from sklearn.model_selection import StratifiedKFold

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import (
    DataLoader,
    Subset
)


# =====================================================
# MAIN FUNCTION
# =====================================================

def get_data_folds(
    data_dir,
    batch_size,
    num_folds=5
):

    # -------------------------------------------------
    # TRANSFORMS
    # -------------------------------------------------

    train_transform = transforms.Compose([

        transforms.Resize((224,224)),

        transforms.RandomRotation(10),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[0.485,0.456,0.406],

            std=[0.229,0.224,0.225]
        )
    ])

    val_transform = transforms.Compose([

        transforms.Resize((224,224)),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[0.485,0.456,0.406],

            std=[0.229,0.224,0.225]
        )
    ])

    # -------------------------------------------------
    # BASE DATASET
    # -------------------------------------------------

    base_dataset = datasets.ImageFolder(
        root=data_dir
    )

    labels = np.array([
        label
        for _, label
        in base_dataset.samples
    ])

    print(
        "Class distribution:",
        np.unique(
            labels,
            return_counts=True
        )
    )

    print(
        "Class mapping:",
        base_dataset.class_to_idx
    )

    # -------------------------------------------------
    # STRATIFIED KFOLD
    # -------------------------------------------------

    skf = StratifiedKFold(

        n_splits=num_folds,

        shuffle=True,

        random_state=42
    )

    indices = np.arange(
        len(labels)
    )

    folds = []

    for fold_idx, (
        train_idx,
        val_idx
    ) in enumerate(
        skf.split(indices, labels),
        start=1
    ):

        train_dataset = datasets.ImageFolder(
            root=data_dir,
            transform=train_transform
        )

        val_dataset = datasets.ImageFolder(
            root=data_dir,
            transform=val_transform
        )

        train_subset = Subset(
            train_dataset,
            train_idx
        )

        val_subset = Subset(
            val_dataset,
            val_idx
        )

        train_loader = DataLoader(

            train_subset,

            batch_size=batch_size,

            shuffle=True,

            num_workers=4
        )

        val_loader = DataLoader(

            val_subset,

            batch_size=batch_size,

            shuffle=False,

            num_workers=4
        )

        print(
            f"Fold {fold_idx}: "
            f"{len(train_idx)} train / "
            f"{len(val_idx)} val"
        )

        folds.append(

            (
                train_loader,
                val_loader,
                5
            )
        )

    return folds, base_dataset