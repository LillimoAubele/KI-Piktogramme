import numpy as np

from sklearn.model_selection import StratifiedKFold

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset
)


# =====================================================
# MERGE
# =====================================================

MERGE_MAP = {

    0: 0,   # Hoher Flankenschnitt
    1: 0,   # Intercostal

    2: 1,   # MCL
    3: 2,   # Links
    4: 3    # Subcostal
}


class FourClassDataset(Dataset):

    def __init__(
        self,
        dataset,
        indices,
        transform=None
    ):

        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        idx
    ):

        real_idx = self.indices[idx]

        path, label = self.dataset.samples[
            real_idx
        ]

        image = self.dataset.loader(path)

        if self.transform:

            image = self.transform(
                image
            )

        label = MERGE_MAP[
            int(label)
        ]

        return image, label


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

        MERGE_MAP[label]
    
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

        train_dataset = FourClassDataset(
            base_dataset,
            train_idx,
            transform=train_transform
        )
        
        val_dataset = FourClassDataset(
            base_dataset,
            val_idx,
            transform=val_transform
        )



        train_loader = DataLoader(

        train_dataset,
    
        batch_size=batch_size,
    
        shuffle=True,
    
        num_workers=4
        )
        
        val_loader = DataLoader(
        
            val_dataset,
        
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
                4
            )
        )

    return folds, base_dataset