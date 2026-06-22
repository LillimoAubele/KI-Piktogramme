import numpy as np
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import random



# -------------------------------------------------
# MERGE MAP 
# -------------------------------------------------
STEP2B_MAP = {
    2: 0,  # Oberbauchlaengsschnitt MCL rechts
    3: 1,  # Oberbauchlaengsschnitt links
    4: 2   # Subcostaler Schraegschnitt rechts
}

# -------------------------------------------------
# Dataset Step 2b
# -------------------------------------------------

class Step2bDataset(Dataset):

    def __init__(self, dataset, indices, transform=None):

        self.dataset = dataset
        self.transform = transform

        self.indices = [
            i for i in indices
            if dataset.samples[i][1] in [2,3,4]
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        real_idx = self.indices[idx]

        path, label = self.dataset.samples[real_idx]

        img = self.dataset.loader(path)

        if self.transform:
            img = self.transform(img)

        label = STEP2B_MAP[label]

        return img, label

# -------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------

def get_data_folds(data_dir, batch_size, num_folds=5):

    train_transform = transforms.Compose([

        transforms.Resize((224,224)),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(5),

        transforms.ColorJitter(
            brightness=0.05,
            contrast=0.05
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([

        transforms.Resize((224,224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    base_dataset = datasets.ImageFolder(root=data_dir)

    labels = np.array([
        label for _, label in base_dataset.samples
        if label in [2,3,4]
    ])

    valid_indices = np.array([
        i for i, (_, label) in enumerate(base_dataset.samples)
        if label in [2,3,4]
    ])

    skf = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=42
    )

    folds = []

    for fold_idx, (train_split, val_split) in enumerate(
        skf.split(valid_indices, labels), 1
    ):

        train_idx = valid_indices[train_split]
        val_idx = valid_indices[val_split]

        train_dataset = Step2bDataset(
            base_dataset,
            train_idx,
            transform=train_transform
        )

        val_dataset = Step2bDataset(
            base_dataset,
            val_idx,
            transform=val_transform
        )

        # DEBUG
        all_labels = [
            train_dataset[i][1]
            for i in range(len(train_dataset))
        ]

        print(
            f"Fold {fold_idx} labels:",
            np.unique(all_labels, return_counts=True)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        folds.append((train_loader, val_loader, 3))

    return folds, base_dataset