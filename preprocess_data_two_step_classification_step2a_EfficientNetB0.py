import numpy as np
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import random



# -------------------------------------------------
# MERGE MAP 
# -------------------------------------------------
STEP2A_MAP = {
    0: 0,  # Hoher Flankenschnitt
    1: 1   # Intercostalschnitt
}


# -------------------------------------------------
# Dataset Step 2a
# -------------------------------------------------

class Step2aDataset(Dataset):

    def __init__(self, dataset, indices, transform=None):

        self.dataset = dataset
        self.transform = transform

     
        self.indices = [
            i for i in indices
            if dataset.samples[i][1] in [0, 1]
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        real_idx = self.indices[idx]

        path, label = self.dataset.samples[real_idx]

        img = self.dataset.loader(path)

        if self.transform:
            img = self.transform(img)

        label = STEP2A_MAP[label]

        return img, label

# -------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------

def get_data_folds(data_dir, batch_size, num_folds=5):

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),

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
        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    base_dataset = datasets.ImageFolder(root=data_dir)

    labels = np.array([
        label for _, label in base_dataset.samples
    ])

    indices = np.arange(len(labels))

    skf = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=42
    )

    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels), 1):

        train_dataset = Step2aDataset(
            base_dataset,
            train_idx,
            transform=train_transform
        )

        val_dataset = Step2aDataset(
            base_dataset,
            val_idx,
            transform=val_transform
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

        folds.append((train_loader, val_loader, 2))

    return folds, base_dataset