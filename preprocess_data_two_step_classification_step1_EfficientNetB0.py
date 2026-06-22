import numpy as np
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import random



# -------------------------------------------------
# MERGE MAP (5 → 4 Klassen)
# -------------------------------------------------
STEP1_MAP = {
    0: 0,  # Flanke
    1: 0,  # Flanke
    2: 1,  # Oberbauch
    3: 1,
    4: 1
}


# -------------------------------------------------
# Dataset Step 1
# -------------------------------------------------


class Step1Dataset(Dataset):
    def __init__(self, dataset, indices, mapping, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.mapping = mapping
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        path, label = self.dataset.samples[real_idx]
        img = self.dataset.loader(path)

        if self.transform:
            img = self.transform(img)

        label = self.mapping[int(label)]
        return img, label

# -------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------
def get_data_folds(data_dir, batch_size, num_folds=5):

    # ---------------------------
    # TRANSFORMS
    # ---------------------------
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.RandomRotation(10),

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]

            # mean=[0.1140, 0.1193, 0.1326],
            # std=[0.1105, 0.1164, 0.1293]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(
          
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]

            # mean=[0.1140, 0.1193, 0.1326],
            # std=[0.1105, 0.1164, 0.1293]
        )
    ])

    # ---------------------------
    # BASE DATASET (ohne Transform)
    # ---------------------------
    base_dataset = datasets.ImageFolder(root=data_dir)

    # OPTIONAL DEBUG
    print("UNIQUE ORIGINAL LABELS:",
          np.unique([label for _, label in base_dataset.samples]))
    
    labels = [STEP1_MAP[label] for _, label in base_dataset.samples]
    
    print("STEP1 label distribution:", np.unique(labels, return_counts=True))

 
    labels = [STEP1_MAP[label] for _, label in base_dataset.samples]

    print("Merged label distribution:", np.unique(labels, return_counts=True))

    # ---------------------------
    # STRATIFIED KFOLD
    # ---------------------------
    skf = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=42
    )
    
    labels = np.array(labels)
    indices = np.arange(len(labels))
    
    folds = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels), 1):
    
        train_dataset = Step1Dataset(
            base_dataset,
            train_idx,
            STEP1_MAP,
            transform=train_transform
        )
        
        val_dataset = Step1Dataset(
            base_dataset,
            val_idx,
            STEP1_MAP,
            transform=val_transform
        )
      

        sample_indices = random.sample(range(len(train_dataset)), min(50, len(train_dataset)))
        sample_labels = [train_dataset[i][1] for i in sample_indices]

        print(f"Fold {fold_idx} sample labels:", np.unique(sample_labels))
    
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    folds.append((train_loader, val_loader, 2))
    
    return folds, base_dataset