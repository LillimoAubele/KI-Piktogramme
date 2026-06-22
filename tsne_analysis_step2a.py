import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE

from models import build_efficientnet_b0


# -------------------------------------------------
# SETTINGS
# -------------------------------------------------

data_dir = "/home/xrad/KI_Piktogramm/data/Tuebingen/Training/data_sorted_by_classes"

model_path = "/home/xrad/KI_Piktogramm/trained_models_two_step_classification/EfficientNetB0/step2a/EfficientNetB0.pth"

save_path = "/home/xrad/KI_Piktogramm/results/EfficientNetB0/step2a/tsne/tsne_step2a.png"

batch_size = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# STEP 2a CLASSES
# -------------------------------------------------

MERGE_MAP = {
    0: 0,  # Hoher Flankenschnitt rechts
    1: 1   # Intercostalschnitt rechts
}

CLASS_NAMES = [
    "Hoher Flankenschnitt rechts",
    "Intercostalschnitt rechts"
]


# -------------------------------------------------
# TRANSFORMS
# -------------------------------------------------

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------------------------------
# DATASET
# -------------------------------------------------

base_dataset = datasets.ImageFolder(
    root=data_dir,
    transform=transform
)

valid_indices = [
    i for i, (_, label) in enumerate(base_dataset.samples)
    if label in [0,1]
]

subset = torch.utils.data.Subset(
    base_dataset,
    valid_indices
)

loader = DataLoader(
    subset,
    batch_size=batch_size,
    shuffle=False
)

# -------------------------------------------------
# MODEL
# -------------------------------------------------

model = build_efficientnet_b0(
    num_classes=2,
    pretrained=False
)

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model.to(device)

model.eval()


# -------------------------------------------------
# FEATURE EXTRACTOR
# -------------------------------------------------

all_preds = []

feature_extractor = model.features



# -------------------------------------------------
# EXTRACT FEATURES
# -------------------------------------------------

all_features = []

all_labels = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        features = feature_extractor(images)
        
        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())

        features = torch.nn.functional.adaptive_avg_pool2d(
            features,
            1
        )

        features = features.view(features.size(0), -1)

        all_features.append(
            features.cpu().numpy()
        )

        merged_labels = [
            MERGE_MAP[label.item()]
            for label in labels
        ]

        all_labels.extend(merged_labels)


X = np.concatenate(all_features, axis=0)

y = np.array(all_labels)

preds = np.array(all_preds)

print("Feature shape:", X.shape)


# -------------------------------------------------
# t-SNE
# -------------------------------------------------

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

X_tsne = tsne.fit_transform(X)

misclassified = preds != y

# -------------------------------------------------
# PLOT
# -------------------------------------------------

plt.figure(figsize=(10,8))

for class_idx, class_name in enumerate(CLASS_NAMES):

    idx = y == class_idx

    plt.scatter(
    X_tsne[idx, 0],
    X_tsne[idx, 1],
    label=class_name,
    alpha=0.6,
    s=40
    )
    
    mis_idx = idx & misclassified
    
    plt.scatter(
        X_tsne[mis_idx, 0],
        X_tsne[mis_idx, 1],
        facecolors='none',
        edgecolors='red',
        s=120,
        linewidths=2
    )

plt.legend()

plt.title("t-SNE of EfficientNet Features (Step 2a)")

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

plt.tight_layout()


plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"t-SNE plot saved to: {save_path}")

plt.show()
