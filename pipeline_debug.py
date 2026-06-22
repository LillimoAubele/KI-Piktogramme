import os
import re
import torch
import pandas as pd

from PIL import Image
from torchvision import transforms

from models import build_efficientnet_b0


# =====================================================
# PATHS
# =====================================================

TEST_FOLDER = "/home/xrad/KI_Piktogramm/data/Tuebingen/Pat_ohne_Piktogramm"

STEP1_MODEL = "/home/xrad/KI_Piktogramm/trained_models_two_step_classification/EfficientNetB0/step1/EfficientNetB0.pth"

STEP2A_MODEL = "/home/xrad/KI_Piktogramm/trained_models_two_step_classification/EfficientNetB0/step2a/EfficientNetB0.pth"

STEP2B_MODEL = "/home/xrad/KI_Piktogramm/trained_models_two_step_classification/EfficientNetB0/step2b/EfficientNetB0.pth"

OUTPUT_CSV = "/home/xrad/KI_Piktogramm/results/EfficientNetB0/5_classes/results_test_data/pipeline_debug.csv"



# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =====================================================
# LABELS
# =====================================================

CLASS_NAMES = {
    0: "Hoher_Flankenschnitt",
    1: "Intercostal",
    2: "MCL",
    3: "Links",
    4: "Subcostal"
}

FILENAME_TO_CLASSIDX = {
    1: 3,
    2: 2,
    3: 4,
    4: 1,
    5: 0
}


def get_label_from_filename(filename):

    stem = os.path.splitext(filename)[0]

    try:
        idx = int(
            stem.split("_")[-1]
        )
    except:
        return None

    return FILENAME_TO_CLASSIDX.get(
        idx,
        None
    )


# =====================================================
# TRANSFORM
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# =====================================================
# MODELS
# =====================================================

step1_model = build_efficientnet_b0(
    num_classes=2,
    pretrained=False
)

step2a_model = build_efficientnet_b0(
    num_classes=2,
    pretrained=False
)

step2b_model = build_efficientnet_b0(
    num_classes=3,
    pretrained=False
)

step1_model.load_state_dict(
    torch.load(
        STEP1_MODEL,
        map_location=device
    )
)

step2a_model.load_state_dict(
    torch.load(
        STEP2A_MODEL,
        map_location=device
    )
)

step2b_model.load_state_dict(
    torch.load(
        STEP2B_MODEL,
        map_location=device
    )
)

for m in [
    step1_model,
    step2a_model,
    step2b_model
]:
    m.to(device)
    m.eval()


# =====================================================
# EVALUATION
# =====================================================

rows = []

for filename in os.listdir(TEST_FOLDER):

    if not filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    filepath = os.path.join(
        TEST_FOLDER,
        filename
    )

    true_label = get_label_from_filename(
        filename
    )

    if true_label is None:
        continue

    image = Image.open(
        filepath
    ).convert("RGB")

    tensor = transform(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        # =====================================
        # STEP1
        # =====================================

        out1 = step1_model(
            tensor
        )

        probs1 = torch.softmax(
            out1,
            dim=1
        )

        step1_pred = (
            probs1[:,1] > 0.43
        ).long().item()

        # =====================================
        # STEP2A
        # =====================================

        out2a = step2a_model(
            tensor
        )

        probs2a = torch.softmax(
            out2a,
            dim=1
        )

        step2a_pred = torch.argmax(
            probs2a,
            dim=1
        ).item()

        # =====================================
        # STEP2B
        # =====================================

        out2b = step2b_model(
            tensor
        )

        probs2b = torch.softmax(
            out2b,
            dim=1
        )

        step2b_pred = torch.argmax(
            probs2b,
            dim=1
        ).item()

        # =====================================
        # FINAL PIPELINE
        # =====================================

        if step1_pred == 0:

            final_pred = step2a_pred

        else:

            final_pred = {
                0: 2,
                1: 3,
                2: 4
            }[step2b_pred]

    rows.append({

        "filename":
            filename,

        "true_class":
            CLASS_NAMES[true_label],

        "true_idx":
            true_label,

        "step1_pred":
            step1_pred,

        "step1_prob_flanke":
            probs1[0,0].item(),

        "step1_prob_oberbauch":
            probs1[0,1].item(),

        "step2a_pred":
            step2a_pred,

        "step2b_pred":
            step2b_pred,

        "final_pred":
            final_pred,

        "final_class":
            CLASS_NAMES[final_pred]

    })


# =====================================================
# SAVE
# =====================================================

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print(
    f"\nSaved: {OUTPUT_CSV}"
)


print(
    df.head(20)
)

step1_correct = 0
step1_total = 0

for _, row in df.iterrows():

    true_idx = row["true_idx"]

    if true_idx in [0,1]:
        true_step1 = 0
    else:
        true_step1 = 1

    if row["step1_pred"] == true_step1:
        step1_correct += 1

    step1_total += 1

print(
    "\nSTEP1 TEST ACC:",
    step1_correct / step1_total
)

step2a_correct = 0
step2a_total = 0

for _, row in df.iterrows():

    if row["true_idx"] not in [0,1]:
        continue

    if row["step2a_pred"] == row["true_idx"]:
        step2a_correct += 1

    step2a_total += 1

print(
    "STEP2A TEST ACC:",
    step2a_correct / step2a_total
)

mapping = {
    2:0,
    3:1,
    4:2
}

step2b_correct = 0
step2b_total = 0

for _, row in df.iterrows():

    if row["true_idx"] not in [2,3,4]:
        continue

    true_b = mapping[row["true_idx"]]

    if row["step2b_pred"] == true_b:
        step2b_correct += 1

    step2b_total += 1

print(
    "STEP2B TEST ACC:",
    step2b_correct / step2b_total
)