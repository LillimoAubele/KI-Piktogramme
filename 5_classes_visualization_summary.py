import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

STEP1_HISTORY = "/home/xrad/KI_Piktogramm/results/EfficientNetB0/step1/all_fold_history.pkl"

STEP2A_HISTORY = "/home/xrad/KI_Piktogramm/results/EfficientNetB0/step2a/all_fold_history.pkl"

STEP2B_HISTORY = "/home/xrad/KI_Piktogramm/results/EfficientNetB0/step2b/all_fold_history.pkl"

CM_IMG = "/home/xrad/KI_Piktogramm/results/EfficientNetB0/5_classes/5_classes_confusion_matrix_normalized.png"

REPORT_CSV = "/home/xrad/KI_Piktogramm/results/EfficientNetB0/5_classes/5_classes_classification_report.csv"

SAVE_PATH = "/home/xrad/KI_Piktogramm/results/EfficientNetB0/5_classes/5_classes_summary_plot.png"


# ---------------------------------------------------
# LOAD HISTORIES
# ---------------------------------------------------

with open(STEP1_HISTORY, "rb") as f:
    step1_history = pickle.load(f)

with open(STEP2A_HISTORY, "rb") as f:
    step2a_history = pickle.load(f)

with open(STEP2B_HISTORY, "rb") as f:
    step2b_history = pickle.load(f)

step1 = step1_history["EfficientNetB0"]
step2a = step2a_history["EfficientNetB0"]
step2b = step2b_history["EfficientNetB0"]


# ---------------------------------------------------
# HELPER
# ---------------------------------------------------

def mean_curve(curves):

    min_len = min(len(c) for c in curves)

    curves = np.array(
        [c[:min_len] for c in curves]
    )

    return curves.mean(axis=0)


# ---------------------------------------------------
# MEAN CURVES
# ---------------------------------------------------

step1_train_acc = mean_curve(step1["train_acc"])
step1_val_acc   = mean_curve(step1["val_acc"])

step2a_train_acc = mean_curve(step2a["train_acc"])
step2a_val_acc   = mean_curve(step2a["val_acc"])

step2b_train_acc = mean_curve(step2b["train_acc"])
step2b_val_acc   = mean_curve(step2b["val_acc"])


# ---------------------------------------------------
# LOAD CM + REPORT
# ---------------------------------------------------

cm_img = plt.imread(CM_IMG)

report_df = pd.read_csv(
    REPORT_CSV,
    index_col=0
)

# nur echte Klassen
report_df = report_df.iloc[:5][
    ["precision", "recall", "f1-score"]
]


# ---------------------------------------------------
# FIGURE
# ---------------------------------------------------

fig, axs = plt.subplots(
    2,
    2,
    figsize=(15,10)
)

# ---------------------------------------------------
# STEP1
# ---------------------------------------------------

ax = axs[0,0]

epochs = np.arange(
    1,
    len(step1_train_acc)+1
)

ax.plot(
    epochs,
    step1_train_acc,
    label="Train"
)

ax.plot(
    epochs,
    step1_val_acc,
    label="Validation"
)

ax.set_title("Step 1 (Flanke vs Oberbauch)")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Epoch")
ax.legend()


# ---------------------------------------------------
# STEP2A + STEP2B
# ---------------------------------------------------

ax = axs[0,1]

epochs2a = np.arange(
    1,
    len(step2a_train_acc)+1
)

epochs2b = np.arange(
    1,
    len(step2b_train_acc)+1
)

ax.plot(
    epochs2a,
    step2a_val_acc,
    label="Step2a Val"
)

ax.plot(
    epochs2b,
    step2b_val_acc,
    label="Step2b Val"
)

ax.set_title("Step 2 Models")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Epoch")
ax.legend()


# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------

ax = axs[1,0]

ax.imshow(cm_img)

ax.axis("off")

ax.set_title(
    "Final 5 Classes Confusion Matrix"
)


# ---------------------------------------------------
# METRICS TABLE
# ---------------------------------------------------

ax = axs[1,1]

ax.axis("off")

table = ax.table(
    cellText=np.round(
        report_df.values,
        2
    ),
    rowLabels=report_df.index,
    colLabels=report_df.columns,
    cellLoc="center",
    bbox=[0.18, 0.05, 0.64, 0.85]
)

table.auto_set_font_size(False)
table.set_fontsize(9)

for cell in table.get_celld().values():
    cell.set_height(0.12)

ax.set_title(
    "Classification Report",
    fontsize=12,
    pad=10
)

# ---------------------------------------------------
# SAVE
# ---------------------------------------------------

plt.tight_layout()

plt.savefig(
    SAVE_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"Saved to: {SAVE_PATH}"
)