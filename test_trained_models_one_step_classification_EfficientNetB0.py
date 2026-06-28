import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

from torchvision import transforms

import tkinter as tk

from tkinter import (
    filedialog,
    messagebox,
    ttk
)

from models import build_efficientnet_b0

from visualization_utils_test_trained_models_one_step_classification_EfficientNetB0 import (
    GradCAM,
    get_target_layer,
    save_triplet,
    save_misclassified_triplet
)

# =====================================================
# SETTINGS
# =====================================================



PICTOGRAM_DIR = "/home/xrad/KI_Piktogramm/Pictogram_templates"

TEST_FOLDER = "/home/xrad/KI_Piktogramm/data/Tuebingen/Pat_ohne_Piktogramm"

OUTPUT_FOLDER = "/home/xrad/KI_Piktogramm/results/EfficientNetB0/5_classes/results_test_data"

THRESHOLD_STEP1 = 0.43

MODEL_PATH = None

# =====================================================
# OUTPUT FOLDERS
# =====================================================

TRIPLET_DIR = os.path.join(
    OUTPUT_FOLDER,
    "triplets"
)

MIS_TRIPLET_DIR = os.path.join(
    OUTPUT_FOLDER,
    "misclassified_triplets"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

os.makedirs(
    TRIPLET_DIR,
    exist_ok=True
)

os.makedirs(
    MIS_TRIPLET_DIR,
    exist_ok=True
)

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# =====================================================
# LABEL MAPPING
# =====================================================

FILENAME_TO_CLASSIDX = {

    1: 3,   # Links
    2: 2,   # MCL
    3: 4,   # Subcostal
    4: 1,   # Intercostal
    5: 0    # Hoher Flankenschnitt
}

CLASS_NAMES_5 = [

    "Hoher_Flankenschnitt",
    "Intercostal",
    "MCL_rechts",
    "Oberbauch_links",
    "Subcostal_rechts"
]

CLASS_NAMES_4 = [

    "Flanke",
    "MCL_rechts",
    "Oberbauch_links",
    "Subcostal_rechts"
]

DISPLAY_NAMES_4 = [
    "Flanke",
    "MCL rechts",
    "Oberbauch links",
    "Subcostal rechts"
]

DISPLAY_NAMES_5 = [
    "Hoher Flankenschnitt",
    "Intercostal",
    "MCL rechts",
    "Oberbauch links",
    "Subcostal rechts"
]

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
# HELPERS
# =====================================================

def get_label_from_filename(filename):

    stem = os.path.splitext(
        filename
    )[0]

    try:

        idx = int(
            stem.split("_")[-1]
        )

    except ValueError:

        return None

    return FILENAME_TO_CLASSIDX.get(
        idx,
        None
    )


def load_model():

    num_classes = 5 if MODE == 5 else 4

    model = build_efficientnet_b0(
        num_classes=num_classes,
        pretrained=False
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model
# =====================================================
# MAIN
# =====================================================

def main(progress_callback=None):
    
    print("MODE INSIDE MAIN =", MODE)
    model = load_model()
    
    cam_model = GradCAM(
    model,
    get_target_layer(model)
)
    

    # -----------------------------------------
    # Metrics containers
    # -----------------------------------------

    y_true = []
    y_pred = []



    prediction_rows = []
    

    # =====================================================
    # LOOP OVER TEST IMAGES
    # =====================================================
    
    image_files = [

    f for f in os.listdir(TEST_FOLDER)

        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ]
    
    total_images = len(image_files)
    
    for idx, filename in enumerate(
    sorted(image_files),
    start=1
    ):
    
        if progress_callback is not None:
    
            progress_callback(
                idx,
                total_images,
                filename
            )
    
        filepath = os.path.join(
            TEST_FOLDER,
            filename
        )
    
        true_label = get_label_from_filename(
            filename
        )
    
        if true_label is None:
            print(
                f"Skipped: {filename}"
            )
            continue
    
        # -----------------------------------------
        # Load image
        # -----------------------------------------
    
        image = Image.open(
            filepath
        ).convert("RGB")
    
        tensor = transform(
            image
        ).unsqueeze(0).to(device)
    
        # -----------------------------------------
        with torch.no_grad():

            outputs = model(tensor)
        
            probs = torch.softmax(
                outputs,
                dim=1
            )
        
            final_pred = torch.argmax(
                probs,
                dim=1
            ).item()
            
        
    
        # -----------------------------------------
        # Convert GT for 4 classes
        # -----------------------------------------
    
        gt_for_pipeline = true_label
    
        if MODE == 4:
    
            if true_label in [0,1]:
                gt_for_pipeline = 0
    
            elif true_label == 2:
                gt_for_pipeline = 1
    
            elif true_label == 3:
                gt_for_pipeline = 2
    
            elif true_label == 4:
                gt_for_pipeline = 3
    
        y_true.append(
            gt_for_pipeline
        )
    
        y_pred.append(
            final_pred
        )
    
        # -----------------------------------------
        # GRADCAM
        # -----------------------------------------
    
        tensor_grad = tensor.clone()

        tensor_grad.requires_grad_(True)
        
        with torch.enable_grad():
        
            cam = cam_model(
                tensor_grad,
                final_pred
            )
    
        # -----------------------------------------
        # Triplet
        # -----------------------------------------

        save_triplet(
    
            image=image,
    
            pred_class=final_pred,
    
            cam=cam,
    
            pictogram_dir=PICTOGRAM_DIR,
    
            save_path=os.path.join(
    
                TRIPLET_DIR,
    
                os.path.splitext(
                    filename
                )[0]
    
                + "_triplet.png"
            )
        )
    
        # -----------------------------------------
        # Misclassified Triplet
        # -----------------------------------------
    
        if final_pred != gt_for_pipeline:
    
            save_misclassified_triplet(
    
                image=image,
    
                true_class=gt_for_pipeline,
    
                pred_class=final_pred,
    
                cam=cam,
    
                pictogram_dir=PICTOGRAM_DIR,
    
                save_path=os.path.join(
    
                    MIS_TRIPLET_DIR,
    
                    os.path.splitext(
                        filename
                    )[0]
    
                    + "_misclassified.png"
                )
            )
    
        # -----------------------------------------
        # CSV Row
        # -----------------------------------------
    
        prediction_rows.append({

            "filename":
                filename,
        
            "true_label":
                gt_for_pipeline,
        
            "pred_label":
                final_pred
        })
            
    # =====================================================
    # SAVE PREDICTIONS
    # =====================================================
    
    predictions_df = pd.DataFrame(
        prediction_rows
    )
    
    predictions_df.to_csv(
    
        os.path.join(
            OUTPUT_FOLDER,
            "predictions.csv"
        ),
    
        index=False
    )
    
    # =====================================================
    # ACCURACIES
    # =====================================================
    
    pipeline_acc = accuracy_score(
        y_true,
        y_pred
    )
    
    
    
    if MODE == 4:

        labels = [0,1,2,3]
        class_names = DISPLAY_NAMES_4
    
    else:
    
        labels = [0,1,2,3,4]
        class_names = DISPLAY_NAMES_5
        
        
    # =====================================================
    # CLASSIFICATION REPORT
    # ===================================================
    
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    report_df = pd.DataFrame(report).transpose()
    
    report_df.to_csv(
        os.path.join(
            OUTPUT_FOLDER,
            "classification_report.csv"
        )
    )
    
    # =====================================================
    # EVALUATION SUMMARY
    # ===================================================
    summary_df = pd.DataFrame({

        "metric": [
            "pipeline_accuracy"
        ],
    
        "value": [
            pipeline_acc
        ]
    })
    
    summary_df.to_csv(
    
        os.path.join(
            OUTPUT_FOLDER,
            "evaluation_summary.csv"
        ),
    
        index=False
    )
    
    
    # =====================================================
    # CONFUSION MATRIX
    # =====================================================
    
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )
       
    
    # NORMALIZE CONFUSION MATRIX
    
    cm = cm.astype(float)
    
    cm = (
        cm /
        cm.sum(
            axis=1,
            keepdims=True
        )
    )

    cm = np.nan_to_num(cm)
    
    
    plt.figure(
        figsize=(8,6)
    )
    
    sns.heatmap(
    
        cm,
    
        annot=True,
    
        fmt=".2f",
    
        cmap="Blues",
    
        xticklabels=class_names,
    
        yticklabels=class_names
    )
    
    plt.title(
        f"Accuracy={pipeline_acc:.3f}"
    )
    
    plt.xlabel("Prediction")
    plt.ylabel("True")
    
    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=16
    )
    
    plt.tight_layout()
    
    plt.savefig(
    
        os.path.join(
            OUTPUT_FOLDER,
            "confusion_matrix.png"
        ),
    
        dpi=300
    )
    
    plt.close()
        
    
class EvaluationGUI:

    def __init__(self):

        self.root = tk.Tk()

        self.root.title(
            ""
        )

        self.root.geometry("450x350")

        self.test_folder = None
        self.output_folder = None

        # -----------------------------------
        # Title
        # -----------------------------------

        ttk.Label(
            self.root,
            text="One-Step Classification Evaluation",
            font=("Arial", 12, "bold")
        ).pack(
            pady=10
        )

        # -----------------------------------
        # MODE
        # -----------------------------------

        self.mode_var = tk.IntVar(
            value=5
        )

        ttk.Label(
            self.root,
            text="Classification Mode"
        ).pack()

        ttk.Radiobutton(
            self.root,
            text="4 Classes",
            variable=self.mode_var,
            value=4
        ).pack()

        ttk.Radiobutton(
            self.root,
            text="5 Classes",
            variable=self.mode_var,
            value=5
        ).pack()
        
        # -----------------------------------
        # Model Button
        # -----------------------------------
        
        ttk.Button(
            self.root,
            text="Select Model (.pth)",
            command=self.select_model
        ).pack(pady=5)

        # -----------------------------------
        # Folder Buttons
        # -----------------------------------

        ttk.Button(
            self.root,
            text="Select Test Folder",
            command=self.select_test_folder
        ).pack(
            pady=5
        )

        ttk.Button(
            self.root,
            text="Select Output Folder",
            command=self.select_output_folder
        ).pack(
            pady=5
        )

        # -----------------------------------
        # Run Button
        # -----------------------------------

        ttk.Button(
            self.root,
            text="Run Evaluation",
            command=self.run_evaluation
        ).pack(
            pady=15
        )

        # -----------------------------------
        # Progress Bar
        # -----------------------------------

        self.progress = ttk.Progressbar(
            self.root,
            orient="horizontal",
            length=350,
            mode="determinate"
        )

        self.progress.pack(
            pady=10
        )

        # -----------------------------------
        # Status Label
        # -----------------------------------

        self.status_var = tk.StringVar()

        self.status_var.set(
            "Ready"
        )

        ttk.Label(
            self.root,
            textvariable=self.status_var
        ).pack(
            pady=5
        )

        self.root.mainloop()
        
    # ==================================================
    # MODEL
    # ==================================================
        
    def select_model(self):

        self.model_path = filedialog.askopenfilename(
        
            title="Select Model",
        
            filetypes=[
                ("PyTorch Model", "*.pth")
        ]
    )

    # ==================================================
    # TEST FOLDER
    # ==================================================

    def select_test_folder(self):

        self.test_folder = filedialog.askdirectory()

        if self.test_folder:

            self.status_var.set(
                f"Test folder selected"
            )

    # ==================================================
    # OUTPUT FOLDER
    # ==================================================

    def select_output_folder(self):

        self.output_folder = filedialog.askdirectory()

        if self.output_folder:

            self.status_var.set(
                f"Output folder selected"
            )

    # ==================================================
    # PROGRESS CALLBACK
    # ==================================================

    def update_progress(
        self,
        current,
        total,
        filename
    ):

        self.progress["maximum"] = total

        self.progress["value"] = current

        self.status_var.set(
            f"{current}/{total}: {filename}"
        )

        self.root.update_idletasks()

    # ==================================================
    # RUN
    # ==================================================

    def run_evaluation(self):

        global MODE
        global TEST_FOLDER
        global OUTPUT_FOLDER
        global TRIPLET_DIR
        global MIS_TRIPLET_DIR
        global MODEL_PATH

        MODEL_PATH = self.model_path

        if self.test_folder is None:

            messagebox.showerror(
                "Error",
                "Please select a test folder."
            )

            return

        if self.output_folder is None:

            messagebox.showerror(
                "Error",
                "Please select an output folder."
            )

            return
        
        if self.model_path is None:

            messagebox.showerror(
                "Error",
                "Please select a model."
            )
        
            return

        MODE = self.mode_var.get()

        TEST_FOLDER = self.test_folder

        OUTPUT_FOLDER = self.output_folder

        TRIPLET_DIR = os.path.join(
            OUTPUT_FOLDER,
            "triplets"
        )

        MIS_TRIPLET_DIR = os.path.join(
            OUTPUT_FOLDER,
            "misclassified_triplets"
        )

        os.makedirs(
            OUTPUT_FOLDER,
            exist_ok=True
        )

        os.makedirs(
            TRIPLET_DIR,
            exist_ok=True
        )

        os.makedirs(
            MIS_TRIPLET_DIR,
            exist_ok=True
        )

        self.progress["value"] = 0

        self.status_var.set(
            "Loading models..."
        )

        self.root.update()
        
        

        try:

            main(
                progress_callback=
                self.update_progress
            )

            self.progress["value"] = (
                self.progress["maximum"]
            )

            self.status_var.set(
                "Finished"
            )

            messagebox.showinfo(
                "Finished",
                "Evaluation completed successfully."
            )

        except Exception as e:

            messagebox.showerror(
                "Error",
                str(e)
            )

            self.status_var.set(
                "Error"
            )



        
            
# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    EvaluationGUI()