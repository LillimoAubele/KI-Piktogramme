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

from visualization_utils_test_trained_models_two_step_classification_EfficientNetB0 import (
    GradCAM,
    get_target_layer,
    save_triplet,
    save_misclassified_triplet
)

# =====================================================
# SETTINGS
# =====================================================


STEP1_MODEL = "/home/xrad/KI_Piktogramm/trained_models_two_step_classification/EfficientNetB0/step1/EfficientNetB0.pth"

STEP2A_MODEL = "/home/xrad/KI_Piktogramm/trained_models_two_step_classification/EfficientNetB0/step2a/EfficientNetB0.pth"

STEP2B_MODEL = "/home/xrad/KI_Piktogramm/trained_models_two_step_classification/EfficientNetB0/step2b/EfficientNetB0.pth"

PICTOGRAM_DIR = "/home/xrad/KI_Piktogramm/Pictogram_templates"

TEST_FOLDER = "/home/xrad/KI_Piktogramm/data/Tuebingen/Pat_ohne_Piktogramm"

OUTPUT_FOLDER = "/home/xrad/KI_Piktogramm/results/EfficientNetB0/5_classes/results_test_data"

THRESHOLD_STEP1 = 0.43

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

# =====================================================
# LOAD MODELS
# =====================================================

def load_models():

    step1 = build_efficientnet_b0(
        num_classes=2,
        pretrained=False
    )

    step2a = build_efficientnet_b0(
        num_classes=2,
        pretrained=False
    )

    step2b = build_efficientnet_b0(
        num_classes=3,
        pretrained=False
    )

    step1.load_state_dict(
        torch.load(
            STEP1_MODEL,
            map_location=device
        )
    )

    step2a.load_state_dict(
        torch.load(
            STEP2A_MODEL,
            map_location=device
        )
    )

    step2b.load_state_dict(
        torch.load(
            STEP2B_MODEL,
            map_location=device
        )
    )

    step1.to(device).eval()
    step2a.to(device).eval()
    step2b.to(device).eval()

    return step1, step2a, step2b

# =====================================================
# MAIN
# =====================================================

def main(progress_callback=None):
    
    print("MODE INSIDE MAIN =", MODE)

    step1_model, step2a_model, step2b_model = load_models()


    cam_step1 = GradCAM(
        step1_model,
        get_target_layer(step1_model)
    )

    cam_step2a = GradCAM(
        step2a_model,
        get_target_layer(step2a_model)
    )

    cam_step2b = GradCAM(
        step2b_model,
        get_target_layer(step2b_model)
    )

    # -----------------------------------------
    # Metrics containers
    # -----------------------------------------

    y_true = []
    y_pred = []

    step1_true = []
    step1_preds = []

    step2a_true = []
    step2a_preds = []

    step2b_true = []
    step2b_preds = []

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
        # STEP 1
        # -----------------------------------------
    
        with torch.no_grad():
    
            out1 = step1_model(
                tensor
            )
    
            probs1 = torch.softmax(
                out1,
                dim=1
            )
    
            step1_pred = (
                probs1[:,1] >
                THRESHOLD_STEP1
            ).long().item()
    
        if true_label in [0,1]:
            true_step1 = 0
        else:
            true_step1 = 1
    
        step1_true.append(
            true_step1
        )
    
        step1_preds.append(
            step1_pred
        )
    
        # -----------------------------------------
        # STEP 2A
        # -----------------------------------------
        
        if MODE == 5:
        
            with torch.no_grad():
        
                out2a = step2a_model(
                    tensor
                )
        
                step2a_pred = torch.argmax(
                    out2a,
                    dim=1
                ).item()
        
            if true_label in [0,1]:
        
                step2a_true.append(
                    true_label
                )
        
                step2a_preds.append(
                    step2a_pred
                )
        
        else:
        
            step2a_pred = None
    
        # -----------------------------------------
        # STEP 2B
        # -----------------------------------------
    
        with torch.no_grad():
    
            out2b = step2b_model(
                tensor
            )
    
            step2b_pred = torch.argmax(
                out2b,
                dim=1
            ).item()
    
        if true_label in [2,3,4]:
    
            mapped_true_step2b = {
    
                2:0,
                3:1,
                4:2
    
            }[true_label]
    
            step2b_true.append(
                mapped_true_step2b
            )
    
            step2b_preds.append(
                step2b_pred
            )
    
        # -----------------------------------------
        # FINAL PREDICTION
        # -----------------------------------------
    
        if MODE == 5:
    
            if step1_pred == 0:
    
                final_pred = step2a_pred
    
                gradcam_model = cam_step2a
    
                gradcam_class = (
                    step2a_pred
                )
    
            else:
    
                final_pred = {
    
                    0:2,
                    1:3,
                    2:4
    
                }[step2b_pred]
    
                gradcam_model = cam_step2b
    
                gradcam_class = (
                    step2b_pred
                )
    
        else:
    
            if step1_pred == 0:
    
                final_pred = 0
    
            else:
    
                final_pred = {
    
                    0:1,
                    1:2,
                    2:3
    
                }[step2b_pred]
                
                
        if MODE == 4:

            if step1_pred == 0:
        
                final_pred = 0   # Flanke
        
                gradcam_model = cam_step1
                gradcam_class = step1_pred
        
            else:
        
                final_pred = {
                    0:1,   # MCL
                    1:2,   # Links
                    2:3    # Subcostal
                }[step2b_pred]
        
                gradcam_model = cam_step2b
                gradcam_class = step2b_pred
    
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
    
            cam = gradcam_model(
                tensor_grad,
                gradcam_class
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
                final_pred,
    
            "step1_pred":
                step1_pred,
    
            "step2a_pred":
                step2a_pred,
    
            "step2b_pred":
                step2b_pred,
    
            "step1_prob_flanke":
                probs1[0,0].item(),
    
            "step1_prob_oberbauch":
                probs1[0,1].item()
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
    
    step1_acc = accuracy_score(
        step1_true,
        step1_preds
    )
    
    step2b_acc = accuracy_score(
        step2b_true,
        step2b_preds
    )
    
    if MODE == 5:

        step2a_acc = accuracy_score(
            step2a_true,
            step2a_preds
        )
        
        summary_df = pd.DataFrame({
    
            "metric":[
    
                "pipeline_accuracy",
                "step1_accuracy",
                "step2a_accuracy",
                "step2b_accuracy"
            ],
    
            "value":[
    
                pipeline_acc,
                step1_acc,
                step2a_acc,
                step2b_acc
            ]
        })
    
    else:
    
        step2a_acc = np.nan
    
    
        summary_df = pd.DataFrame({
    
            "metric":[
    
                "pipeline_accuracy",
                "step1_accuracy",
                "step2b_accuracy"
            ],
    
            "value":[
    
                pipeline_acc,
                step1_acc,
                step2b_acc
            ]
        })
    
    summary_df.to_csv(
    
        os.path.join(
            OUTPUT_FOLDER,
            "evaluation_summary.csv"
        ),
    
        index=False
    )
    
    print("\n==========================")
    print("RESULTS")
    print("==========================")
    
    print(
        f"Pipeline Accuracy: "
        f"{pipeline_acc:.4f}"
    )
    
    print(
        f"Step1 Accuracy: "
        f"{step1_acc:.4f}"
    )
    
    if MODE == 5:
    
        print(
            f"Step2a Accuracy: "
            f"{step2a_acc:.4f}"
        )
    
    print(
        f"Step2b Accuracy: "
        f"{step2b_acc:.4f}"
    )
    
    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================
    
    class_names = (
        CLASS_NAMES_5
        if MODE == 5
        else CLASS_NAMES_4
    )
    
    
    
    report = classification_report(
    
        y_true,
        y_pred,
    
        target_names=class_names,
    
        output_dict=True,
    
        zero_division=0
    )
    
    report_df = pd.DataFrame(
        report
    ).transpose()
    
    report_df.to_csv(
    
        os.path.join(
            OUTPUT_FOLDER,
            "classification_report.csv"
        )
    )
    
    # =====================================================
    # PIPELINE CONFUSION MATRIX
    # =====================================================
    
    display_names = (
        DISPLAY_NAMES_5
        if MODE == 5
        else DISPLAY_NAMES_4
    )
    
    
    if MODE == 4:

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0,1,2,3]
        )
    
    else:
        
        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0,1,2,3,4]
        )
        
    
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
    
        xticklabels= display_names,
    
        yticklabels=display_names
    )
    
    plt.title(
    
        f"Pipeline\nAccuracy={pipeline_acc:.3f}"
    
    )
    
    plt.xlabel(
        "Prediction"
    )
    
    plt.ylabel(
        "True"
    )
    
    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=16
    )
    
    plt.tight_layout()
    
    plt.savefig(
    
        os.path.join(
            OUTPUT_FOLDER,
            "confusion_matrix_pipeline.png"
        ),
    
        dpi=300
    )
    
    plt.close()
    
    # =====================================================
    # STEP1 CONFUSION MATRIX
    # =====================================================
    
    cm1 = confusion_matrix(
        step1_true,
        step1_preds
    )
    
    cm1 = cm1.astype(float)
    
    cm1 = (
        cm1 /
        cm1.sum(
            axis=1,
            keepdims=True
        )
    )
    
    cm1 = np.nan_to_num(cm1)
    
    plt.figure(
        figsize=(6,5)
    )
    
    sns.heatmap(
    
        cm1,
    
        annot=True,
    
        fmt=".2f",
    
        cmap="Blues",
    
        xticklabels=[
            "Flanke",
            "Oberbauch"
        ],
    
        yticklabels=[
            "Flanke",
            "Oberbauch"
        ]
    )
    
    plt.title(
    
        f"Step1\nAccuracy={step1_acc:.3f}"
    
    )
    
    plt.tight_layout()
    
    plt.savefig(
    
        os.path.join(
            OUTPUT_FOLDER,
            "confusion_matrix_step1.png"
        ),
    
        dpi=300
    )
    
    plt.close()
   
# =====================================================
# STEP2A CONFUSION MATRIX
# =====================================================
    print("STEP2A BLOCK CALLED, MODE =", MODE)
    
    if MODE == 5:
    
        cm2a = confusion_matrix(
            step2a_true,
            step2a_preds
        )
    
        cm2a = cm2a.astype(float)
    
        cm2a = (
            cm2a /
            cm2a.sum(
                axis=1,
                keepdims=True
            )
        )
    
        cm2a = np.nan_to_num(cm2a)
    
        plt.figure(
            figsize=(6,5)
        )
    
        sns.heatmap(
            cm2a,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=[
                "Hoher Flankenschnitt",
                "Intercostal"
            ],
            yticklabels=[
                "Hoher Flankenschnitt",
                "Intercostal"
            ]
        )
    
        plt.title(
            f"Step2a\nAccuracy={step2a_acc:.3f}"
        )
    
        plt.tight_layout()
    
        plt.savefig(
            os.path.join(
                OUTPUT_FOLDER,
                "confusion_matrix_step2a.png"
            ),
            dpi=300
        )
    
        plt.close()
        
    # =====================================================
    # STEP2B CONFUSION MATRIX
    # =====================================================
    
    cm2b = confusion_matrix(
        step2b_true,
        step2b_preds
    )
    
    cm2b = cm2b.astype(float)
    
    cm2b = (
        cm2b /
        cm2b.sum(
            axis=1,
            keepdims=True
        )
    )
    
    cm2b = np.nan_to_num(cm2b)
    
    plt.figure(
        figsize=(6,5)
    )
    
    sns.heatmap(
    
        cm2b,
    
        annot=True,
    
        fmt=".2f",
    
        cmap="Blues",
    
        xticklabels=[
            "MCL",
            "Links",
            "Subcostal"
        ],
    
        yticklabels=[
            "MCL",
            "Links",
            "Subcostal"
        ]
    )
    
    plt.title(
    
        f"Step2b\nAccuracy={step2b_acc:.3f}"
    
    )
    
    plt.tight_layout()
    
    plt.savefig(
    
        os.path.join(
            OUTPUT_FOLDER,
            "confusion_matrix_step2b.png"
        ),
    
        dpi=300
    )
    
    plt.close()
    
    print(
        "\nFinished."
    )
    
    
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
            text="Two-Step Classification Evaluation",
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