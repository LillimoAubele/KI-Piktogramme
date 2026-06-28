import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================
# CLASSIFICATION REPORT
# =====================================================

def save_classification_report(
    y_true,
    y_pred,
    class_names,
    output_dir
):

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
            output_dir,
            "classification_report.csv"
        )
    )

    return report_df


# =====================================================
# PREDICTIONS CSV
# =====================================================

def save_predictions_csv(
    y_true,
    y_pred,
    y_probs,
    output_dir
):

    rows = []

    for i in range(len(y_true)):

        row = {

            "true_label":
                y_true[i],

            "pred_label":
                y_pred[i]
        }

        for c in range(
            len(y_probs[i])
        ):

            row[
                f"prob_class_{c}"
            ] = float(
                y_probs[i][c]
            )

        rows.append(row)

    df = pd.DataFrame(
        rows
    )

    df.to_csv(

        os.path.join(
            output_dir,
            "predictions.csv"
        ),

        index=False
    )

    return df


# =====================================================
# CONFUSION MATRIX
# =====================================================

def plot_confusion_matrix(
    cm,
    class_names,
    accuracy,
    save_path,
    model_name="EfficientNet B0"
):

    plt.figure(
        figsize=(10,8)
    )

    sns.heatmap(

        cm,

        annot=True,

        fmt=".2f",

        cmap="Blues",

        square=True,

        linewidths=0.5,

        xticklabels=class_names,

        yticklabels=class_names,

        annot_kws={
            "size":16
        }
    )

    plt.title(

        f"{model_name} "
        f"{len(class_names)} Classes "
        f"Normalized Confusion Matrix\n"
        f"Accuracy = {accuracy:.3f}",

        fontsize=22
    )

    plt.xlabel(
        "Prediction",
        fontsize=18
    )

    plt.ylabel(
        "True",
        fontsize=18
    )

    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=16
    )

    plt.yticks(
        rotation=0,
        fontsize=16
    )

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()
    
    return cm


# =====================================================
# SUMMARY CSV
# =====================================================

def save_summary(
    val_loss,
    val_acc,
    top2_acc,
    output_dir
):

    summary = pd.DataFrame({

        "metric": [

            "validation_loss",

            "validation_accuracy",

            "top2_accuracy"
        ],

        "value": [

            val_loss,

            val_acc,

            top2_acc
        ]
    })

    summary.to_csv(

        os.path.join(
            output_dir,
            "evaluation_summary.csv"
        ),

        index=False
    )

    return summary


# =====================================================
# COMPLETE PIPELINE
# =====================================================



def create_visualizations(

    y_true,
    y_pred,
    y_probs,

    class_names,

    output_dir,

    val_loss,
    val_acc,
    top2_acc
):

    print(class_names)
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print(
        "\nSaving classification report..."
    )

    save_classification_report(

        y_true,
        y_pred,

        class_names,

        output_dir
    )

    print(
        "Saving prediction CSV..."
    )

    save_predictions_csv(

        y_true,
        y_pred,
        y_probs,

        output_dir
    )

    # =====================================
    # CONFUSION MATRIX
    # =====================================

    cm = confusion_matrix(
        y_true,
        y_pred
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

    print(
        "Saving confusion matrix..."
    )

    plot_confusion_matrix(

        cm=cm,

        class_names=class_names,

        accuracy=val_acc,

        save_path=os.path.join(
            output_dir,
            "confusion_matrix.png"
        )
    )

    print(
        "Saving summary..."
    )

    save_summary(

        val_loss,
        val_acc,
        top2_acc,

        output_dir
    )

    print(
        "Visualization finished."
    )