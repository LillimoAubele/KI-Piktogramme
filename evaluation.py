"""
Evaluation utilities: predictions, accuracy, classification report and confusion matrix.
"""
import torch
from sklearn.metrics import classification_report, confusion_matrix


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (accuracy, true labels, predicted labels) for a DataLoader."""
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        out = model(x.to(device))
        y_pred += out.argmax(1).cpu().tolist()
        y_true += y.tolist()
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    return acc, y_true, y_pred


def report(model, loader, device, class_names):
    """Print accuracy, per-class metrics and the confusion matrix."""
    acc, y_true, y_pred = evaluate(model, loader, device)
    print(f"\nAccuracy: {acc:.3f}\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(confusion_matrix(y_true, y_pred))
    return acc, y_true, y_pred