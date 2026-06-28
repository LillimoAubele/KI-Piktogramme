import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# FOCAL LOSS
# =====================================================

class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha=None,
        gamma=2.0
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits,
        targets
    ):

        ce_loss = F.cross_entropy(

            logits,
            targets,

            weight=self.alpha,

            reduction="none"
        )

        pt = torch.exp(
            -ce_loss
        )

        focal_loss = (
            (1 - pt) ** self.gamma
        ) * ce_loss

        return focal_loss.mean()


# =====================================================
# LOSS
# =====================================================

def get_loss_function(
    weight=None,
    label_smoothing=0.0
):

    return nn.CrossEntropyLoss(

        weight=weight,

        label_smoothing=label_smoothing
    )


# =====================================================
# EVALUATION
# =====================================================

def evaluate_model(
    model,
    val_loader,
    criterion,
    device,
    return_preds=False
):

    model.eval()

    val_loss = 0.0

    correct = 0
    total = 0

    top2_correct = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += (
                loss.item()
                * images.size(0)
            )

            probs = torch.softmax(
                outputs,
                dim=1
            )

            # ----------------------------------
            # TOP-1
            # ----------------------------------

            preds = torch.argmax(
                probs,
                dim=1
            )

            correct += (
                preds == labels
            ).sum().item()

            # ----------------------------------
            # TOP-2
            # ----------------------------------

            top2 = torch.topk(
                probs,
                k=2,
                dim=1
            ).indices

            for i in range(labels.size(0)):

                if labels[i] in top2[i]:

                    top2_correct += 1

            total += labels.size(0)

            # ----------------------------------
            # STORE PREDICTIONS
            # ----------------------------------

            if return_preds:

                all_labels.extend(
                    labels.cpu().numpy()
                )

                all_preds.extend(
                    preds.cpu().numpy()
                )

                all_probs.extend(
                    probs.cpu().numpy()
                )

    val_loss /= total

    val_acc = (
        correct / total
    )

    top2_acc = (
        top2_correct / total
    )

    if return_preds:

        return (

            val_loss,

            val_acc,

            top2_acc,

            all_labels,

            all_preds,

            all_probs
        )

    return (

        val_loss,

        val_acc,

        top2_acc
    )
