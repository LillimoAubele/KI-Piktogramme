"""
Training loop with validation-based model selection and early stopping.
"""
import torch
import torch.nn as nn

from evaluation import evaluate


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch and return the mean training loss."""
    model.train()
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
    return loss_sum / len(loader.dataset)


def train_model(model, loaders, device, epochs=40, lr=3e-4, weight_decay=1e-4,
                patience=8, checkpoint_path="best_model.pt"):
    """
    Train the model, keep the checkpoint with the best validation accuracy,
    and stop early if validation accuracy does not improve for `patience` epochs.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, epochs_without_improvement = 0.0, 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        scheduler.step()

        val_acc, _, _ = evaluate(model, loaders["val"], device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc})
        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val acc {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc, epochs_without_improvement = val_acc, 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping")
                break

    print(f"Best val acc: {best_acc:.3f} (saved to {checkpoint_path})")

    # Restore the best checkpoint before returning
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, history