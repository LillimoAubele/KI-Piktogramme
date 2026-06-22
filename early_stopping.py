import numpy as np
import torch

# -----------------------------------------------------------------------------
# Early stopping
# -----------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, min_epochs=50):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            return

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

     
        if epoch >= self.min_epochs and self.counter >= self.patience:
            self.early_stop = True
