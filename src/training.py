"""Training loop and inference for the BiLSTM."""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

from .config import DEVICE


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total, n = 0.0, 0
    for _, x, _, y in tqdm(loader, desc='train', leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.inference_mode()
def predict_lstm(model, loader, criterion=None):
    """Returns (probabilities (N, 3) in the original data order, mean loss if a criterion is given)."""
    model.eval()
    n = len(loader.dataset)
    probs, total = np.zeros((n, 3), dtype=np.float32), 0.0
    for idx, x, _, y in loader:
        logits = model(x.to(DEVICE))
        if criterion is not None:
            total += criterion(logits, y.to(DEVICE)).item() * len(y)
        probs[idx.numpy()] = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs, total / n


def fit_bilstm(model, train_loader, val_loader, y_val, criterion, epochs, lr, patience,
               weight_decay=1e-4, ckpt_path=None, verbose=True):
    """Train with AdamW + ReduceLROnPlateau (on validation Macro-F1) + early stopping.

    Keeps the weights of the epoch with the best validation Macro-F1 and reloads them into `model`.
    Returns (history per epoch, best validation Macro-F1, training time in seconds).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0)
    history, best_f1, best_state, bad_epochs = [], -1.0, None, 0
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_probs, val_loss = predict_lstm(model, val_loader, criterion)
        val_pred = val_probs.argmax(1)
        val_acc, val_f1 = accuracy_score(y_val, val_pred), f1_score(y_val, val_pred, average='macro')
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                        'val_acc': val_acc, 'val_macro_f1': val_f1, 'lr': optimizer.param_groups[0]['lr']})
        if verbose:
            print(f'Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} '
                  f'| val_macro_F1={val_f1:.4f} | lr={optimizer.param_groups[0]["lr"]:.1e} | {time.time() - t0:.0f}s')
        scheduler.step(val_f1)
        if val_f1 > best_f1:
            best_f1, bad_epochs = val_f1, 0
            best_state = copy.deepcopy(model.state_dict())
            if ckpt_path is not None:
                torch.save(best_state, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f'Early stopping: validation Macro-F1 did not improve for {bad_epochs} epochs.')
                break
    train_time = time.time() - t_start
    model.load_state_dict(best_state)
    return history, best_f1, train_time
