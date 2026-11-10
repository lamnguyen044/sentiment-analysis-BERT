"""Metrics, confusion matrices, bootstrap, threshold calibration and saving/loading of model predictions."""
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .config import LABELS, PRED_DIR
from .data import collate_lstm, make_loader
from .text import tokenize
from .training import predict_lstm
from .transformer import predict_transformer

METRIC_COLS = ['Accuracy', 'Macro-F1', 'Weighted-F1', 'F1 negative', 'F1 neutral', 'F1 positive']


def compute_metrics(y_true, y_pred) -> dict:
    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Macro-F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Weighted-F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 negative': per_class[0], 'F1 neutral': per_class[1], 'F1 positive': per_class[2],
    }


def report(name, y_true, y_pred) -> dict:
    m = compute_metrics(y_true, y_pred)
    print(f'=== {name} ===  Accuracy = {m["Accuracy"]:.4f} | Macro-F1 = {m["Macro-F1"]:.4f}')
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4, zero_division=0))
    return m


def plot_confusion(y_true, y_pred, title, ax=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    annot = np.array([[f'{cm_norm[i, j]:.1%}\n({cm[i, j]:,})' for j in range(3)] for i in range(3)])
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues', vmin=0, vmax=1, cbar=False,
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set(xlabel='Predicted', ylabel='Actual', title=title)
    return ax


def plot_history(hist, title, step_losses=None):
    """Learning curves. With `step_losses` (BERT): loss per step + validation metrics; without (BiLSTM): train/val loss + metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    if step_losses is not None and len(step_losses):
        window = max(1, len(step_losses) // 50)
        axes[0].plot(step_losses, alpha=.25, color='#4c72b0', label='step loss')
        axes[0].plot(pd.Series(step_losses).rolling(window, min_periods=1).mean(), color='#4c72b0',
                     label=f'moving average ({window} steps)')
        per_epoch = len(step_losses) // len(hist)
        for e in range(1, len(hist)):
            axes[0].axvline(e * per_epoch, ls='--', color='gray', lw=1)
        axes[0].set(title=f'{title} – train loss per step', xlabel='Step (batch)', ylabel='Loss')
    else:
        axes[0].plot(hist['epoch'], hist['train_loss'], 'o-', label='train')
        axes[0].plot(hist['epoch'], hist['val_loss'], 's-', label='validation')
        axes[0].set(title=f'{title} – loss (class-weighted cross-entropy)', xlabel='Epoch', ylabel='Loss')
    axes[0].legend()
    axes[1].plot(hist['epoch'], hist['val_acc'], 'o-', label='val accuracy')
    axes[1].plot(hist['epoch'], hist['val_macro_f1'], 's-', label='val Macro-F1')
    axes[1].set(title=f'{title} – validation metrics', xlabel='Epoch')
    axes[1].legend()
    for ax in axes:
        ax.xaxis.get_major_locator().set_params(integer=True)
    plt.tight_layout()
    plt.show()


def confusion_table(y_true, y_pred) -> pd.DataFrame:
    """Confusion matrix as a table, row-normalised (each row = 100% of the reviews of one true label)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    table = pd.DataFrame(cm / cm.sum(axis=1, keepdims=True).clip(min=1),
                         index=[f'actual: {l}' for l in LABELS], columns=[f'predicted: {l}' for l in LABELS])
    return table.style.format('{:.1%}')


def macro_f1_from_confusion(cm) -> float:
    """Macro-F1 from a 3x3 confusion matrix; F1_c = 2TP / (2TP + FP + FN)."""
    tp = np.diag(cm).astype(float)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)))


def bootstrap_macro_f1(y_true, preds: dict, n_boot=1000, seed=42) -> dict:
    """Resample the test set with replacement n_boot times, using the SAME sample for every model (so differences can be paired).

    preds: {model name: predicted labels}. Returns {name: array of n_boot Macro-F1 values}.
    """
    y = np.asarray(y_true)
    P = {k: np.asarray(v) for k, v in preds.items()}
    rng = np.random.default_rng(seed)
    scores = {k: np.empty(n_boot) for k in P}
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        for k, p in P.items():
            scores[k][b] = macro_f1_from_confusion(np.bincount(yb * 3 + p[idx], minlength=9).reshape(3, 3))
    return scores


def summarize_bootstrap(scores: dict, point: dict, level=0.95) -> pd.DataFrame:
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return pd.DataFrame({k: {'Macro-F1': point[k], f'CI {level:.0%} low': np.percentile(v, lo),
                             f'CI {level:.0%} high': np.percentile(v, hi)} for k, v in scores.items()}).T


def paired_difference(scores: dict, a: str, b: str, level=0.95) -> dict:
    """Macro-F1 difference between two models (a - b) over the same bootstrap resamples."""
    d = scores[a] - scores[b]
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {'mean': float(d.mean()), 'low': float(np.percentile(d, lo)), 'high': float(np.percentile(d, hi)),
            'P(a > b)': float((d > 0).mean())}


def predict_with_neutral_bias(probs, bias: float) -> np.ndarray:
    """Add `bias` to the neutral log-probability, then take the argmax (bias < 0: predict neutral less often)."""
    logp = np.log(np.clip(np.asarray(probs, dtype=np.float64), 1e-9, 1.0))
    logp[:, 1] += bias
    return logp.argmax(1)


def tune_neutral_bias(val_probs, y_val, grid=None):
    """Choose the neutral `bias` that maximises Macro-F1 on VALIDATION (the test set is not used)."""
    grid = np.round(np.arange(-2.0, 2.0001, 0.1), 2) if grid is None else np.asarray(grid)
    y = np.asarray(y_val)
    scores = np.array([macro_f1_from_confusion(
        np.bincount(y * 3 + predict_with_neutral_bias(val_probs, b), minlength=9).reshape(3, 3)) for b in grid])
    return float(grid[scores.argmax()]), grid, scores


def predict_lstm_probs(model, vocab, texts, max_len, batch_size=256) -> np.ndarray:
    """BiLSTM probabilities for a list of cleaned texts (in batches, faster than one by one)."""
    ids = [vocab.encode(tokenize(t), max_len) for t in texts]
    loader = make_loader(ids, np.zeros(len(ids), dtype=int), batch_size, shuffle=False, collate_fn=collate_lstm)
    return predict_lstm(model, loader)[0]


def _slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def save_predictions(name: str, probs: np.ndarray, info: dict, test_index=None) -> None:
    """Save test-set probabilities and extra info (training time, parameters...) so notebook 06 can compare models.

    test_index: positions of the test rows that were predicted (None = the whole test set).
    """
    slug = _slug(name)
    arrays = {'probs': probs}
    if test_index is not None:
        arrays['test_index'] = np.asarray(test_index)
    np.savez_compressed(PRED_DIR / f'{slug}.npz', **arrays)
    (PRED_DIR / f'{slug}.json').write_text(json.dumps({'name': name, **info}, ensure_ascii=False, indent=2),
                                           encoding='utf-8')


def load_predictions() -> dict:
    """Returns {model name: {'probs', 'test_index', ...info}} for every saved model."""
    out = {}
    for meta_path in sorted(PRED_DIR.glob('*.json')):
        info = json.loads(meta_path.read_text(encoding='utf-8'))
        arrays = np.load(meta_path.with_suffix('.npz'))
        info['probs'] = arrays['probs']
        info['test_index'] = arrays['test_index'] if 'test_index' in arrays else None
        out[info.pop('name')] = info
    return out
