"""Shared configuration: paths, labels, hyperparameters, seed, device."""
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'dataset' / 'Reviews.csv'
OUTPUT_DIR = ROOT / 'outputs'
DATA_DIR = OUTPUT_DIR / 'data'
PRED_DIR = OUTPUT_DIR / 'predictions'
BILSTM_DIR = OUTPUT_DIR / 'bilstm'
BERT_DIR = OUTPUT_DIR / 'bert'
for d in (DATA_DIR, PRED_DIR, BILSTM_DIR, BERT_DIR):
    d.mkdir(parents=True, exist_ok=True)

LABELS = ['negative', 'neutral', 'positive']
SCORE_TO_LABEL = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
PALETTE = {'negative': '#d62728', 'neutral': '#7f7f7f', 'positive': '#2ca02c'}

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ON_GPU = DEVICE.type == 'cuda'

QUICK_RUN = os.environ.get('QUICK_RUN', '0') == '1'

CFG = {
    'sample_size': 30_000 if QUICK_RUN else None,
    'lstm_max_vocab': 50_000, 'lstm_min_freq': 3, 'lstm_max_len': 200,
    'lstm_emb_dim': 128, 'lstm_hidden': 128, 'lstm_layers': 1, 'lstm_dropout': 0.3,
    'lstm_batch_size': 256, 'lstm_epochs': 1 if QUICK_RUN else 5, 'lstm_lr': 2e-3, 'lstm_patience': 2,
    'trf_name': 'bert-base-uncased',
    'trf_max_len': 256 if ON_GPU else 128,
    'trf_train_size': None if ON_GPU else (800 if QUICK_RUN else 20_000),
    'trf_val_size': None if ON_GPU else (300 if QUICK_RUN else 4_000),
    'trf_batch_size': 32 if ON_GPU else 16,
    'trf_epochs': 1 if QUICK_RUN else 2, 'trf_lr': 3e-5, 'trf_warmup': 0.1, 'trf_weight_decay': 0.01,
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_info() -> str:
    if ON_GPU:
        return f'PyTorch {torch.__version__} | device: cuda ({torch.cuda.get_device_name(0)})'
    return f'PyTorch {torch.__version__} | device: cpu ({torch.get_num_threads()} threads)'


def setup_notebook() -> None:
    """Common notebook setup: plot style, silence warnings, seed."""
    import warnings
    import matplotlib.pyplot as plt
    import pandas as pd
    warnings.filterwarnings('ignore')
    plt.style.use('ggplot')
    pd.set_option('display.max_colwidth', 200)
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except ImportError:
        pass
    set_seed(SEED)
    print(device_info() + (' | QUICK_RUN' if QUICK_RUN else ''))
