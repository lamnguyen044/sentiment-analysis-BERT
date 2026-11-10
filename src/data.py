"""Intermediate data I/O, vocabulary, Dataset and DataLoader with length bucketing."""
import json
import math
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import DATA_DIR, SEED

CLEAN_PATH = DATA_DIR / 'reviews_clean.pkl'
SPLIT_PATHS = {name: DATA_DIR / f'{name}.pkl' for name in ('train', 'val', 'test')}
CLASS_WEIGHTS_PATH = DATA_DIR / 'class_weights.json'


def load_clean() -> pd.DataFrame:
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f'{CLEAN_PATH} not found. Run notebook 01_data_cleaning first.')
    return pd.read_pickle(CLEAN_PATH)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [p for p in SPLIT_PATHS.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f'{missing[0]} not found. Run notebook 03_data_preparation first.')
    return tuple(pd.read_pickle(SPLIT_PATHS[n]) for n in ('train', 'val', 'test'))


def load_class_weights() -> np.ndarray:
    return np.array(json.loads(CLASS_WEIGHTS_PATH.read_text(encoding='utf-8'))['weights'], dtype=np.float32)


def stratified_subset(frame: pd.DataFrame, n, seed=SEED) -> pd.DataFrame:
    """Take n rows stratified by label (n = None or n >= number of rows: keep all)."""
    from sklearn.model_selection import train_test_split
    if n is None or n >= len(frame):
        return frame.reset_index(drop=True)
    return train_test_split(frame, train_size=n, stratify=frame['label'], random_state=seed)[0].reset_index(drop=True)


PAD_IDX, UNK_IDX = 0, 1


class Vocab:
    """Token <-> id mapping. id 0 = <pad>, id 1 = <unk>."""

    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    @classmethod
    def build(cls, token_lists, max_size: int, min_freq: int):
        counter = Counter(tok for toks in token_lists for tok in toks)
        itos = ['<pad>', '<unk>'] + [w for w, c in counter.most_common(max_size - 2) if c >= min_freq]
        vocab = cls(itos)
        vocab.counter = counter
        return vocab

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: list[str], max_len: int) -> list[int]:
        ids = [self.stoi.get(t, UNK_IDX) for t in tokens[:max_len]]
        return ids if ids else [UNK_IDX]

    def save(self, path):
        path.write_text(json.dumps(self.itos, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def load(cls, path):
        return cls(json.loads(path.read_text(encoding='utf-8')))


class SeqDataset(Dataset):
    """Returns (index, id sequence, label). The index lets bucketed inference write probabilities back to the right row."""

    def __init__(self, ids, labels):
        self.ids, self.labels = ids, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return i, self.ids[i], int(self.labels[i])


class BucketBatchSampler(Sampler):
    """Group reviews of similar length into the same batch to reduce padding.

    shuffle=True : shuffle -> split into chunks -> sort by length inside each chunk -> cut batches -> shuffle batches.
    shuffle=False: sort everything by length (used for inference).
    """

    def __init__(self, lengths, batch_size, shuffle=True, bucket_mult=50, seed=SEED):
        self.lengths, self.bs, self.shuffle = np.asarray(lengths), batch_size, shuffle
        self.chunk = batch_size * bucket_mult if shuffle else len(lengths)
        self.rng = np.random.default_rng(seed)

    def _batches(self):
        idx = self.rng.permutation(len(self.lengths)) if self.shuffle else np.arange(len(self.lengths))
        batches = []
        for s in range(0, len(idx), self.chunk):
            chunk = idx[s:s + self.chunk]
            chunk = chunk[np.argsort(self.lengths[chunk], kind='stable')]
            batches += [chunk[j:j + self.bs] for j in range(0, len(chunk), self.bs)]
        if self.shuffle:
            batches = [batches[k] for k in self.rng.permutation(len(batches))]
        return batches

    def __iter__(self):
        for b in self._batches():
            yield b.tolist()

    def __len__(self):
        return sum(math.ceil(min(self.chunk, len(self.lengths) - s) / self.bs)
                   for s in range(0, len(self.lengths), self.chunk))


def collate_lstm(batch):
    idx, seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    x = torch.full((len(seqs), int(lengths.max())), PAD_IDX, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return torch.tensor(idx), x, lengths, torch.tensor(labels)


def make_collate_transformer(pad_token_id: int):
    def collate(batch):
        idx, seqs, labels = zip(*batch)
        max_len = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = torch.tensor(s)
            attention_mask[i, :len(s)] = 1
        return torch.tensor(idx), input_ids, attention_mask, torch.tensor(labels)
    return collate


def make_loader(ids, labels, batch_size, shuffle, collate_fn, seed=SEED):
    return DataLoader(SeqDataset(ids, labels),
                      batch_sampler=BucketBatchSampler([len(s) for s in ids], batch_size, shuffle=shuffle, seed=seed),
                      collate_fn=collate_fn)
