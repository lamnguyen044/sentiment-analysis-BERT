"""Neural network architectures."""
import torch
import torch.nn as nn

from .data import PAD_IDX


class BiLSTMAttention(nn.Module):
    """Embedding -> Dropout -> (Bi)LSTM -> Attention pooling -> Dropout -> Linear.

    The default (bidirectional=True, use_attention=True) is the main model of the project.
    The two other options (one direction / mean pooling instead of attention) are not used in the final report.
    """

    def __init__(self, vocab_size, emb_dim, hidden, n_layers, n_classes, dropout, pad_idx=PAD_IDX,
                 bidirectional=True, use_attention=True):
        super().__init__()
        self.pad_idx = pad_idx
        self.use_attention = use_attention
        out_dim = hidden * (2 if bidirectional else 1)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=n_layers, batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if n_layers > 1 else 0.0)
        if use_attention:
            self.attn = nn.Linear(out_dim, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, n_classes))

    def forward(self, x, return_attn=False):
        pad = x == self.pad_idx
        emb = self.emb_dropout(self.embedding(x))
        out, _ = self.lstm(emb)
        if self.use_attention:
            scores = self.attn(out).squeeze(-1).masked_fill(pad, -1e9)
            alpha = torch.softmax(scores, dim=1)
        else:
            keep = (~pad).float()
            alpha = keep / keep.sum(dim=1, keepdim=True).clamp(min=1.0)
        context = torch.bmm(alpha.unsqueeze(1), out).squeeze(1)
        logits = self.classifier(context)
        return (logits, alpha) if return_attn else logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
