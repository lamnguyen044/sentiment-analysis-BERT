"""Interpret BiLSTM + attention: colour words by their attention weight."""
import html

import numpy as np
import torch
from IPython.display import HTML, display

from .config import DEVICE, LABELS
from .text import clean_text, tokenize


@torch.inference_mode()
def lstm_explain(model, vocab, text, max_len):
    """Returns (tokens, attention weights, 3-class probabilities) for one sentence."""
    model.eval()
    toks = tokenize(clean_text(text))[:max_len] or ['<unk>']
    ids = vocab.encode(toks, max_len)
    logits, alpha = model(torch.tensor([ids], device=DEVICE), return_attn=True)
    return toks, alpha[0].cpu().numpy(), torch.softmax(logits[0], -1).cpu().numpy()


def lstm_predict_texts(model, vocab, texts, max_len) -> np.ndarray:
    return np.stack([lstm_explain(model, vocab, t, max_len)[2] for t in texts])


def show_attention(model, vocab, text, max_len, true_label=None):
    toks, alpha, probs = lstm_explain(model, vocab, text, max_len)
    a = alpha / alpha.max()
    spans = ' '.join(f'<span style="background-color: rgba(220,50,50,{w:.2f}); padding:1px 2px; border-radius:3px">'
                     f'{html.escape(t)}</span>' for t, w in zip(toks, a))
    head = f'<b>Predicted:</b> {LABELS[probs.argmax()]} ({probs.max():.0%})'
    if true_label is not None:
        head += f' &nbsp;|&nbsp; <b>Actual:</b> {LABELS[true_label]}'
    display(HTML(f'<div style="margin:6px 0 14px 0; line-height:1.9">{head}<br>{spans}</div>'))
