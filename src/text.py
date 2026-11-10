"""Text cleaning and tokenisation."""
import html
import re

import numpy as np
import pandas as pd


def clean_text(text: str) -> str:
    """Light cleaning: remove HTML and URLs, normalise whitespace. Keeps punctuation, case and negations."""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_input_text(summary: pd.Series, body: pd.Series) -> np.ndarray:
    """text = clean(Summary) + '. ' + clean(Text)  (no period is added if Summary already ends with . ! ?)."""
    summary_clean = summary.fillna('').astype(str).map(clean_text)
    body_clean = body.astype(str).map(clean_text)
    sep = np.where(summary_clean.str[-1:].isin(['.', '!', '?']), ' ', '. ')
    return np.where(summary_clean.str.len() > 0, summary_clean + sep + body_clean, body_clean)


TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[!?]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


NEGATIONS = {'not', 'no', 'nor', 'never', 'don', "don't", 'didn', "didn't", 'doesn', "doesn't", 'isn', "isn't",
             'wasn', "wasn't", 'weren', "weren't", 'wouldn', "wouldn't", 'couldn', "couldn't", 'won', "won't",
             'aren', "aren't", 'hasn', "hasn't", 'haven', "haven't", 'shouldn', "shouldn't"}
