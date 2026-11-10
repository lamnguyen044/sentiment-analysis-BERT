"""Streamlit demo: sentiment of a food review with the two trained models (BiLSTM + attention, fine-tuned BERT).

Run from the project folder:   streamlit run app.py
Needs the saved models in outputs/bilstm/ and outputs/bert/ (created by notebooks 04 and 05).
"""
import html
import json

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import BERT_DIR, BILSTM_DIR, CFG, DEVICE, LABELS, ON_GPU
from src.data import Vocab
from src.explain import lstm_explain
from src.models import BiLSTMAttention
from src.text import clean_text
from src.transformer import predict_transformer

COLORS = {'negative': '#c0392b', 'neutral': '#7f7f7f', 'positive': '#2e8b57'}
EMOJI = {'negative': '😞', 'neutral': '😐', 'positive': '😊'}
BILSTM, BERT = 'BiLSTM + attention', 'BERT-base (fine-tuned)'
EXAMPLES = {
    'Clearly positive': "This coffee is absolutely amazing, I will definitely buy it again!",
    'Clearly negative': "I would not recommend this to anyone. Total waste of money.",
    'Neutral, lukewarm': "It's okay. Nothing special, but it does the job.",
    'Mixed: praise and complaint': "My dog loves these treats, but they are way too expensive for such a small bag.",
    'Negation ("not bad")': "Not bad at all, actually pretty tasty.",
    'Sarcasm (hard)': "Great, another box arrived completely crushed. Just what I needed.",
}

st.set_page_config(page_title='Food review sentiment', page_icon='🍽️', layout='wide')


# ------------------------------------------------------------------ models
@st.cache_resource(show_spinner='Loading models (first time only)...')
def load_models():
    needed = [BILSTM_DIR / 'model.pt', BILSTM_DIR / 'vocab.json', BERT_DIR / 'config.json', BERT_DIR / 'model.safetensors']
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        return None, missing
    vocab = Vocab.load(BILSTM_DIR / 'vocab.json')
    bilstm = BiLSTMAttention(len(vocab), CFG['lstm_emb_dim'], CFG['lstm_hidden'], CFG['lstm_layers'], 3,
                             CFG['lstm_dropout']).to(DEVICE)
    bilstm.load_state_dict(torch.load(BILSTM_DIR / 'model.pt', map_location=DEVICE))
    bilstm.eval()
    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    bert = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).to(DEVICE).eval()
    info = BERT_DIR / 'train_info.json'
    max_len = json.loads(info.read_text()).get('max_len', 256) if info.exists() else 256
    return {'vocab': vocab, 'bilstm': bilstm, 'tokenizer': tokenizer, 'bert': bert, 'bert_max_len': max_len}, []


def predict_bilstm(models, text):
    tokens, alpha, probs = lstm_explain(models['bilstm'], models['vocab'], text, CFG['lstm_max_len'])
    return probs, (tokens, alpha)


def predict_bert(models, text):
    return predict_transformer(models['bert'], models['tokenizer'], [clean_text(text)], models['bert_max_len'],
                               batch_size=1, desc='BERT')[0]


# ------------------------------------------------------------------ rendering
def prob_bars(probs):
    rows = []
    for label, p in zip(LABELS, probs):
        rows.append(
            f'<div style="display:flex;align-items:center;margin:4px 0">'
            f'<div style="width:78px">{label}</div>'
            f'<div style="flex:1;background:rgba(128,128,128,0.25);border-radius:4px">'
            f'<div style="width:{p * 100:.1f}%;background:{COLORS[label]};height:16px;border-radius:4px"></div></div>'
            f'<div style="width:64px;text-align:right;font-variant-numeric:tabular-nums">{p:.1%}</div></div>')
    return ''.join(rows)


def verdict(probs):
    k = int(np.argmax(probs))
    label = LABELS[k]
    return (f'<div style="font-size:1.6rem;font-weight:700;color:{COLORS[label]}">{EMOJI[label]} {label.upper()}'
            f'<span style="font-size:1rem;font-weight:400;opacity:.8"> &nbsp;{probs[k]:.0%} confidence</span></div>')


def attention_html(tokens, alpha):
    a = alpha / alpha.max()
    spans = ' '.join(f'<span style="background-color: rgba(220,50,50,{w:.2f}); padding:1px 3px; border-radius:3px">'
                     f'{html.escape(t)}</span>' for t, w in zip(tokens, a))
    return f'<div style="line-height:2">{spans}</div>'


def show_result(name, probs, attention=None, show_attention=False):
    st.markdown(f'#### {name}')
    st.markdown(verdict(probs), unsafe_allow_html=True)
    st.markdown(prob_bars(probs), unsafe_allow_html=True)
    if attention is not None and show_attention:
        st.caption('Words the BiLSTM attends to (darker = larger attention weight)')
        st.markdown(attention_html(*attention), unsafe_allow_html=True)


# ------------------------------------------------------------------ page
st.title('🍽️ Food review sentiment')
st.caption('Three classes: **negative / neutral / positive** (labels come from star ratings: 1-2, 3, 4-5). '
           'Two neural models trained on 393,577 Amazon Fine Food reviews.')

models, missing = load_models()
if models is None:
    st.error('Saved models not found. Run notebooks 04 and 05 first (or copy `outputs/bilstm/` and `outputs/bert/` here).')
    st.code('\n'.join(missing))
    st.stop()

with st.sidebar:
    st.header('Options')
    use_bilstm = st.checkbox(BILSTM, value=True)
    use_bert = st.checkbox(BERT, value=True)
    show_att = st.checkbox('Show BiLSTM attention words', value=True)
    st.divider()
    st.subheader('Test-set results')
    st.dataframe(pd.DataFrame({'Macro-F1': [0.7877, 0.7422], 'Accuracy': [0.9074, 0.8788]}, index=['BERT-base', 'BiLSTM']),
                 width='stretch')
    st.caption(f'Running on **{"GPU" if ON_GPU else "CPU"}**')

if not (use_bilstm or use_bert):
    st.warning('Select at least one model in the sidebar.')
    st.stop()

tab_one, tab_batch, tab_about = st.tabs(['Single review', 'Batch (one review per line)', 'About'])

with tab_one:
    def _load_example():
        choice = st.session_state.get('example')
        if choice in EXAMPLES:
            st.session_state['review_text'] = EXAMPLES[choice]

    st.selectbox('Try an example', ['(type your own)'] + list(EXAMPLES), key='example', on_change=_load_example)
    text = st.text_area('Review text (English)', key='review_text', height=130,
                        placeholder='e.g. The chips were stale and the bag was half empty.')
    go = st.button('Analyze', type='primary')

    if go or text.strip():
        if not text.strip():
            st.warning('Please type a review first.')
        else:
            cols = st.columns(int(use_bilstm) + int(use_bert))
            results = {}
            i = 0
            if use_bilstm:
                probs, attention = predict_bilstm(models, text)
                results[BILSTM] = probs
                with cols[i]:
                    show_result(BILSTM, probs, attention, show_att)
                i += 1
            if use_bert:
                probs = predict_bert(models, text)
                results[BERT] = probs
                with cols[i]:
                    show_result(BERT, probs)
            if len(results) == 2:
                labels = {k: LABELS[int(np.argmax(v))] for k, v in results.items()}
                if len(set(labels.values())) > 1:
                    st.info('The two models disagree on this review: ' + ', '.join(f'{k} says **{v}**' for k, v in labels.items()) + '.')
            if len(text.split()) > CFG['lstm_max_len']:
                st.caption(f'Long review: the BiLSTM reads only the first {CFG["lstm_max_len"]} words and BERT the first {models["bert_max_len"]} tokens.')

with tab_batch:
    st.write('Paste several reviews, **one per line**. Each line is scored separately.')
    batch = st.text_area('Reviews', key='batch_text', height=170,
                         placeholder='Great taste, will buy again\nArrived broken and late\nIt is fine, nothing special')
    if st.button('Analyze all lines', key='batch_go'):
        lines = [ln.strip() for ln in batch.split('\n') if ln.strip()]
        if not lines:
            st.warning('Please paste at least one review.')
        else:
            rows = []
            bar = st.progress(0.0)
            for n, ln in enumerate(lines, 1):
                row = {'Review': ln}
                if use_bilstm:
                    p, _ = predict_bilstm(models, ln)
                    row['BiLSTM'] = f'{LABELS[int(np.argmax(p))]} ({p.max():.0%})'
                if use_bert:
                    p = predict_bert(models, ln)
                    row['BERT'] = f'{LABELS[int(np.argmax(p))]} ({p.max():.0%})'
                rows.append(row)
                bar.progress(n / len(lines))
            table = pd.DataFrame(rows)
            st.dataframe(table, width='stretch', hide_index=True)
            st.download_button('Download CSV', table.to_csv(index=False).encode('utf-8'), 'predictions.csv', 'text/csv')

with tab_about:
    st.markdown(f"""
**Task.** Classify the sentiment of an English food review as negative, neutral or positive.
The labels are derived from the star rating, so they are *weak labels*: a 5-star review can still contain a complaint.

**Models** (both trained on the same 314,861 training reviews, class-weighted loss):

| Model | Parameters | Test Macro-F1 | Test accuracy |
|---|---|---|---|
| BERT-base, fine-tuned | 109.48 M | **0.7877** | 0.9074 |
| BiLSTM + attention, from scratch | 6.25 M | 0.7422 | 0.8788 |
| Always answer *positive* | - | 0.2920 | 0.7794 |

**Things to know**
- Neutral is the hardest class (F1 about 0.49 to 0.55): reviews that praise and complain at once often land there.
- Sarcasm is usually misread (e.g. *"Great, another box arrived crushed"* is read as positive).
- Trained on Amazon food reviews from 1999-2012; other domains may work worse.
- BERT reads up to {models['bert_max_len']} tokens, the BiLSTM up to {CFG['lstm_max_len']} words.
""")
