"""Inference and fine-tuning of Transformer models (BERT) with HuggingFace."""
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from .config import DEVICE, LABELS, ON_GPU


@torch.inference_mode()
def predict_transformer(model, tokenizer, texts, max_len, batch_size=64, desc='Inference'):
    """Predict probabilities with a HuggingFace model; sorted by length so dynamic padding is cheap."""
    model.eval()
    texts = list(texts)
    order = np.argsort([len(t) for t in texts])
    probs = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    for i in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        idx = order[i:i + batch_size]
        enc = tokenizer([texts[j] for j in idx], truncation=True, max_length=max_len,
                        padding=True, return_tensors='pt').to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=ON_GPU):
            logits = model(**enc).logits
        probs[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return probs


def load_classifier(path_or_name, num_labels=None):
    """Load a classifier; when loading from the original name (not fine-tuned yet), declare the three labels of the project."""
    kwargs = {} if num_labels is None else dict(num_labels=num_labels, id2label=dict(enumerate(LABELS)),
                                                label2id={l: i for i, l in enumerate(LABELS)})
    return AutoModelForSequenceClassification.from_pretrained(path_or_name, **kwargs).to(DEVICE)


def build_optimizer(model, lr, weight_decay, warmup, total_steps):
    """AdamW (weight decay only on weight matrices, not on biases and LayerNorm) + linear warm-up then linear decay."""
    no_decay = ['bias', 'LayerNorm.weight']
    groups = [
        {'params': [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup * total_steps), total_steps)
    return optimizer, scheduler


def fit_transformer(model, tokenizer, train_loader, val_texts, y_val, criterion, epochs, lr, warmup,
                    weight_decay, max_len, save_dir):
    """Fine-tune the whole network. After each epoch, score validation Macro-F1 and save at once if it is the best epoch.

    No early stopping: always runs all `epochs`. Returns (model reloaded from the best epoch, history per epoch,
    loss per step, best validation Macro-F1, training time in seconds).
    """
    total_steps = len(train_loader) * epochs
    optimizer, scheduler = build_optimizer(model, lr, weight_decay, warmup, total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=ON_GPU)
    history, step_losses, best_f1 = [], [], -1.0
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        for _, input_ids, attention_mask, y in tqdm(train_loader, desc=f'epoch {epoch}', leave=False):
            input_ids, attention_mask, y = input_ids.to(DEVICE), attention_mask.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=ON_GPU):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step_losses.append(loss.item())

        val_pred = predict_transformer(model, tokenizer, val_texts, max_len, desc='validation').argmax(1)
        val_acc, val_f1 = accuracy_score(y_val, val_pred), f1_score(y_val, val_pred, average='macro')
        epoch_loss = float(np.mean(step_losses[-len(train_loader):]))
        history.append({'epoch': epoch, 'train_loss': epoch_loss, 'val_acc': val_acc, 'val_macro_f1': val_f1})
        print(f'Epoch {epoch}: train_loss={epoch_loss:.4f} | val_acc={val_acc:.4f} | '
              f'val_macro_F1={val_f1:.4f} | {(time.time() - t0) / 60:.1f} min')
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
        pd.DataFrame(history).to_csv(save_dir / 'history.csv', index=False)
        if ON_GPU:
            torch.cuda.empty_cache()
    train_time = time.time() - t_start
    model = load_classifier(save_dir)
    np.save(save_dir / 'step_losses.npy', np.array(step_losses))
    (save_dir / 'train_info.json').write_text(json.dumps({'train_time_s': train_time, 'best_val_macro_f1': best_f1, 'max_len': max_len}, indent=2))
    return model, pd.DataFrame(history), step_losses, best_f1, train_time
