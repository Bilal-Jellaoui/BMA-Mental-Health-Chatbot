"""
train.py
--------
Fine-tune DistilBERT on the BMA intent classification dataset.
Creators: Bilal Jellaoui, Mohammed Azil, Ayoube Echihami"""

"""
train.py - Train DistilBERT Intent & Emotion classifiers
Usage:
    python ml/train.py --task intent
    python ml/train.py --task emotion
    python ml/train.py --task both
"""

"""
train.py - Train DistilBERT Intent & Emotion classifiers
Usage:
    python ml/train.py --task intent
    python ml/train.py --task emotion
    python ml/train.py --task both
"""

import os
import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR     = os.path.join(BASE_DIR, "ml")
TOK_DIR    = os.path.join(ML_DIR,   "tokenizer")
MODELS_DIR = os.path.join(ML_DIR,   "models")
INTENT_MODEL_DIR  = os.path.join(MODELS_DIR, "distilbert_model")
EMOTION_MODEL_DIR = os.path.join(MODELS_DIR, "emotion_model")
os.makedirs(INTENT_MODEL_DIR,  exist_ok=True)
os.makedirs(EMOTION_MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {DEVICE}")

# ─── HyperParams ──────────────────────────────────────────────────────────────
INTENT_CONFIG = {
    "max_len": 64,
    "batch_size": 32,
    "epochs": 5,
    "lr": 2e-5,
}
EMOTION_CONFIG = {
    "max_len": 128,
    "batch_size": 16,
    "epochs": 4,
    "lr": 2e-5,
}

# ─── Dataset ──────────────────────────────────────────────────────────────────

class MentalHealthDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─── Trainer ──────────────────────────────────────────────────────────────────

def train_model(task: str):
    """Train intent or emotion classifier and save to disk."""
    assert task in ("intent", "emotion")

    cfg        = INTENT_CONFIG  if task == "intent"  else EMOTION_CONFIG
    model_dir  = INTENT_MODEL_DIR if task == "intent" else EMOTION_MODEL_DIR
    prefix     = "intent"       if task == "intent"  else "emotion"

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tok_path   = os.path.join(TOK_DIR, "distilbert_tokenizer")
    tokenizer  = DistilBertTokenizer.from_pretrained(tok_path)

    # ── Load splits ───────────────────────────────────────────────────────────
    train_df = pd.read_csv(os.path.join(TOK_DIR, f"{prefix}_train.csv"))
    val_df   = pd.read_csv(os.path.join(TOK_DIR, f"{prefix}_val.csv"))

    num_labels = train_df["label"].nunique()
    print(f"\n[{task.upper()}] labels={num_labels} | train={len(train_df)} | val={len(val_df)}")

    train_ds = MentalHealthDataset(train_df, tokenizer, cfg["max_len"])
    val_ds   = MentalHealthDataset(val_df,   tokenizer, cfg["max_len"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=cfg["lr"])

    best_val_acc = 0.0

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']} [train]"):
            optimizer.zero_grad()
            ids   = batch["input_ids"].to(DEVICE)
            mask  = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids    = batch["input_ids"].to(DEVICE)
                mask   = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                logits = model(input_ids=ids, attention_mask=mask).logits
                preds.extend(logits.argmax(dim=1).cpu().numpy())
                trues.extend(labels.cpu().numpy())

        val_acc = accuracy_score(trues, preds)
        print(f"  Epoch {epoch} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"  ✅ Saved best model (acc={best_val_acc:.4f}) → {model_dir}")

    print(f"\n🎯 [{task.upper()}] Best Val Accuracy: {best_val_acc:.4f}")
    return best_val_acc


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["intent", "emotion", "both"], default="both")
    args = parser.parse_args()

    if args.task in ("intent", "both"):
        train_model("intent")

    if args.task in ("emotion", "both"):
        train_model("emotion")

    print("\n✅ Training complete!")