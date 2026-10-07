"""
evaluate.py
-----------
Evaluate the fine-tuned DistilBERT model: F1, accuracy, confusion matrix.
Creators: Bilal Jellaoui, Mohammed Azil, Ayoube Echihami

Usage:
    python ml/evaluate.py
"""
"""
evaluate.py - Evaluate trained Intent & Emotion models on test sets.
Usage:
    python ml/evaluate.py --task intent
    python ml/evaluate.py --task emotion
    python ml/evaluate.py --task both
"""

"""
evaluate.py - Evaluate trained Intent & Emotion models on test sets.
Usage:
    python ml/evaluate.py --task intent
    python ml/evaluate.py --task emotion
    python ml/evaluate.py --task both
"""

import os
import argparse
import pickle
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR     = os.path.join(BASE_DIR, "ml")
TOK_DIR    = os.path.join(ML_DIR,   "tokenizer")
MODELS_DIR = os.path.join(ML_DIR,   "models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Shared Dataset ───────────────────────────────────────────────────────────

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tok    = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
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


# ─── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate_model(task: str):
    assert task in ("intent", "emotion")

    prefix     = "intent"       if task == "intent" else "emotion"
    max_len    = 64             if task == "intent" else 128
    model_dir  = os.path.join(MODELS_DIR, "distilbert_model" if task == "intent" else "emotion_model")
    enc_path   = os.path.join(TOK_DIR, f"{prefix}_label_encoder.pkl")
    test_path  = os.path.join(TOK_DIR, f"{prefix}_test.csv")

    if not os.path.exists(model_dir):
        print(f"[{task.upper()}] Model not found at {model_dir}. Run train.py first.")
        return

    print(f"\n{'='*60}")
    print(f"  Evaluating {task.upper()} model")
    print(f"{'='*60}")

    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model     = DistilBertForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
    model.eval()

    with open(enc_path, "rb") as f:
        label_enc = pickle.load(f)

    test_df = pd.read_csv(test_path)
    dataset = EvalDataset(test_df, tokenizer, max_len)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"].to(DEVICE)
            mask   = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(input_ids=ids, attention_mask=mask).logits
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())

    acc  = accuracy_score(trues, preds)
    f1   = f1_score(trues, preds, average="weighted")

    # Decode labels
    true_names = label_enc.inverse_transform(trues)
    pred_names = label_enc.inverse_transform(preds)

    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Weighted F1   : {f1:.4f}")
    print(f"\n--- Classification Report ---")
    print(classification_report(true_names, pred_names, zero_division=0))

    # Confusion matrix (compact)
    cm = confusion_matrix(trues, preds)
    print(f"Confusion matrix shape: {cm.shape}")

    # Save report
    report = classification_report(true_names, pred_names, zero_division=0, output_dict=True)
    report["accuracy"] = acc
    report["f1_weighted"] = f1

    report_path = os.path.join(ML_DIR, f"{prefix}_eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Report saved → {report_path}")

    return acc, f1


# ─── Inference Demo ───────────────────────────────────────────────────────────

def predict_single(text: str, task: str) -> dict:
    """Quick inference test for a single text."""
    prefix    = "intent"       if task == "intent" else "emotion"
    max_len   = 64             if task == "intent" else 128
    model_dir = os.path.join(MODELS_DIR, "distilbert_model" if task == "intent" else "emotion_model")
    enc_path  = os.path.join(TOK_DIR, f"{prefix}_label_encoder.pkl")

    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model     = DistilBertForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
    model.eval()

    with open(enc_path, "rb") as f:
        label_enc = pickle.load(f)

    enc = tokenizer(text, max_length=max_len, padding="max_length",
                    truncation=True, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_idx   = int(np.argmax(probs))
    label     = label_enc.inverse_transform([top_idx])[0]
    confidence = float(probs[top_idx])

    return {"label": label, "confidence": confidence, "text": text}


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["intent", "emotion", "both"], default="both")
    parser.add_argument("--demo", type=str, default=None,
                        help="Run a quick inference on this text string")
    args = parser.parse_args()

    if args.demo:
        for t in (["intent", "emotion"] if args.task == "both" else [args.task]):
            result = predict_single(args.demo, t)
            print(f"\n[{t.upper()}] '{result['text']}'")
            print(f"  → {result['label']} (confidence: {result['confidence']:.2%})")
    else:
        results = {}
        if args.task in ("intent", "both"):
            r = evaluate_model("intent")
            if r: results["intent"] = {"accuracy": r[0], "f1": r[1]}

        if args.task in ("emotion", "both"):
            r = evaluate_model("emotion")
            if r: results["emotion"] = {"accuracy": r[0], "f1": r[1]}

        if results:
            print("\n\n📊 Summary:")
            for task, metrics in results.items():
                print(f"  {task.upper():8s} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")