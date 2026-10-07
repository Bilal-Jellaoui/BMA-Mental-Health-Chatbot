"""
preprocess.py
-------------
Tokenization, data cleaning, and dataset preparation for BMA Mental Health Chatbot.
Creators: Bilal Jellaoui, Mohammed Azil, Ayoube Echihami
"""

"""
preprocess.py - Data Preprocessing for Mental Health Chatbot
Handles: intents.json, training_data.csv, emotions.csv
"""
"""
preprocess.py - Data Preprocessing for Mental Health Chatbot
Handles: intents.json, training_data.csv, emotions.csv
"""

"""
preprocess.py - Data Preprocessing for Mental Health Chatbot
Handles: intents.json, training_data.csv, emotions.csv
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
ML_DIR      = os.path.join(BASE_DIR, "ml")
TOK_DIR     = os.path.join(ML_DIR,   "tokenizer")
os.makedirs(TOK_DIR, exist_ok=True)

INTENTS_PATH   = os.path.join(DATA_DIR, "intents.json")
TRAINING_PATH  = os.path.join(DATA_DIR, "training_data.csv")
EMOTIONS_PATH  = os.path.join(DATA_DIR, "emotions.csv")

MAX_LEN_INTENT  = 64
MAX_LEN_EMOTION = 128
MODEL_NAME      = "distilbert-base-uncased"

# ─── 1. Intent Preprocessing ──────────────────────────────────────────────────

def load_intent_data():
    """Load and encode training data for intent classification."""
    df = pd.read_csv(TRAINING_PATH)
    df.dropna(subset=["text", "intent"], inplace=True)
    df["text"]   = df["text"].astype(str).str.strip().str.lower()
    df["intent"] = df["intent"].astype(str).str.strip()

    label_enc = LabelEncoder()
    df["label"] = label_enc.fit_transform(df["intent"])

    # Save encoder
    with open(os.path.join(TOK_DIR, "intent_label_encoder.pkl"), "wb") as f:
        pickle.dump(label_enc, f)

    print(f"[Intent] Loaded {len(df)} samples | {df['label'].nunique()} classes")
    return df, label_enc

def tokenize_intent(df, tokenizer):
    """Tokenize text for DistilBERT intent model."""
    encodings = tokenizer(
        df["text"].tolist(),
        max_length=MAX_LEN_INTENT,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encodings, df["label"].values

# ─── 2. Emotion Preprocessing (CORRIGÉE) ──────────────────────────────────────

EMOTION_LABELS = [
    "exam_stress", "joy", "anger", "depression", "burnout",
    "anxiety", "body_image_issue", "sadness", "stress",
    "loneliness", "low_self_esteem", "neutral",
]

def load_emotion_data(sample_per_class: int = 2000):
    """
    Load and balance emotion dataset.
    Caps each class at sample_per_class rows for training speed.
    """
    df = pd.read_csv(EMOTIONS_PATH)
    
    # Afficher les colonnes pour debug (peut être enlevé après)
    print("Colonnes du fichier emotions.csv :", df.columns.tolist())
    
    # Nettoyage
    df.dropna(subset=["text", "emotion"], inplace=True)
    df["text"]    = df["text"].astype(str).str.strip().str.lower()
    df["emotion"] = df["emotion"].astype(str).str.strip()
    
    # Filtrer les émotions inconnues (optionnel mais recommandé)
    df = df[df["emotion"].isin(EMOTION_LABELS)]
    
    # Équilibrer les classes
    balanced_dfs = []
    for emotion, group in df.groupby("emotion"):
        if len(group) > sample_per_class:
            group = group.sample(n=sample_per_class, random_state=42)
        balanced_dfs.append(group)
    df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Encodage des labels
    label_enc = LabelEncoder()
    label_enc.fit(EMOTION_LABELS)  # ordre fixe
    df["label"] = label_enc.transform(df["emotion"])
    
    # Sauvegarde de l'encodeur
    with open(os.path.join(TOK_DIR, "emotion_label_encoder.pkl"), "wb") as f:
        pickle.dump(label_enc, f)
    
    print(f"[Emotion] Loaded {len(df)} samples | {df['label'].nunique()} classes")
    print(df["emotion"].value_counts().to_string())
    return df, label_enc

def tokenize_emotion(df, tokenizer):
    """Tokenize text for DistilBERT emotion model."""
    encodings = tokenizer(
        df["text"].tolist(),
        max_length=MAX_LEN_EMOTION,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encodings, df["label"].values

# ─── 3. Shared Helpers ────────────────────────────────────────────────────────

def get_tokenizer():
    """Load (or download) the shared DistilBERT tokenizer and save locally."""
    tok_path = os.path.join(TOK_DIR, "distilbert_tokenizer")
    if os.path.exists(tok_path):
        tokenizer = DistilBertTokenizer.from_pretrained(tok_path)
        print("[Tokenizer] Loaded from cache.")
    else:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(tok_path)
        print(f"[Tokenizer] Downloaded and saved to {tok_path}")
    return tokenizer

def split_data(df, test_size=0.15, val_size=0.15, random_state=42):
    """Split DataFrame into train / val / test."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    relative_val = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=relative_val, random_state=random_state, stratify=train_df["label"]
    )
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df

def build_intents_lookup():
    """Return a dict { tag: [responses] } for the response engine."""
    with open(INTENTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    lookup = {intent["tag"]: intent["responses"] for intent in data["intents"]}
    save_path = os.path.join(TOK_DIR, "intents_lookup.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(lookup, f)
    print(f"[Intents] Lookup built → {len(lookup)} tags saved to {save_path}")
    return lookup

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("=" * 60)

    tokenizer = get_tokenizer()

    print("\n--- Intent Data ---")
    intent_df, intent_enc = load_intent_data()
    train_i, val_i, test_i = split_data(intent_df)

    for name, split in [("train", train_i), ("val", val_i), ("test", test_i)]:
        split.to_csv(os.path.join(TOK_DIR, f"intent_{name}.csv"), index=False)

    print("\n--- Emotion Data ---")
    emotion_df, emotion_enc = load_emotion_data()
    train_e, val_e, test_e = split_data(emotion_df)

    for name, split in [("train", train_e), ("val", val_e), ("test", test_e)]:
        split.to_csv(os.path.join(TOK_DIR, f"emotion_{name}.csv"), index=False)

    print("\n--- Intents Lookup ---")
    build_intents_lookup()

    print("\n✅ Preprocessing complete. Files saved to:", TOK_DIR)