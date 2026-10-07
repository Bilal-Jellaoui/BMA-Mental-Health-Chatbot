"""
app/services/emotion_detector.py - Emotion classification using fine-tuned DistilBERT
"""

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pickle
from config.config import Config
from app.utils.text_cleaner import clean_text

class EmotionDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self._load_model()

    def _load_model(self):
        model_dir = Config.EMOTION_MODEL_DIR  # Assure-toi que cette variable pointe vers ml/models/emotion_model
        encoder_path = Config.EMOTION_ENCODER_PATH  # chemin vers emotion_label_encoder.pkl

        if not model_dir.exists() or not encoder_path.exists():
            raise RuntimeError("Emotion model not found. Train first or check paths.")

        self.tokenizer = DistilBertTokenizer.from_pretrained(str(model_dir))
        self.model = DistilBertForSequenceClassification.from_pretrained(str(model_dir)).to(self.device)
        self.model.eval()

        with open(encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

    def predict(self, text: str) -> dict:
        if not text.strip():
            return {"emotion": "neutral", "confidence": 0.0}
        clean = clean_text(text)
        inputs = self.tokenizer(clean, return_tensors="pt", truncation=True, padding=True, max_length=64)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            idx = probs.argmax()
            emotion = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(probs[idx])
        return {"emotion": emotion, "confidence": confidence}