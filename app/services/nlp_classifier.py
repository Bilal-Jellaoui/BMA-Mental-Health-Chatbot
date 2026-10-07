"""
app/models/nlp_classifier.py
-----------------------------
DistilBERT intent classifier wrapper for BMA.
Loads the fine-tuned model from app/models/saved_model/
and provides fast offline intent classification.
Creators: Bilal Jellaoui, Mohammed Azil, Ayoube Echihami
"""
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pickle
from config.config import Config

class NLPClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self._load_model()

    def _load_model(self):
        if not Config.INTENT_MODEL_DIR.exists() or not Config.INTENT_ENCODER_PATH.exists():
            raise RuntimeError("Intent model not found. Run ml/train.py first.")
        self.tokenizer = DistilBertTokenizer.from_pretrained(str(Config.INTENT_MODEL_DIR))
        self.model = DistilBertForSequenceClassification.from_pretrained(str(Config.INTENT_MODEL_DIR)).to(self.device)
        self.model.eval()
        with open(Config.INTENT_ENCODER_PATH, "rb") as f:
            self.label_encoder = pickle.load(f)

    def predict(self, text):
        if not text.strip():
            return {"intent": "neutral-response", "confidence": 0.0}
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            idx = probs.argmax()
            intent = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(probs[idx])
        return {"intent": intent, "confidence": confidence}