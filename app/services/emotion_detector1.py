import pickle
import re
from config.config import Config

class EmotionDetector:
    def __init__(self):
        with open(Config.EMOTION_MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.model = data["classifier"]

    def predict(self, text):
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        X = self.vectorizer.transform([text_clean])
        probs = self.model.predict_proba(X)[0]
        idx = probs.argmax()
        emotion = self.model.classes_[idx]
        confidence = float(probs[idx])
        return {"emotion": emotion, "confidence": confidence}