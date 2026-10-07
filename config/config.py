"""
config.py - Central configuration for the BMA Mental Health Chatbot.
Creators: Bilal Jellaoui, Mohammed Azil, Ayoube Echihami
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "bma-dev-secret")

    # ── LLM Primary: Groq ─────────────────────────────────────────────────────
    GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")  # modifiable via .env

    # ── LLM Fallback: Gemini ──────────────────────────────────────────────────
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash")

    # ── Database ───────────────────────────────────────────────────────────────
    BASE_DIR = Path(__file__).parent.parent
    DATABASE_PATH = os.getenv(
        "DB_PATH", str(BASE_DIR / "database" / "mental_health.db")
    )

    # ── ML model paths ─────────────────────────────────────────────────────────
    ML_DIR = BASE_DIR / "ml"
    TOK_DIR = ML_DIR / "tokenizer"
    MODELS_DIR = ML_DIR / "models"
    INTENT_MODEL_DIR    = MODELS_DIR / "distilbert_model"
    EMOTION_MODEL_PATH  = MODELS_DIR / "emotion_model" / "model.pkl"
    EMOTION_MODEL_DIR   = MODELS_DIR / "emotion_model"
    INTENT_ENCODER_PATH = TOK_DIR / "intent_label_encoder.pkl"
    EMOTION_ENCODER_PATH= TOK_DIR / "emotion_label_encoder.pkl"
    INTENTS_LOOKUP_PATH = TOK_DIR / "intents_lookup.pkl"

    DATA_DIR      = BASE_DIR / "data"
    RESPONSES_PATH= DATA_DIR / "responses.json"
    INTENTS_PATH  = DATA_DIR / "intents.json"

    # ── Flask ──────────────────────────────────────────────────────────────────
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    HOST  = "0.0.0.0"
    PORT  = int(os.getenv("PORT", 5000))

    # ── Hybrid threshold ───────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
    MAX_HISTORY_TURNS    = int(os.getenv("MAX_HISTORY_TURNS", "10"))

    # ── Language ───────────────────────────────────────────────────────────────
    SUPPORTED_LANGUAGES = ["en", "fr", "ar"]
    DEFAULT_LANGUAGE    = "en"

    # ── Crisis ────────────────────────────────────────────────────────────────
    CRISIS_KEYWORDS = [
        "suicide", "kill myself", "end my life", "want to die",
        "self harm", "hurt myself", "don't want to live", "end it all",
        "take my life", "no reason to live",
        "me suicider", "mettre fin à ma vie", "je veux mourir",
        "je veux me blesser",
        "انتحار", "أريد الموت", "إنهاء حياتي", "إيذاء نفسي",
    ]
    CRISIS_RESPONSE = (
        "I'm really concerned about your safety right now. "
        "Please reach out immediately:\n"
        "• Morocco: 15 or 3114\n"
        "• US: 988\n"
        "• International: befrienders.org\n"
        "You are not alone. Help is available."
    )