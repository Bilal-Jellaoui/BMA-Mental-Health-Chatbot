"""
chat_controller.py - Core chatbot logic (Hybrid ML + LLM + Multilingual).
Creators: Bilal Jellaoui, Mohammed Azil, Ayoube Echihami

Flow:
  1. Detect user language; translate to English for NLP
  2. Crisis detection (highest priority)
  3. Intent + Emotion classification (DistilBERT / sklearn)
  4. ResponseEngine: dataset response if confidence >= threshold
  5. LLM fallback (Gemini) when dataset has no confident match
  6. Translate response back to user's original language
  7. Persist conversation + emotion logs in SQLite
"""
"""
chat_controller.py - Core chatbot logic (Hybrid ML + LLM + Multilingual + Conversation ID)
Creators: Bilal Jellaoui, Mohammed Azil, Ayoube Echihami
"""

import logging
from app.services.safety import CrisisDetector
from app.services.nlp_classifier import NLPClassifier
from app.services.emotion_detector import EmotionDetector
from app.services.response_engine import ResponseEngine
from app.services.llm_service import LLMService
from app.services.memory import ConversationMemory
from app.utils.language_detector import normalize_to_english, localize_response
from config.config import Config

logger = logging.getLogger(__name__)


class ChatController:

    def __init__(self):
        self.crisis = CrisisDetector()
        self.intent = NLPClassifier()
        self.emotion = EmotionDetector()
        self.response_engine = ResponseEngine()
        self.llm = LLMService()
        self.memory = ConversationMemory()

    def process_message(self, user_id: int, message: str, conv_id: int = None, session_id: str = None) -> dict:
        """
        Traite un message utilisateur.
        Si conv_id est None, une nouvelle conversation sera créée automatiquement.
        """

        # 1. Détection langue + traduction en anglais
        english_message, user_lang = normalize_to_english(message)
        logger.info(f"user_lang={user_lang} | original='{message[:60]}' | en='{english_message[:60]}'")

        # 2. Détection de crise (priorité max)
        if self.crisis.is_crisis(english_message):
            crisis_resp_en = self.crisis.get_crisis_response()
            crisis_resp = localize_response(crisis_resp_en, user_lang)
            # Sauvegarde automatique (création conversation si conv_id absent)
            self.memory.save_conversation(user_id, message, crisis_resp, "crisis", "", "crisis",
                                          conv_id=conv_id, session_id=session_id)
            self.memory.save_safety_log(user_id, message, "HIGH", "Crisis response sent")
            return {
                "response": crisis_resp,
                "is_crisis": True,
                "intent": "crisis",
                "confidence": 1.0,
                "emotion": "",
                "source": "crisis",
                "language": user_lang,
            }

        # 3. Classification intent + émotion
        intent_res = self.intent.predict(english_message)
        emotion_res = self.emotion.predict(english_message)
        intent = intent_res["intent"]
        confidence = intent_res["confidence"]
        emotion = emotion_res["emotion"]
        self.memory.save_emotion_log(user_id, english_message, emotion, confidence)
        logger.debug(f"intent={intent} conf={confidence:.2f} emotion={emotion}")

        # 4. Réponse par dataset si confiance suffisante
        dataset_response, source = self.response_engine.get_response(intent, confidence)

        if source == "dataset" and dataset_response:
            final_response = localize_response(dataset_response, user_lang)
            logger.info(f"Dataset response used (intent={intent}, conf={confidence:.2f})")
        else:
            # 5. Fallback LLM
            history = self.memory.get_recent_conversations(user_id, limit=Config.MAX_HISTORY_TURNS)
            context = "\n".join([f"User: {h['user_message']}\nBMA: {h['bot_response']}" for h in history])
            llm_response_en = self.llm.generate_response(english_message, context)
            final_response = localize_response(llm_response_en, user_lang)
            source = "llm"
            logger.info(f"LLM response used (intent={intent}, conf={confidence:.2f})")

        # 6. Persistance (avec gestion automatique de conv_id)
        self.memory.save_conversation(user_id, message, final_response, intent, emotion, source,
                                      conv_id=conv_id, session_id=session_id)

        return {
            "response": final_response,
            "is_crisis": False,
            "intent": intent,
            "confidence": confidence,
            "emotion": emotion,
            "source": source,
            "language": user_lang,
        }