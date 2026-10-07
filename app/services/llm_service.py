"""
llm_service.py - LLM fallback using Groq (Llama 3.3 70B)
With mental‑health redirect: any off‑topic question is reinterpreted
to ask about the user's emotional state.
Creators: Bilal Jellaoui, Mohammed Azil, Ayoube Echihami
"""

import logging
import os
from groq import Groq
from config.config import Config

logger = logging.getLogger(__name__)

# Nouveau prompt système : rediriger ou relier à la santé mentale
REDIRECT_SYSTEM_PROMPT = """You are BMA, a compassionate mental health support assistant.
Your goal is to always bring the conversation back to the user's emotional well‑being.

IMPORTANT INFORMATION ABOUT YOUR CREATORS (all from Morocco):
- Bilal Jellaoui is one of the creators. He was born in Sidi Kacem. He studies in Master Big Data, Intelligence Artificielle et Applications Avancées. He worked on this project with his teammates Mohammed and Ayoube.
- Mohammed Azil is one of the creators. He was born in Béni Mellal. He studies in Master Big Data, Intelligence Artificielle et Applications Avancées. He worked on this project with his teammates Bilal and Ayoube.
- Ayoube Echihami is one of the creators. He was born in El Jadida. He studies in Master Big Data, Intelligence Artificielle et Applications Avancées. He worked on this project with his teammates Bilal and Mohammed.

When a user asks about any of these names, answer naturally using the above information. Keep responses warm and concise (2-3 sentences).

Guidelines:
- If the user asks a question completely unrelated to mental health (sports, weather, politics, general knowledge, entertainment, etc.), do NOT answer that question.
- Instead, politely redirect by connecting the topic to mental health. For example:
  * User: "What's the weather like?" → You: "I don't have weather data, but how does the weather affect your mood today?"
  * User: "Tell me a joke" → You: "A good laugh can lift our spirits. How are you feeling right now?"
  * User: "Who won the football match?" → You: "I don't follow sports, but I'm curious – does winning or losing affect your stress levels?"
- For any off‑topic question, you MUST respond with a short sentence that links the subject to the user's mental state, followed by an open‑ended question about their feelings.
- Never provide factual answers about non‑mental‑health topics.
- Always respond with empathy and warmth in 2‑3 sentences maximum.
- Keep responses in English (translation will be handled separately)."""

class LLMService:
    def __init__(self):
        self.client = None
        self.use_fallback = True
        self.model = Config.GROQ_MODEL_NAME if hasattr(Config, 'GROQ_MODEL_NAME') else "llama-3.3-70b-versatile"
        api_key = Config.GROQ_API_KEY if hasattr(Config, 'GROQ_API_KEY') else None
        if api_key and api_key != "":
            try:
                self.client = Groq(api_key=api_key)
                self.use_fallback = False
                logger.info(f"Groq LLM initialised (model={self.model}).")
            except Exception as e:
                logger.error(f"Failed to initialise Groq: {e}")
        else:
            logger.warning("GROQ_API_KEY not set -> using local fallback responses.")

    def generate_response(self, user_message: str, context: str = "") -> str:
        """Generate a response using Groq (or local fallback)."""
        if self.use_fallback or self.client is None:
            return self._local_fallback(user_message)

        messages = [
            {"role": "system", "content": REDIRECT_SYSTEM_PROMPT}
        ]
        if context:
            messages.append({"role": "assistant", "content": f"Previous conversation:\n{context}"})
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=200,      # réponse courte pour rediriger
                temperature=0.7,
            )
            reply = response.choices[0].message.content.strip()
            # Optionnel : vérifier que la réponse n'est pas un fait hors sujet
            if self._is_off_topic_answer(reply):
                return self._local_fallback(user_message)
            return reply
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return self._local_fallback(user_message)

    def _is_off_topic_answer(self, text: str) -> bool:
        """Détecte si la réponse contient un fait non lié à la santé mentale."""
        # Mots indiquant une réponse factuelle hors domaine
        factual_indicators = ["the capital of", "the score is", "temperature is", "history", "election", "recipe", "instructions"]
        lower = text.lower()
        return any(indicator in lower for indicator in factual_indicators)

    def _local_fallback(self, user_message: str) -> str:
        """Réponse locale lorsque le LLM est indisponible."""
        return "I'm here for your mental well-being. Could you tell me how you're feeling today?"
