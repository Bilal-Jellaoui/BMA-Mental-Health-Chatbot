"""
response_engine.py - Hybrid response system: dataset first, LLM fallback.
Creators: Bilal Jellaoui, Mohammed Azil, Ayoube Echihami
"""

import json
import random
import logging
from pathlib import Path
from config.config import Config

logger = logging.getLogger(__name__)


class ResponseEngine:
    """
    Decides how to respond:
      1. If intent confidence >= threshold  → pick a random response from responses.json
      2. Otherwise                          → signal to use LLM fallback
    """

    def __init__(self, responses_path: str = None):
        path = responses_path or str(Path(__file__).parent.parent.parent / "data" / "responses.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Support both {"responses": {...}} and {"intents": [...]} shapes
            if "responses" in data:
                self.responses: dict = data["responses"]
            elif "intents" in data:
                self.responses = {
                    intent["tag"]: intent["responses"]
                    for intent in data["intents"]
                }
            else:
                self.responses = {}
            logger.info(f"ResponseEngine loaded {len(self.responses)} intent tags.")
        except FileNotFoundError:
            logger.warning(f"responses.json not found at {path}. Dataset fallback disabled.")
            self.responses = {}

    # ------------------------------------------------------------------
    def get_response(
        self,
        intent: str,
        confidence: float,
        threshold: float = None,
    ) -> tuple[str | None, str]:
        """
        Returns (response_text | None, source).
        source is 'dataset' or 'llm_needed'.
        """
        thresh = threshold if threshold is not None else Config.CONFIDENCE_THRESHOLD

        if confidence >= thresh and intent in self.responses:
            candidates = self.responses[intent]
            if candidates:
                chosen = random.choice(candidates)
                logger.debug(f"Dataset hit: intent={intent} conf={confidence:.2f}")
                return chosen, "dataset"

        logger.debug(f"LLM needed: intent={intent} conf={confidence:.2f}")
        return None, "llm_needed"