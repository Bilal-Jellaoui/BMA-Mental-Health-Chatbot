"""
app/services/safety.py
-----------------------
Crisis detection module for BMA Mental Health Chatbot.
Scans user messages for signs of self-harm or suicide ideation,
supports Arabic, French, and English keywords.
Creators: Bilal Jellaoui, Mohammed Azil, Ayoube Echihami
"""

import re
from config.config import Config


class CrisisDetector:

    def __init__(self):
        self.keywords = Config.CRISIS_KEYWORDS

    def is_crisis(self, message: str) -> bool:
        """
        Return True if the message contains any crisis keyword.
        Checks both the original message and lowercased version
        to catch Arabic (which has no case) and Latin scripts.
        """
        msg_lower = message.lower()

        for kw in self.keywords:
            kw_lower = kw.lower()
            # For Arabic keywords: simple substring check (no word boundary in Arabic)
            if any('\u0600' <= c <= '\u06FF' for c in kw):
                if kw_lower in message:
                    return True
            else:
                # For Latin keywords: use word boundary to avoid false positives
                # e.g. "kills" should not trigger on "kill"
                if re.search(rf'\b{re.escape(kw_lower)}\b', msg_lower):
                    return True

        return False

    def get_crisis_response(self) -> str:
        return Config.CRISIS_RESPONSE