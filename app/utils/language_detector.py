"""
language_service.py - Multilingual support for BMA chatbot.
Detects Arabic / French, translates input → English for NLP,
then translates the English response back to the user's language.

Creators: Bilal Jellaoui, Mohammed Azil, Ayoube Echihami

Translation strategy (priority order):
  1. deep-translator (googletrans wrapper) — free, no API key needed
  2. Fallback: return original text unchanged (graceful degradation)

Install:   pip install deep-translator langdetect
"""

import logging
import re

logger = logging.getLogger(__name__)

# ── Optional imports (graceful fallback if not installed) ─────────────────────
try:
    from deep_translator import GoogleTranslator
    _TRANSLATOR_AVAILABLE = True
except ImportError:
    _TRANSLATOR_AVAILABLE = False
    logger.warning("deep-translator not installed. Translation disabled. Run: pip install deep-translator")

try:
    from langdetect import detect as _langdetect
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not installed. Using heuristic detection. Run: pip install langdetect")


# ── Language detection ────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Return ISO 639-1 language code: 'ar', 'fr', or 'en'.
    Uses heuristics first (fast), then langdetect if available.
    """
    # Arabic Unicode block heuristic (fastest, most reliable for Arabic)
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    if arabic_chars / max(len(text), 1) > 0.15:
        return 'ar'

    # French heuristic: accented characters
    if re.search(r'[éèêëàâçùûîïôœæ]', text.lower()):
        return 'fr'

    # langdetect for ambiguous cases
    if _LANGDETECT_AVAILABLE:
        try:
            code = _langdetect(text)
            if code in ('ar', 'fr'):
                return code
        except Exception:
            pass

    return 'en'


# ── Translation ───────────────────────────────────────────────────────────────

def translate(text: str, source: str, target: str) -> str:
    """
    Translate text from `source` language to `target` language.
    Returns original text if translation is unavailable or fails.
    """
    if source == target:
        return text

    if not _TRANSLATOR_AVAILABLE:
        logger.debug("Translation skipped (deep-translator not installed).")
        return text

    try:
        translated = GoogleTranslator(source=source, target=target).translate(text)
        logger.debug(f"Translated [{source}→{target}]: '{text[:40]}' → '{translated[:40]}'")
        return translated or text
    except Exception as e:
        logger.warning(f"Translation failed ({source}→{target}): {e}")
        return text


# ── High-level helpers ────────────────────────────────────────────────────────

def normalize_to_english(user_message: str) -> tuple[str, str]:
    """
    Detect the user's language and translate the message to English.

    Returns:
        (english_text, detected_language_code)
    """
    lang = detect_language(user_message)
    if lang == 'en':
        return user_message, 'en'
    english = translate(user_message, source=lang, target='en')
    return english, lang


def localize_response(english_response: str, target_language: str) -> str:
    """
    Translate an English response back into the user's language.
    Returns the English response unchanged if target is 'en'.
    """
    if target_language == 'en':
        return english_response
    return translate(english_response, source='en', target=target_language)