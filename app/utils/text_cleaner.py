"""
text_cleaner.py - Text cleaning utilities for NLP preprocessing.
"""

import re

def clean_text(text: str) -> str:
    """
    Clean text: lowercase, remove punctuation, collapse whitespace.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)   # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # collapse spaces
    return text

def basic_clean(text: str) -> str:
    """Light cleaning for quick tasks."""
    return text.lower().strip()