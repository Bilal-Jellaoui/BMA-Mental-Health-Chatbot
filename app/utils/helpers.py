"""
helpers.py - Miscellaneous helper functions.
"""

import time
import re

def get_timestamp() -> int:
    """Return current Unix timestamp."""
    return int(time.time())

def extract_number(text: str) -> int or None:
    """Extract first integer from a string."""
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None

def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate long text for logging or display."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."