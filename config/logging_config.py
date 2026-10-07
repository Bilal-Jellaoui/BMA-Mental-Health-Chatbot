"""
logging_config.py - Logging setup for the Mental Health Chatbot
"""

import logging
import logging.handlers
import os
import sys

def setup_logging(log_file: str, log_level: str = "INFO"):
    """
    Configure root logger with:
      - Rotating file handler (max 5 MB × 3 backups)
      - Console (stdout) handler
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # ── File handler ─────────────────────────────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # ── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "transformers", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    return root_logger