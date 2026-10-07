"""
run.py - Entry point for BMA Mental Health Chatbot.
Creators: Bilal Jellaoui, Mohammed Azil, Ayoube Echihami

Usage:
    python run.py
"""
from dotenv import load_dotenv
load_dotenv()

import os
from app import create_app
from config.config import Config
from config.logging_config import setup_logging

# ── Logging ────────────────────────────────────────────────────────────────────
log_file = os.path.join(os.path.dirname(__file__), "logs", "app.log")
setup_logging(log_file=log_file, log_level="DEBUG" if Config.DEBUG else "INFO")

# ── App ────────────────────────────────────────────────────────────────────────
app = create_app(Config)

if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)