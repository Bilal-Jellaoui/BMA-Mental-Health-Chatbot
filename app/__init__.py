"""
app/__init__.py - Flask application factory for BMA Mental Health Chatbot.
Creators: Bilal Jellaoui, Mohammed Azil, Ayoube Echihami
"""

import logging
from flask import Flask
from config.config import Config

logger = logging.getLogger(__name__)


def create_app(config_class=Config):
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config.from_object(config_class)

    # ── Blueprints ─────────────────────────────────────────────────────────────
    from app.routes import chat_bp
    app.register_blueprint(chat_bp)

    logger.info("BMA Flask app created successfully.")
    return app