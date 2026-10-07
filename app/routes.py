"""
app/routes.py — API endpoints for BMA Mental Health Chatbot.
Creators: Bilal Jellaoui, Mohammed Azil, Ayoube Echihami

Endpoints:
  POST /api/chat           → send message, get BMA response
  POST /api/session/start  → create a new session
  POST /api/mood           → log a mood entry
  GET  /api/mood/history   → mood history for a session
  GET  /api/stats          → global usage statistics
  POST /api/feedback       → submit a response rating
"""

"""
app/routes.py — API endpoints for BMA Mental Health Chatbot.
Creators: Bilal Jellaoui, Mohammed Azil, Ayoube Echihami
"""

from flask import Blueprint, request, jsonify, render_template
from app.controllers.chat_controller import ChatController
from app.services.memory import ConversationMemory
from app.models.user import UserModel
from app.models.conversation import ConversationModel
import uuid

chat_bp = Blueprint("chat", __name__)
controller = ChatController()
memory = ConversationMemory()


# ── Pages ─────────────────────────────────────────────────────────────────────

@chat_bp.route("/")
def index():
    return render_template("index.html")

@chat_bp.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ── API – Sessions ───────────────────────────────────────────────────────────

@chat_bp.route("/api/session/start", methods=["POST"])
def start_session():
    session_id = str(uuid.uuid4())
    return jsonify({"session_id": session_id}), 201


# ── API – Messages ───────────────────────────────────────────────────────────

@chat_bp.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "").strip()
    user_message = data.get("message", "").strip()
    conv_id = data.get("conv_id")  # peut être None

    if not session_id or not user_message:
        return jsonify({"error": "session_id and message are required"}), 400

    user_id = UserModel.get_or_create(session_id)

    # Si aucune conversation n'est spécifiée, on en crée une nouvelle
    if conv_id is None:
        conv_id = ConversationModel.create_conversation(user_id, user_message[:50])

    result = controller.process_message(user_id, user_message, conv_id, session_id)

    return jsonify({
        "reply":      result["response"],
        "is_crisis":  result.get("is_crisis", False),
        "intent":     result.get("intent", ""),
        "emotion":    result.get("emotion", ""),
        "source":     result.get("source", ""),
        "language":   result.get("language", "en"),
        "conv_id":    conv_id,
        "confidence": round(result.get("confidence", 0.0), 3),
    }), 200


# ── API – Mood ────────────────────────────────────────────────────────────────

@chat_bp.route("/api/mood", methods=["POST"])
def log_mood():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "").strip()
    score = data.get("score")
    note = data.get("note", "")
    if not session_id or score is None:
        return jsonify({"error": "session_id and score are required"}), 400
    if not isinstance(score, (int, float)) or not (1 <= score <= 10):
        return jsonify({"error": "score must be a number between 1 and 10"}), 400
    user_id = UserModel.get_or_create(session_id)
    with memory.db.get_connection() as conn:
        conn.execute(
            "INSERT INTO emotions (user_id, message, emotion_label, confidence) VALUES (?, ?, ?, ?)",
            (user_id, f"Mood score: {score}. Note: {note}", f"mood_{int(score)}", 1.0),
        )
    return jsonify({"logged": True, "score": score}), 201

@chat_bp.route("/api/mood/history", methods=["GET"])
def mood_history():
    session_id = request.args.get("session_id", "").strip()
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    user_id = UserModel.get_or_create(session_id)
    with memory.db.get_connection() as conn:
        rows = conn.execute(
            """SELECT emotion_label, confidence, timestamp
               FROM emotions
               WHERE user_id = ? AND emotion_label LIKE 'mood_%'
               ORDER BY timestamp DESC LIMIT 30""",
            (user_id,),
        ).fetchall()
    entries = []
    for r in rows:
        try:
            score = int(r["emotion_label"].split("_")[1])
        except (IndexError, ValueError):
            score = 0
        entries.append({"score": score, "logged_at": r["timestamp"]})
    return jsonify({"session_id": session_id, "mood_history": entries}), 200


# ── API – Statistiques ────────────────────────────────────────────────────────

@chat_bp.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(memory.get_stats()), 200


# ── API – Feedback ────────────────────────────────────────────────────────────

@chat_bp.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "").strip()
    rating = data.get("rating")
    comment = data.get("comment", "")
    if not session_id or rating is None:
        return jsonify({"error": "session_id and rating are required"}), 400
    if not isinstance(rating, int) or not (1 <= rating <= 5):
        return jsonify({"error": "rating must be an integer between 1 and 5"}), 400
    user_id = UserModel.get_or_create(session_id)
    with memory.db.get_connection() as conn:
        conn.execute(
            "INSERT INTO feedback (user_id, rating, comment) VALUES (?, ?, ?)",
            (user_id, rating, comment),
        )
    return jsonify({"received": True}), 201


# ── API – Gestion de l’historique des conversations (Nouveaux endpoints) ──────

@chat_bp.route("/api/conversations", methods=["GET"])
def get_conversations():
    """Liste toutes les conversations d’un utilisateur."""
    session_id = request.args.get("session_id", "").strip()
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    user_id = UserModel.get_or_create(session_id)
    convs = ConversationModel.get_user_conversations(user_id)
    return jsonify(convs)

@chat_bp.route("/api/conversations/<int:conv_id>", methods=["GET"])
def get_conversation(conv_id):
    """Récupère tous les messages d’une conversation spécifique."""
    messages = ConversationModel.get_conversation_messages(conv_id)
    return jsonify({"messages": messages})

@chat_bp.route("/api/conversations/new", methods=["POST"])
def new_conversation():
    """Crée une nouvelle conversation vide et retourne son ID."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "").strip()
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    user_id = UserModel.get_or_create(session_id)
    conv_id = ConversationModel.create_conversation(user_id)
    return jsonify({"conv_id": conv_id}), 201