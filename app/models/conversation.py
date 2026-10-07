"""
app/models/chat_model.py
------------------------
LLM wrapper for BMA — uses Anthropic Claude API with a
mental-health-specialized system prompt.
Creators: Bilal Jellaoui, Mohammed Azil, Ayoube Echihami
"""
"""
app/models/conversation.py - Gestion des conversations et historique.
Créateurs : Bilal Jellaoui, Mohammed Azil, Ayoube Echihami
"""

"""
app/models/conversation.py - Gestion des conversations
Créateurs : Bilal Jellaoui, Mohammed Azil, Ayoube Echihami
"""

from app.models.database import Database

class ConversationModel:
    @staticmethod
    def create_conversation(user_id: int, title: str = None) -> int:
        db = Database()
        with db.get_connection() as conn:
            cur = conn.execute(
                "INSERT INTO conversations_meta (user_id, title) VALUES (?, ?)",
                (user_id, title or 'Nouvelle conversation')
            )
            return cur.lastrowid

    @staticmethod
    def update_conversation_title(conv_id: int, title: str) -> None:
        db = Database()
        with db.get_connection() as conn:
            conn.execute(
                "UPDATE conversations_meta SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (title, conv_id)
            )

    @staticmethod
    def get_user_conversations(user_id: int, limit: int = 50) -> list:
        db = Database()
        with db.get_connection() as conn:
            rows = conn.execute(
                """SELECT id, title, updated_at
                   FROM conversations_meta
                   WHERE user_id = ?
                   ORDER BY updated_at DESC
                   LIMIT ?""",
                (user_id, limit)
            ).fetchall()
            return [dict(row) for row in rows]

    @staticmethod
    def get_conversation_messages(conv_id: int) -> list:
        db = Database()
        with db.get_connection() as conn:
            rows = conn.execute(
                """SELECT user_message, bot_message, timestamp
                   FROM conversations
                   WHERE conv_id = ?
                   ORDER BY timestamp ASC""",
                (conv_id,)
            ).fetchall()
            messages = []
            for row in rows:
                messages.append({"role": "user", "content": row["user_message"], "timestamp": row["timestamp"]})
                messages.append({"role": "bot", "content": row["bot_message"], "timestamp": row["timestamp"]})
            return messages

    @staticmethod
    def add_message(conv_id: int, user_id: int, user_msg: str, bot_msg: str,
                    intent: str, emotion: str, source: str, session_id: str = None) -> None:
        """Ajoute un échange (user + bot) à une conversation."""
        db = Database()
        with db.get_connection() as conn:
            conn.execute(
                """INSERT INTO conversations
                   (session_id, user_id, user_message, bot_message, intent, emotion, response_source, conv_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, user_id, user_msg, bot_msg, intent, emotion, source, conv_id)
            )
            conn.execute(
                "UPDATE conversations_meta SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (conv_id,)
            )