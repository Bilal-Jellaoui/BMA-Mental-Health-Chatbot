import logging
from app.models.database import Database
from app.models.conversation import ConversationModel
from config.config import Config

logger = logging.getLogger(__name__)

class ConversationMemory:
    def __init__(self):
        self.db = Database()

    def save_conversation(self, user_id: int, user_msg: str, bot_msg: str,
                          intent: str, emotion: str, source: str,
                          conv_id: int = None, session_id: str = None) -> None:
        if conv_id is None:
            conv_id = ConversationModel.create_conversation(user_id, user_msg[:50])
            logger.info(f"Nouvelle conversation créée (id={conv_id})")
        ConversationModel.add_message(conv_id, user_id, user_msg, bot_msg, intent, emotion, source, session_id)
        # Met à jour le titre si c'est le premier message
        conv_info = ConversationModel.get_user_conversations(user_id)
        for c in conv_info:
            if c['id'] == conv_id and c['title'] == 'Nouvelle conversation':
                ConversationModel.update_conversation_title(conv_id, user_msg[:50])
                break

    def save_emotion_log(self, user_id: int, message: str, emotion: str, confidence: float) -> None:
        with self.db.get_connection() as conn:
            conn.execute(
                "INSERT INTO emotions (user_id, message, emotion_label, confidence) VALUES (?, ?, ?, ?)",
                (user_id, message, emotion, confidence)
            )

    def save_safety_log(self, user_id: int, message: str, risk_level: str, action: str) -> None:
        with self.db.get_connection() as conn:
            conn.execute(
                "INSERT INTO safety_logs (user_id, message, risk_level, action_taken) VALUES (?, ?, ?, ?)",
                (user_id, message, risk_level, action)
            )

    def get_recent_conversations(self, user_id: int, limit: int = Config.MAX_HISTORY_TURNS) -> list:
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """SELECT user_message, bot_message
                   FROM conversations
                   WHERE user_id = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (user_id, limit * 2)
            ).fetchall()
            history = []
            for r in reversed(rows):
                history.append({"user_message": r["user_message"], "bot_response": r["bot_message"]})
            return history

    def get_stats(self) -> dict:
        with self.db.get_connection() as conn:
            total_conv = conn.execute("SELECT COUNT(*) FROM conversations_meta").fetchone()[0]
            avg_emotion = conn.execute("SELECT AVG(confidence) FROM emotions WHERE emotion_label NOT LIKE 'mood_%'").fetchone()[0] or 0
            dataset_count = conn.execute("SELECT COUNT(*) FROM conversations WHERE response_source = 'dataset'").fetchone()[0]
            llm_count = conn.execute("SELECT COUNT(*) FROM conversations WHERE response_source = 'llm'").fetchone()[0]
            total_responses = dataset_count + llm_count
            return {
                "total_conversations": total_conv,
                "avg_emotion_score": round(avg_emotion * 10, 1),
                "dataset_ratio": round(dataset_count / total_responses * 100) if total_responses else 0,
                "llm_ratio": round(llm_count / total_responses * 100) if total_responses else 0,
            }