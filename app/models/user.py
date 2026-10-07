from app.models.database import Database

class UserModel:
    @staticmethod
    def get_or_create(username, email=None):
        db = Database()
        with db.get_connection() as conn:
            user = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if user:
                return user['id']
            else:
                cur = conn.execute("INSERT INTO users (username, email) VALUES (?, ?)", (username, email))
                return cur.lastrowid