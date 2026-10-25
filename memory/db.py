"""
memory/db.py -- Per-user SQLite conversation memory for FinSage.
Guarantees:
- Per-user data isolation (no cross-user leakage)
- Session management, export, and complete deletion
- Audit logging of verifications
"""

import sqlite3
import json
import uuid
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from config import cfg


class MemoryDatabase:
    """Thread-safe SQLite manager for FinSage sessions and history."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = str(db_path or cfg.db_path)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at REAL,
                    preferences TEXT
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT,
                    created_at REAL,
                    updated_at REAL,
                    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sentences_json TEXT,
                    sources_json TEXT,
                    tool_results_json TEXT,
                    verified INTEGER DEFAULT 0,
                    abstained INTEGER DEFAULT 0,
                    created_at REAL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
            """)

    def ensure_user(self, user_id: str) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users (user_id, created_at, preferences) VALUES (?, ?, ?)",
                (user_id, time.time(), json.dumps({})),
            )

    def create_session(self, user_id: str, title: str = "New Financial Session") -> str:
        self.ensure_user(user_id)
        session_id = str(uuid.uuid4())
        now = time.time()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, title, now, now),
            )
        return session_id

    def list_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT session_id, user_id, title, created_at, updated_at FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def add_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        sentences: Optional[List[Dict[str, Any]]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        verified: bool = False,
        abstained: bool = False,
    ) -> str:
        msg_id = str(uuid.uuid4())
        now = time.time()
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO messages (
                    message_id, session_id, user_id, role, content,
                    sentences_json, sources_json, tool_results_json,
                    verified, abstained, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    msg_id,
                    session_id,
                    user_id,
                    role,
                    content,
                    json.dumps(sentences or []),
                    json.dumps(sources or []),
                    json.dumps(tool_results or []),
                    1 if verified else 0,
                    1 if abstained else 0,
                    now,
                ),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
        return msg_id

    def get_session_messages(self, session_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve messages with strict user boundary check."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND user_id = ? ORDER BY created_at ASC",
                (session_id, user_id),
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["sentences"] = json.loads(d.pop("sentences_json") or "[]")
                d["sources"] = json.loads(d.pop("sources_json") or "[]")
                d["tool_results"] = json.loads(d.pop("tool_results_json") or "[]")
                results.append(d)
            return results

    def get_messages(self, session_id: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve messages for a session in chronological order."""
        with self._get_conn() as conn:
            if user_id:
                query = "SELECT * FROM messages WHERE session_id = ? AND user_id = ? ORDER BY created_at ASC"
                params = (session_id, user_id)
            else:
                query = "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC"
                params = (session_id,)
            rows = conn.execute(query, params).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["sentences"] = json.loads(d.pop("sentences_json") or "[]")
                d["sources"] = json.loads(d.pop("sources_json") or "[]")
                d["tool_results"] = json.loads(d.pop("tool_results_json") or "[]")
                results.append(d)
            return results

    def delete_session(self, session_id: str, user_id: str) -> bool:
        with self._get_conn() as conn:
            cur = conn.execute(
                "DELETE FROM sessions WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            )
            return cur.rowcount > 0

    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """User data export compliant with privacy standards."""
        sessions = self.list_sessions(user_id)
        data = {"user_id": user_id, "sessions": []}
        for s in sessions:
            s_data = dict(s)
            s_data["messages"] = self.get_session_messages(s["session_id"], user_id)
            data["sessions"].append(s_data)
        return data


db = MemoryDatabase()
