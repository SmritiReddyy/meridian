"""Episodic memory: every research session and the events it streamed."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from memory.store import Store


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpisodicMemory:
    def __init__(self, store: Store) -> None:
        self.s = store

    def start(self, session_id: str, question: str) -> None:
        with self.s.lock:
            self.s.db.execute("INSERT INTO sessions (session_id, question, created_at, status) VALUES (?, ?, ?, 'running')",
                              (session_id, question, now_iso()))
            self.s.db.commit()

    def event(self, session_id: str, type_: str, agent: str, content: str, data: dict) -> dict:
        ts = now_iso()
        with self.s.lock:
            seq = self.s.db.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM events WHERE session_id = ?",
                                    (session_id,)).fetchone()[0]
            self.s.db.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)",
                              (session_id, seq, type_, agent, content, json.dumps(data, default=str), ts))
            self.s.db.commit()
        return {"seq": seq, "type": type_, "agent": agent, "content": content, "data": data, "timestamp": ts}

    def finish(self, session_id: str, status: str, answer: str = "", citations: list | None = None,
               confidence: dict | None = None, usage: dict | None = None) -> None:
        with self.s.lock:
            self.s.db.execute("UPDATE sessions SET status = ?, answer = ?, citations = ?, confidence = ?, usage = ? "
                              "WHERE session_id = ?",
                              (status, answer, json.dumps(citations or []), json.dumps(confidence or {}),
                               json.dumps(usage or {}), session_id))
            self.s.db.commit()

    def sessions(self, limit: int = 50) -> list[dict]:
        with self.s.lock:
            rows = self.s.db.execute("SELECT session_id, question, created_at, status FROM sessions "
                                     "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def events(self, session_id: str, after: int = -1) -> list[dict]:
        with self.s.lock:
            rows = self.s.db.execute("SELECT * FROM events WHERE session_id = ? AND seq > ? ORDER BY seq",
                                     (session_id, after)).fetchall()
        return [{"seq": r["seq"], "type": r["type"], "agent": r["agent"], "content": r["content"],
                 "data": json.loads(r["data"]), "timestamp": r["ts"]} for r in rows]

    def status(self, session_id: str) -> str | None:
        with self.s.lock:
            row = self.s.db.execute("SELECT status FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return row["status"] if row else None
