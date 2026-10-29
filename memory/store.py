"""SQLite store for episodic and procedural memory."""
from __future__ import annotations

import os
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    question   TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status     TEXT NOT NULL,
    answer     TEXT,
    citations  TEXT,
    confidence TEXT,
    usage      TEXT
);
CREATE TABLE IF NOT EXISTS events (
    session_id TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    type       TEXT NOT NULL,
    agent      TEXT NOT NULL,
    content    TEXT NOT NULL,
    data       TEXT NOT NULL,
    ts         TEXT NOT NULL,
    PRIMARY KEY (session_id, seq)
);
CREATE TABLE IF NOT EXISTS tool_stats (
    tool       TEXT PRIMARY KEY,
    calls      INTEGER NOT NULL DEFAULT 0,
    failures   INTEGER NOT NULL DEFAULT 0,
    empty      INTEGER NOT NULL DEFAULT 0,
    total_ms   REAL    NOT NULL DEFAULT 0,
    cited      INTEGER NOT NULL DEFAULT 0
);
"""


class Store:
    def __init__(self, path: str) -> None:
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)
