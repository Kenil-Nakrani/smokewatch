"""SQLite alert log. Stores alerts, snapshot paths and user feedback."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

DB_PATH = Path("outputs/alerts.db")

SCHEMA = """CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created REAL, source TEXT, frame_index INTEGER,
    level TEXT, reason TEXT, snapshot TEXT,
    feedback TEXT DEFAULT NULL)"""


def _conn(db: Path = DB_PATH) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.execute(SCHEMA)
    return con


def add_alert(source: str, frame_index: int, level: str, reason: str,
              snapshot: str, db: Path = DB_PATH) -> int:
    with _conn(db) as con:
        cur = con.execute(
            "INSERT INTO alerts(created, source, frame_index, level, reason, snapshot)"
            " VALUES (?,?,?,?,?,?)",
            (time.time(), source, frame_index, level, reason, snapshot))
        return cur.lastrowid


def list_alerts(db: Path = DB_PATH) -> list[dict]:
    con = _conn(db)
    con.row_factory = sqlite3.Row
    return [dict(r) for r in con.execute("SELECT * FROM alerts ORDER BY id DESC")]


def set_feedback(alert_id: int, feedback: str, db: Path = DB_PATH) -> None:
    """feedback is 'confirmed' or 'false_alarm'."""
    with _conn(db) as con:
        con.execute("UPDATE alerts SET feedback=? WHERE id=?", (feedback, alert_id))
