"""
blackwell/logger.py
Conversation logger — every Zephyr exchange stored in SQLite
as actual coordinate points in the 5-dimensional reward space.

Schema uses proper numeric columns (not JSON blobs) so we can
do AVG(), projection math, and SQL-level aggregation directly.
"""

import sqlite3
import json
import datetime
import uuid
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "blackwell.db")
DIMS = ["accuracy", "logic", "tone", "curiosity", "safety"]


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create or migrate tables."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                started_at  TEXT NOT NULL,
                model       TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS exchanges (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                turn        INTEGER NOT NULL,
                timestamp   TEXT NOT NULL,
                human       TEXT NOT NULL,
                zephyr      TEXT NOT NULL,
                tools_used  TEXT,
                v_accuracy  REAL,
                v_logic     REAL,
                v_tone      REAL,
                v_curiosity REAL,
                v_safety    REAL,
                eval_notes  TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        conn.commit()
        _migrate(conn)


def _migrate(conn):
    """Add new columns to existing DB if upgrading from old JSON-blob schema."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(exchanges)")}
    new_cols = {
        "v_accuracy":  "REAL",
        "v_logic":     "REAL",
        "v_tone":      "REAL",
        "v_curiosity": "REAL",
        "v_safety":    "REAL",
        "eval_notes":  "TEXT",
    }
    for col, typ in new_cols.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE exchanges ADD COLUMN {col} {typ}")
    conn.commit()

    # Migrate old JSON scores → numeric columns
    rows = conn.execute(
        "SELECT id, scores FROM exchanges WHERE v_accuracy IS NULL AND scores IS NOT NULL"
    ).fetchall()
    for row in rows:
        try:
            s = json.loads(row["scores"])
            conn.execute(
                """UPDATE exchanges SET
                   v_accuracy=?, v_logic=?, v_tone=?, v_curiosity=?, v_safety=?
                   WHERE id=?""",
                (s.get("accuracy"), s.get("logic"), s.get("tone"),
                 s.get("curiosity"), s.get("safety"), row["id"])
            )
        except Exception:
            pass
    conn.commit()


def new_session(model: str) -> str:
    init_db()
    session_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?)",
            (session_id, datetime.datetime.now().isoformat(), model)
        )
        conn.commit()
    return session_id


def log_exchange(session_id: str, turn: int, human: str, zephyr: str,
                 tools_used: list = None, scores: dict = None) -> str:
    """Log one exchange. Scores stored as numeric columns."""
    init_db()
    exchange_id = str(uuid.uuid4())
    s = scores or {}
    with _connect() as conn:
        conn.execute(
            """INSERT INTO exchanges
               (id, session_id, turn, timestamp, human, zephyr, tools_used,
                v_accuracy, v_logic, v_tone, v_curiosity, v_safety, eval_notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                exchange_id, session_id, turn,
                datetime.datetime.now().isoformat(),
                human, zephyr,
                json.dumps(tools_used or []),
                s.get("accuracy"), s.get("logic"), s.get("tone"),
                s.get("curiosity"), s.get("safety"),
                s.get("notes"),
            )
        )
        conn.commit()
    return exchange_id


def update_scores(exchange_id: str, scores: dict):
    """Write evaluation scores to numeric columns."""
    init_db()
    with _connect() as conn:
        conn.execute(
            """UPDATE exchanges SET
               v_accuracy=?, v_logic=?, v_tone=?, v_curiosity=?, v_safety=?, eval_notes=?
               WHERE id=?""",
            (scores.get("accuracy"), scores.get("logic"), scores.get("tone"),
             scores.get("curiosity"), scores.get("safety"),
             scores.get("notes"), exchange_id)
        )
        conn.commit()


def get_average_vector(limit: int = 500):
    """
    Compute average payoff vector directly in SQL.
    Returns None if no scored exchanges exist.
    """
    init_db()
    with _connect() as conn:
        row = conn.execute(
            """SELECT
               AVG(v_accuracy)  as accuracy,
               AVG(v_logic)     as logic,
               AVG(v_tone)      as tone,
               AVG(v_curiosity) as curiosity,
               AVG(v_safety)    as safety,
               COUNT(*)         as n
               FROM (
                   SELECT v_accuracy, v_logic, v_tone, v_curiosity, v_safety
                   FROM exchanges
                   WHERE v_accuracy IS NOT NULL
                   ORDER BY timestamp DESC
                   LIMIT ?
               )""",
            (limit,)
        ).fetchone()
    if not row or row["n"] == 0:
        return None
    return {d: round(row[d], 4) for d in DIMS}


def get_recent_exchanges(limit: int = 100) -> list[dict]:
    """Fetch recent raw exchanges."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT id, session_id, turn, timestamp, human, zephyr,
               tools_used, v_accuracy, v_logic, v_tone, v_curiosity, v_safety, eval_notes
               FROM exchanges ORDER BY timestamp DESC LIMIT ?""",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"DB: {os.path.abspath(DB_PATH)}")
    avg = get_average_vector()
    if avg:
        print(f"Current average vector: {avg}")
    else:
        print("No scored exchanges yet.")
