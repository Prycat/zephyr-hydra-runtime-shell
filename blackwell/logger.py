"""
blackwell/logger.py
Conversation logger — every Zephyr exchange stored in SQLite
as actual coordinate points in the 5-dimensional reward space.

Fix 1 (Decay / EMA):
    get_average_vector() now returns an Exponential Moving Average
    rather than a plain SQL AVG over the last 500 rows.

    Why: a plain AVG acts as "cognitive friction" — old exchanges from a
    weaker model era drag the signal down even after LoRA improvements.
    EMA gives recent exchanges exponentially more weight, so the Oracle
    sees current behaviour, not history.

    EMA is persisted in the `ema_state` table so it survives restarts.
    The raw SQL average is still available via get_sql_average_vector()
    for auditing.

EMA_ALPHA tuning guide:
    0.05  — very smooth, slow to react (good for stable production)
    0.15  — balanced (default)
    0.30  — responsive, reacts within ~10 exchanges (good for rapid iteration)
"""
from __future__ import annotations

import sqlite3
import json
import datetime
import uuid
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "blackwell.db")
DIMS = ["accuracy", "logic", "tone", "curiosity", "safety"]

# Fix 1 — EMA alpha.  Exported so tests and background_eval can inspect it.
EMA_ALPHA = 0.15


# ── Internal connection ────────────────────────────────────────────────────────

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Schema init & migration ───────────────────────────────────────────────────

def init_db():
    """Create or migrate all tables."""
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
        # Fix 1 — EMA state table: one row per dimension key
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ema_state (
                key   TEXT PRIMARY KEY,
                value REAL NOT NULL
            )
        """)
        conn.commit()
        _migrate(conn)


def _migrate(conn):
    """Add new columns / tables to existing DB when upgrading."""
    # --- exchanges columns ---
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

    # --- ema_state table (may not exist on old installs) ---
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ema_state (
            key   TEXT PRIMARY KEY,
            value REAL NOT NULL
        )
    """)
    conn.commit()

    # --- Migrate old JSON scores → numeric columns (only if old schema exists) ---
    if "scores" not in existing:
        return   # fresh DB — nothing to migrate
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


# ── EMA helpers (Fix 1) ───────────────────────────────────────────────────────

def _get_ema(conn) -> dict | None:
    """
    Read the persisted EMA vector from the DB.
    Returns None if no EMA has been computed yet.
    """
    rows = conn.execute(
        "SELECT key, value FROM ema_state WHERE key LIKE 'ema_%'"
    ).fetchall()
    if not rows:
        return None
    result = {}
    for key, value in rows:
        dim = key[4:]   # strip 'ema_' prefix
        if dim in DIMS:
            result[dim] = value
    return result if len(result) == len(DIMS) else None


def _update_ema(conn, scores: dict) -> None:
    """
    Apply one EMA step:  ema_new = α * score + (1-α) * ema_old

    On the very first call (no existing EMA) initialises directly from scores.
    """
    current = _get_ema(conn)
    if current is None:
        # Bootstrap: first exchange seeds the EMA directly
        new_ema = {d: float(scores.get(d, 0.5)) for d in DIMS}
    else:
        new_ema = {
            d: EMA_ALPHA * float(scores.get(d, 0.5)) + (1.0 - EMA_ALPHA) * current[d]
            for d in DIMS
        }
    for d in DIMS:
        conn.execute(
            "INSERT OR REPLACE INTO ema_state (key, value) VALUES (?, ?)",
            (f"ema_{d}", round(new_ema[d], 6))
        )


# ── Public write API ──────────────────────────────────────────────────────────

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


def update_scores(exchange_id: str, scores: dict) -> None:
    """
    Write evaluation scores to numeric columns AND update the EMA vector.
    Fix 1: EMA updated here so every scored exchange shifts the signal.
    """
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
        # Only update EMA for the 5 numeric dims (not 'notes')
        numeric = {d: scores[d] for d in DIMS if d in scores}
        if len(numeric) == len(DIMS):
            _update_ema(conn, numeric)
        conn.commit()


# ── Public read API ───────────────────────────────────────────────────────────

def get_average_vector(limit: int = 500) -> dict | None:
    """
    Return the EMA payoff vector (Fix 1).

    Falls back to the SQL average on first use (before any EMA is computed),
    then bootstraps the EMA table from that average so subsequent calls
    return the rolling EMA.

    Returns None if no scored exchanges exist at all.
    """
    init_db()
    with _connect() as conn:
        ema = _get_ema(conn)
        if ema is not None:
            return {d: round(ema[d], 4) for d in DIMS}

        # No EMA yet — bootstrap from SQL average
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

        sql_avg = {d: round(row[d], 4) for d in DIMS}
        # Seed the EMA table so next call uses EMA path
        _update_ema(conn, sql_avg)
        conn.commit()
        return sql_avg


def get_sql_average_vector(limit: int = 500) -> dict | None:
    """
    Return the plain SQL AVG (audit / debugging use).
    Not used for Oracle triggering — use get_average_vector() for that.
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


def get_recent_exchange_ids(n: int = 50) -> list[str]:
    """
    Return the IDs of the *n* most recent scored exchanges.
    Used by background_eval to tag which exchanges drove an Oracle trigger,
    so lora_steer can load only those exchanges as training context.
    """
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT id FROM exchanges
               WHERE v_accuracy IS NOT NULL
               ORDER BY timestamp DESC
               LIMIT ?""",
            (n,)
        ).fetchall()
    return [r["id"] for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"DB: {os.path.abspath(DB_PATH)}")
    avg = get_average_vector()
    sql_avg = get_sql_average_vector()
    if avg:
        print(f"EMA vector : {avg}")
        print(f"SQL average: {sql_avg}")
    else:
        print("No scored exchanges yet.")
