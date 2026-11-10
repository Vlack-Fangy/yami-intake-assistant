"""
Async database layer for psychiatric history bot.
Uses aiosqlite for non-blocking SQLite operations.
"""

import aiosqlite
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any
import os
from data_model import SECTIONS_ORDER

DATABASE_URL = os.getenv("DATABASE_URL", "./psych_history.db")


class Session:
    """Data class for session info."""
    def __init__(self, session_id: str, patient_name: Optional[str],
                 preferred_language: str, current_section: str,
                 started_at: str, last_active: str, status: str,
                 progress_message_id: Optional[int],
                 last_topic_asked: Optional[str] = None):
        self.session_id = session_id
        self.patient_name = patient_name
        self.preferred_language = preferred_language
        self.current_section = current_section
        self.started_at = started_at
        self.last_active = last_active
        self.status = status
        self.progress_message_id = progress_message_id
        self.last_topic_asked = last_topic_asked


async def init_db() -> None:
    """Initialize database schema."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        await db.executescript("""
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    patient_name TEXT,
    preferred_language TEXT DEFAULT 'en',
    current_section TEXT DEFAULT 'sociodemographic',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP,
    status TEXT DEFAULT 'active',
    progress_message_id INTEGER,
    last_topic_asked TEXT DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS info_fields (
    session_id TEXT,
    section TEXT,
    field_name TEXT,
    value TEXT,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, section, field_name)
);

CREATE TABLE IF NOT EXISTS section_notes (
    session_id TEXT,
    section TEXT,
    note TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS section_summaries (
    session_id TEXT,
    section TEXT,
    summary_text TEXT,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, section)
);

CREATE TABLE IF NOT EXISTS conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    role TEXT,
    content TEXT,
    original_content TEXT,
    section_at_time TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS message_feedback (
    review_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    message_id INTEGER NOT NULL,
    rating     TEXT    NOT NULL CHECK (rating IN ('up','down')),
    review     TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_feedback_session
    ON message_feedback(session_id, message_id);

CREATE TABLE IF NOT EXISTS clinician_alerts (
    alert_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    risk            TEXT,
    message         TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alerts_open
    ON clinician_alerts(acknowledged_at, alert_id);

CREATE TABLE IF NOT EXISTS progress_update_log (
    session_id TEXT PRIMARY KEY,
    last_update_ts TIMESTAMP
);

CREATE TABLE IF NOT EXISTS session_progress (
    session_id TEXT,
    section TEXT,
    completion_pct INTEGER DEFAULT 0,
    filled_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, section)
);
        """)
        await db.commit()

        # Migration for DBs created before this column existed.
        # CREATE TABLE IF NOT EXISTS above won't alter an already-existing
        # table, so existing deployments need this explicit ALTER.
        # Safe to run on every startup — ignores "duplicate column" if
        # it's already there.
        try:
            await db.execute(
                "ALTER TABLE conversation_history ADD COLUMN original_content TEXT"
            )
            await db.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise


async def create_session(chat_id: int) -> None:
    """Create or reset a session for a chat.

    This is idempotent: if the chat already exists (for example after a user
    deletes Telegram chat history and starts again), we reset only that chat's
    in-progress data instead of failing with a UNIQUE constraint error.
    """
    session_id = str(chat_id)
    now = datetime.now().isoformat()

    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        exists = await cursor.fetchone()

        if exists:
            # Reset this chat's session state and working tables.
            # This keeps the rest of the database untouched.
            await db.execute(
                """UPDATE sessions
                   SET patient_name = NULL,
                       preferred_language = 'en',
                       current_section = 'sociodemographic',
                       started_at = ?,
                       last_active = ?,
                       status = 'active',
                       progress_message_id = NULL,
                       last_topic_asked = NULL
                   WHERE session_id = ?""",
                (now, now, session_id)
            )

            await db.execute("DELETE FROM info_fields WHERE session_id = ?", (session_id,))
            await db.execute("DELETE FROM section_notes WHERE session_id = ?", (session_id,))
            await db.execute("DELETE FROM section_summaries WHERE session_id = ?", (session_id,))
            await db.execute("DELETE FROM conversation_history WHERE session_id = ?", (session_id,))
            # Feedback points at conversation_history rows that are going away;
            # leaving it would dangle against messages that no longer exist.
            await db.execute("DELETE FROM message_feedback WHERE session_id = ?", (session_id,))
            await db.execute("DELETE FROM clinician_alerts WHERE session_id = ?", (session_id,))
            await db.execute("DELETE FROM progress_update_log WHERE session_id = ?", (session_id,))
            await db.execute("DELETE FROM session_progress WHERE session_id = ?", (session_id,))
        else:
            await db.execute(
                """INSERT INTO sessions (session_id, last_active)
                   VALUES (?, ?)""",
                (session_id, now)
            )

        # Initialize 0% progress snapshot for all sections (both new + reset)
        for section in SECTIONS_ORDER:
            await db.execute(
                """INSERT OR REPLACE INTO session_progress
                   (session_id, section, completion_pct, filled_count, updated_at)
                   VALUES (?, ?, 0, 0, ?)""",
                (session_id, section, now)
            )

        await db.commit()


async def get_session(chat_id: int) -> Optional[Session]:
    """Retrieve session info."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (str(chat_id),)
        )
        row = await cursor.fetchone()
        if row:
            return Session(
                session_id=row[0],
                patient_name=row[1],
                preferred_language=row[2],
                current_section=row[3],
                started_at=row[4],
                last_active=row[5],
                status=row[6],
                progress_message_id=row[7],
                last_topic_asked=row[8] if len(row) > 8 else None
            )
    return None


async def set_current_section(chat_id: int, section: str) -> None:
    """Update current section."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        await db.execute(
            "UPDATE sessions SET current_section = ?, last_active = ? WHERE session_id = ?",
            (section, datetime.now().isoformat(), str(chat_id))
        )
        await db.commit()


async def set_session_status(chat_id: int, status: str) -> None:
    """Update session status (active|completed|paused|crisis)."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        await db.execute(
            "UPDATE sessions SET status = ?, last_active = ? WHERE session_id = ?",
            (status, datetime.now().isoformat(), str(chat_id))
        )
        await db.commit()


async def set_preferred_language(chat_id: int, language: Optional[str]) -> None:
    """Save patient's language preference. Pass None to mark as pending selection."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        await db.execute(
            "UPDATE sessions SET preferred_language = ? WHERE session_id = ?",
            (language, str(chat_id))
        )
        await db.commit()


async def save_field(chat_id: int, section: str, field: str, value: str) -> None:
    """Save or update a field value."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        await db.execute(
            """INSERT OR REPLACE INTO info_fields
               (session_id, section, field_name, value, collected_at)
               VALUES (?, ?, ?, ?, ?)""",
            (str(chat_id), section, field, value, datetime.now().isoformat())
        )
        if field == "name":
            await db.execute(
                "UPDATE sessions SET patient_name = ? WHERE session_id = ?",
                (value, str(chat_id))
            )
        await db.commit()


async def get_section_fields(chat_id: int, section: str) -> Dict[str, Optional[str]]:
    """Get all fields for a section."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            "SELECT field_name, value FROM info_fields WHERE session_id = ? AND section = ?",
            (str(chat_id), section)
        )
        rows = await cursor.fetchall()
        return {row[0]: row[1] for row in rows}


async def get_full_info_store(chat_id: int) -> Dict[str, Dict[str, Optional[str]]]:
    """Get all info organized by section."""
    result = {}
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            "SELECT section, field_name, value FROM info_fields WHERE session_id = ?",
            (str(chat_id),)
        )
        rows = await cursor.fetchall()
        for section, field_name, value in rows:
            if section not in result:
                result[section] = {}
            result[section][field_name] = value
    return result


async def get_section_data(chat_id: int, section: str) -> Dict[str, Any]:
    """Get all data for a section (fields + notes)."""
    fields = await get_section_fields(chat_id, section)
    
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            "SELECT note FROM section_notes WHERE session_id = ? AND section = ?",
            (str(chat_id), section)
        )
        rows = await cursor.fetchall()
        notes = [row[0] for row in rows]
    
    return {
        "fields": fields,
        "notes": notes
    }


async def save_message(
    chat_id: int,
    role: str,
    content: str,
    section: str,
    original_content: Optional[str] = None,
) -> int:
    """
    Save a message to conversation history.

    content stays English always (this is what the LLM's own history context
    reads back via get_recent_history — keeping it English regardless of the
    patient's display language is what keeps the model's conversation context
    stable across turns).

    original_content is only set for the 3 live-translated languages
    (ta/pa/bn) — it holds whatever was actually typed by the patient, or
    actually shown to them, in their own script. Left NULL for en/hi/hinglish,
    where content already IS the real thing said — no second copy needed.

    Returns the new row's id. Callers that don't need it can ignore the return
    value — the Telegram path does. The REST path uses it to tell a client which
    message a reply is, so feedback can be attached to it.
    """
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            """INSERT INTO conversation_history
               (session_id, role, content, original_content, section_at_time, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (str(chat_id), role, content, original_content, section, datetime.now().isoformat())
        )
        await db.commit()
        return cursor.lastrowid


async def get_recent_history(chat_id: int, limit: int = 10) -> list:
    """Get recent conversation history."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            """SELECT role, content FROM conversation_history
               WHERE session_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (str(chat_id), limit)
        )
        rows = await cursor.fetchall()
        # Reverse to get chronological order
        return [{"role": row[0], "content": row[1]} for row in reversed(rows)]


async def get_full_conversation_history(chat_id: int) -> list:
    """Get complete conversation history with metadata in chronological order.

    Ordered by id, not timestamp. Both messages of a turn are written within the
    same second and the timestamps tie, which left the transcript able to render
    the reply before the question. The autoincrement id is the true insertion
    order and never ties.

    message_id is the handle callers use to attach feedback to a specific reply.
    """
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            """SELECT id, role, content, original_content, section_at_time, timestamp
               FROM conversation_history
               WHERE session_id = ?
               ORDER BY id ASC""",
            (str(chat_id),)
        )
        rows = await cursor.fetchall()

    return [
        {
            "message_id": row[0],
            "role": row[1],
            "content": row[2],
            "original_content": row[3],
            "section": row[4],
            "timestamp": row[5],
        }
        for row in rows
    ]


# ── Message feedback ──────────────────────────────────────────────
# Thumbs up/down plus an optional written review, attached to one AI-generated
# message. Append-only by design: every submission is its own review with its
# own review_id, so several reviewers can rate the same reply and none of them
# overwrite each other. Reading the "current" verdict is the caller's job.

async def get_message(session_id: str, message_id: int) -> Optional[dict]:
    """One message, scoped to its session. None if it isn't that session's.

    Scoping by session rather than looking up the id alone is what stops one
    session's id being used to reach into another's transcript.
    """
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            """SELECT id, role, content, section_at_time
               FROM conversation_history
               WHERE id = ? AND session_id = ?""",
            (message_id, str(session_id))
        )
        row = await cursor.fetchone()
    if not row:
        return None
    return {"message_id": row[0], "role": row[1], "content": row[2], "section": row[3]}


async def save_feedback(
    session_id: str,
    message_id: int,
    rating: str,
    review: Optional[str] = None,
) -> int:
    """Record a rating; returns the new review_id."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            """INSERT INTO message_feedback
               (session_id, message_id, rating, review, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (str(session_id), message_id, rating, review, datetime.now().isoformat())
        )
        await db.commit()
        return cursor.lastrowid


def _feedback_row(row) -> dict:
    return {
        "review_id": row[0],
        "session_id": row[1],
        "message_id": row[2],
        "rating": row[3],
        "review": row[4],
        "created_at": row[5],
        "updated_at": row[6],
    }


_FEEDBACK_COLS = "review_id, session_id, message_id, rating, review, created_at, updated_at"


async def get_feedback_for_session(session_id: str) -> list:
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            f"""SELECT {_FEEDBACK_COLS} FROM message_feedback
                WHERE session_id = ? ORDER BY review_id ASC""",
            (str(session_id),)
        )
        rows = await cursor.fetchall()
    return [_feedback_row(r) for r in rows]


async def get_feedback_for_message(session_id: str, message_id: int) -> list:
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            f"""SELECT {_FEEDBACK_COLS} FROM message_feedback
                WHERE session_id = ? AND message_id = ? ORDER BY review_id ASC""",
            (str(session_id), message_id)
        )
        rows = await cursor.fetchall()
    return [_feedback_row(r) for r in rows]


async def get_feedback(session_id: str, review_id: int) -> Optional[dict]:
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            f"""SELECT {_FEEDBACK_COLS} FROM message_feedback
                WHERE review_id = ? AND session_id = ?""",
            (review_id, str(session_id))
        )
        row = await cursor.fetchone()
    return _feedback_row(row) if row else None


async def update_feedback(
    session_id: str,
    review_id: int,
    rating: Optional[str] = None,
    review: Optional[str] = None,
) -> bool:
    """Change a rating or add review text after the fact. False if not found.

    Only the fields actually supplied are written, so adding a review later does
    not blank out the rating that came with it.
    """
    sets, params = [], []
    if rating is not None:
        sets.append("rating = ?"); params.append(rating)
    if review is not None:
        sets.append("review = ?"); params.append(review)
    if not sets:
        return await get_feedback(session_id, review_id) is not None
    sets.append("updated_at = ?"); params.append(datetime.now().isoformat())
    params += [review_id, str(session_id)]

    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            f"UPDATE message_feedback SET {', '.join(sets)} "
            f"WHERE review_id = ? AND session_id = ?",
            tuple(params)
        )
        await db.commit()
        return cursor.rowcount > 0


async def delete_feedback(session_id: str, review_id: int) -> bool:
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            "DELETE FROM message_feedback WHERE review_id = ? AND session_id = ?",
            (review_id, str(session_id))
        )
        await db.commit()
        return cursor.rowcount > 0


async def save_section_note(chat_id: int, section: str, note: str) -> None:
    """Save free-text notes for a section."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        await db.execute(
            """INSERT INTO section_notes (session_id, section, note, created_at)
               VALUES (?, ?, ?, ?)""",
            (str(chat_id), section, note, datetime.now().isoformat())
        )
        await db.commit()


async def save_section_summary(chat_id: int, section: str, summary: str) -> None:
    """Save generated summary for a section."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        await db.execute(
            """INSERT OR REPLACE INTO section_summaries
               (session_id, section, summary_text, generated_at)
               VALUES (?, ?, ?, ?)""",
            (str(chat_id), section, summary, datetime.now().isoformat())
        )
        await db.commit()


async def get_all_summaries(chat_id: int) -> Dict[str, str]:
    """Get all section summaries."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            "SELECT section, summary_text FROM section_summaries WHERE session_id = ?",
            (str(chat_id),)
        )
        rows = await cursor.fetchall()
        return {row[0]: row[1] for row in rows}


async def get_last_progress_update(session_id: str) -> Optional[datetime]:
    """Get timestamp of last progress message update."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            "SELECT last_update_ts FROM progress_update_log WHERE session_id = ?",
            (session_id,)
        )
        row = await cursor.fetchone()
        if row and row[0]:
            return datetime.fromisoformat(row[0])
    return None


async def set_last_progress_update(session_id: str) -> None:
    """Record timestamp of progress message update."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        await db.execute(
            """INSERT OR REPLACE INTO progress_update_log
               (session_id, last_update_ts)
               VALUES (?, ?)""",
            (session_id, datetime.now().isoformat())
        )
        await db.commit()


async def set_progress_message_id(chat_id: int, message_id: int) -> None:
    """Store the Telegram message ID of the pinned progress message."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        await db.execute(
            "UPDATE sessions SET progress_message_id = ? WHERE session_id = ?",
            (message_id, str(chat_id))
        )
        await db.commit()


async def get_last_topic_asked(chat_id: int) -> Optional[str]:
    """Get the last topic discussed for a session."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            "SELECT last_topic_asked FROM sessions WHERE session_id = ?",
            (str(chat_id),)
        )
        row = await cursor.fetchone()
        return row[0] if row else None


async def set_last_topic_asked(chat_id: int, topic_id: Optional[str]) -> None:
    """Save the last topic discussed for a session."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        await db.execute(
            "UPDATE sessions SET last_topic_asked = ? WHERE session_id = ?",
            (topic_id, str(chat_id))
        )
        await db.commit()


async def update_progress_snapshot(chat_id: int, section: str, pct: int, filled: int) -> None:
    """Write current section completion to session_progress table."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        await db.execute(
            """INSERT OR REPLACE INTO session_progress
               (session_id, section, completion_pct, filled_count, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (str(chat_id), section, pct, filled, datetime.now().isoformat())
        )
        await db.commit()


async def get_progress_snapshot(chat_id: int) -> Dict[str, Dict]:
    """Get completion snapshot for all sections of a session."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            """SELECT section, completion_pct, filled_count, updated_at
               FROM session_progress WHERE session_id = ?""",
            (str(chat_id),)
        )
        rows = await cursor.fetchall()
        return {
            row[0]: {
                "completion_pct": row[1],
                "filled_count": row[2],
                "updated_at": row[3],
            }
            for row in rows
        }


async def delete_session(chat_id: int) -> None:
    """Hard delete all data for a session."""
    session_id = str(chat_id)
    async with aiosqlite.connect(DATABASE_URL) as db:
        await db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM info_fields WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM section_notes WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM section_summaries WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM conversation_history WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM message_feedback WHERE session_id = ?", (session_id,))
        # Alerts quote the patient ("Last message: ..."), so a hard delete has to
        # take them too — the privacy promise outranks keeping the audit trail.
        await db.execute("DELETE FROM clinician_alerts WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM progress_update_log WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM session_progress WHERE session_id = ?", (session_id,))
        await db.commit()


async def list_sessions(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list:
    """List sessions with optional status filter, paginated."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        if status:
            cursor = await db.execute(
                """SELECT session_id, patient_name, preferred_language,
                          current_section, started_at, last_active, status
                   FROM sessions WHERE status = ?
                   ORDER BY last_active DESC LIMIT ? OFFSET ?""",
                (status, limit, offset)
            )
        else:
            cursor = await db.execute(
                """SELECT session_id, patient_name, preferred_language,
                          current_section, started_at, last_active, status
                   FROM sessions
                   ORDER BY last_active DESC LIMIT ? OFFSET ?""",
                (limit, offset)
            )
        rows = await cursor.fetchall()
        return [
            {
                "session_id": row[0],
                "patient_name": row[1],
                "preferred_language": row[2],
                "current_section": row[3],
                "started_at": row[4],
                "last_active": row[5],
                "status": row[6],
            }
            for row in rows
        ]

# ── Clinician alerts ──────────────────────────────────────────────
# Raised by crisis.py during suicide-risk triage. On Telegram these are DMed to
# CLINICIAN_TELEGRAM_ID; over REST there is no chat to DM, so they are recorded
# here and served from /admin/alerts. Deleted with their session: the text
# quotes the patient ("Last message: ..."), so the privacy promise of
# DELETE /sessions/{id} outranks keeping the audit trail. An auditable record
# meant to outlive deletion needs a separate, de-identified table.

_ALERT_COLS = "alert_id, session_id, risk, message, created_at, acknowledged_at"


def _alert_row(row) -> dict:
    return {
        "alert_id": row[0],
        "session_id": row[1],
        "risk": row[2],
        "message": row[3],
        "created_at": row[4],
        "acknowledged_at": row[5],
    }


async def save_clinician_alert(session_id: str, message: str, risk: Optional[str] = None) -> int:
    """Record one alert; returns its alert_id."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            """INSERT INTO clinician_alerts (session_id, risk, message, created_at)
               VALUES (?, ?, ?, ?)""",
            (str(session_id), risk, message, datetime.now().isoformat())
        )
        await db.commit()
        return cursor.lastrowid


async def list_clinician_alerts(
    include_acknowledged: bool = False,
    limit: int = 100,
) -> list:
    """Open alerts first, newest first within each group."""
    where = "" if include_acknowledged else "WHERE acknowledged_at IS NULL"
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            f"""SELECT {_ALERT_COLS} FROM clinician_alerts {where}
                ORDER BY (acknowledged_at IS NOT NULL), alert_id DESC LIMIT ?""",
            (limit,)
        )
        rows = await cursor.fetchall()
    return [_alert_row(r) for r in rows]


async def acknowledge_clinician_alert(alert_id: int) -> Optional[dict]:
    """Mark one alert handled. None if it does not exist."""
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute(
            "UPDATE clinician_alerts SET acknowledged_at = ? WHERE alert_id = ?",
            (datetime.now().isoformat(), alert_id)
        )
        await db.commit()
        if cursor.rowcount == 0:
            return None
        cursor = await db.execute(
            f"SELECT {_ALERT_COLS} FROM clinician_alerts WHERE alert_id = ?", (alert_id,)
        )
        row = await cursor.fetchone()
    return _alert_row(row) if row else None
