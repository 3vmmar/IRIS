"""
Persistent visual memory store backed by SQLite (aiosqlite).

Write policy: confidence threshold + consecutive-frame streak + deduplication window.
Provides formatted context strings for injection into Claude system prompts.

Allowed imports: config.settings, aiosqlite, datetime, standard library.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import aiosqlite

from config import settings

logger = logging.getLogger(__name__)

# ── SQL statements ─────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS sightings (
    id          INTEGER  PRIMARY KEY AUTOINCREMENT,
    object      TEXT     NOT NULL,
    zone        TEXT     NOT NULL,
    x_rel       REAL     NOT NULL,
    y_rel       REAL     NOT NULL,
    confidence  REAL     NOT NULL,
    seen_at     TEXT     NOT NULL,
    session_id  TEXT     NOT NULL
);
"""

_PRAGMA_WAL = "PRAGMA journal_mode=WAL;"

_SELECT_RECENT = """
SELECT object, zone, x_rel, y_rel, confidence, seen_at
FROM   sightings
WHERE  object     = ?
AND    zone       = ?
AND    session_id = ?
AND    seen_at   >= ?
LIMIT  1;
"""

_UPDATE_SIGHTING = """
UPDATE sightings
SET    seen_at    = ?,
       x_rel      = ?,
       y_rel      = ?,
       confidence = ?
WHERE  object     = ?
AND    zone       = ?
AND    session_id = ?
AND    seen_at   >= ?;
"""

_INSERT_SIGHTING = """
INSERT INTO sightings (object, zone, x_rel, y_rel, confidence, seen_at, session_id)
VALUES (?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_LATEST_FOR_LABEL = """
SELECT object, zone, x_rel, y_rel, confidence, seen_at
FROM   sightings
WHERE  object     = ?
AND    session_id = ?
ORDER  BY seen_at DESC
LIMIT  1;
"""

_SELECT_RECENT_DISTINCT = """
SELECT object, zone, x_rel, y_rel, confidence, seen_at
FROM   sightings
WHERE  session_id = ?
GROUP  BY object
ORDER  BY MAX(seen_at) DESC
LIMIT  ?;
"""

_DELETE_SESSION = """
DELETE FROM sightings WHERE session_id = ?;
"""


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _cutoff_iso() -> str:
    """Return UTC timestamp for *now minus memory_window_minutes*."""
    delta = timedelta(minutes=settings.memory_window_minutes)
    return (datetime.now(timezone.utc) - delta).isoformat()


def _relative_time(seen_at_iso: str) -> str:
    """Convert an ISO-8601 UTC string to a human-readable relative time label.

    Args:
        seen_at_iso: ISO-8601 UTC timestamp string.

    Returns:
        A string like ``"just now"``, ``"42 seconds ago"``, ``"5 minutes ago"``.
    """
    try:
        seen_dt = datetime.fromisoformat(seen_at_iso)
        # Make timezone-aware if naive (legacy rows)
        if seen_dt.tzinfo is None:
            seen_dt = seen_dt.replace(tzinfo=timezone.utc)
        delta_s = (datetime.now(timezone.utc) - seen_dt).total_seconds()
    except (ValueError, TypeError):
        return "unknown time"

    if delta_s < 10:
        return "just now"
    if delta_s < 60:
        return f"{int(delta_s)} seconds ago"
    if delta_s < 3600:
        mins = int(delta_s // 60)
        return f"{mins} minute{'s' if mins > 1 else ''} ago"
    hrs = int(delta_s // 3600)
    return f"{hrs} hour{'s' if hrs > 1 else ''} ago"


# ── Main class ────────────────────────────────────────────────────────────────

class VisualMemory:
    """Persistent visual memory store backed by a local SQLite database.

    Tracks which objects IRIS has seen, where, and when — so Claude can answer
    questions about objects that are no longer in frame.

    **Write policy** — a sighting is persisted only when all three hold:

    1. Detection confidence ≥ ``settings.memory_min_confidence``
    2. The same label has appeared in ≥ ``settings.memory_streak_frames``
       consecutive frames (tracked via an in-memory ``_streak`` dict)
    3. No record exists for the same ``label + zone + session_id`` within the
       last ``settings.memory_window_minutes`` minutes

    When condition 3 finds a *recent* duplicate, the existing row is **updated**
    (bumping ``seen_at``, ``x_rel``, ``y_rel``, ``confidence``) rather than
    inserting a new one, keeping the table compact.

    Usage::

        mem = VisualMemory(session_id="session-abc")
        await mem.open()

        wrote = await mem.observe("laptop", "center", 0.5, 0.5, 0.91)

        context = await mem.recent_context()   # inject into Claude prompt

        await mem.close()
    """

    def __init__(self, session_id: str = "default") -> None:
        """
        Args:
            session_id: Tag attached to every row so multi-session data can be
                        queried or cleared independently.
        """
        self._session_id: str = session_id
        self._streak: dict[str, int] = {}         # label → consecutive-frame count
        self._db: Optional[aiosqlite.Connection] = None

    # ─────────────────────────── lifecycle ────────────────────────────────────

    async def open(self) -> None:
        """Open the SQLite connection and create the sightings table if absent.

        Also enables WAL journal mode for safer concurrent reads alongside
        the FastAPI server.

        Must be awaited before any other async method.
        """
        self._db = await aiosqlite.connect(settings.memory_db_path)
        await self._db.execute(_DDL)
        await self._db.execute(_PRAGMA_WAL)
        await self._db.commit()
        logger.info(
            "VisualMemory opened (db='%s', session='%s').",
            settings.memory_db_path, self._session_id,
        )

    async def close(self) -> None:
        """Close the SQLite connection if it is open."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            logger.info("VisualMemory closed (session='%s').", self._session_id)

    # ─────────────────────────── write path ───────────────────────────────────

    async def observe(
        self,
        label: str,
        zone: str,
        x_rel: float,
        y_rel: float,
        confidence: float,
    ) -> bool:
        """Evaluate one detection against the write policy and persist if qualified.

        Call this once per detected object per inference frame. The streak
        counter for *label* is incremented on every call where confidence
        passes; reset to zero when confidence fails.

        Args:
            label:      YOLO class name (e.g. ``"laptop"``).
            zone:       Named 3×3 grid zone (e.g. ``"center"``).
            x_rel:      Normalised horizontal centroid (0.0–1.0).
            y_rel:      Normalised vertical centroid (0.0–1.0).
            confidence: YOLO detection confidence (0.0–1.0).

        Returns:
            ``True`` if a new row was inserted or an existing row was updated.
            ``False`` in all other cases (confidence too low, streak not met,
            duplicate within the window, or DB not open).
        """
        assert self._db is not None, "Call open() before observe()."

        # ── Condition 1: confidence threshold ────────────────────────────────
        if confidence < settings.memory_min_confidence:
            self._streak[label] = 0   # reset streak on low-confidence frame
            return False

        # ── Condition 2: consecutive-frame streak ─────────────────────────────
        self._streak[label] = self._streak.get(label, 0) + 1
        if self._streak[label] < settings.memory_streak_frames:
            return False

        # ── Condition 3: deduplication window ─────────────────────────────────
        cutoff = _cutoff_iso()
        async with self._db.execute(
            _SELECT_RECENT,
            (label, zone, self._session_id, cutoff),
        ) as cur:
            row = await cur.fetchone()

        now = _now_iso()

        if row is not None:
            # Existing recent record — UPDATE to bump timestamp and position
            await self._db.execute(
                _UPDATE_SIGHTING,
                (now, x_rel, y_rel, confidence,
                 label, zone, self._session_id, cutoff),
            )
            await self._db.commit()
            logger.debug(
                "Memory UPDATE: '%s' in '%s' (conf=%.2f).", label, zone, confidence,
            )
        else:
            # No recent record — INSERT new row
            await self._db.execute(
                _INSERT_SIGHTING,
                (label, zone, x_rel, y_rel, confidence, now, self._session_id),
            )
            await self._db.commit()
            logger.info(
                "Memory INSERT: '%s' in '%s' (conf=%.2f, streak=%d).",
                label, zone, confidence, self._streak[label],
            )

        # Reset streak after a successful write so the next write requires
        # another full streak of consecutive frames.
        self._streak[label] = 0
        return True

    # ─────────────────────────── read path ────────────────────────────────────

    async def query_object(self, label: str) -> Optional[dict]:
        """Return the most recent sighting of *label* for this session.

        Args:
            label: Object class name to look up (e.g. ``"keys"``).

        Returns:
            A dict with keys ``object``, ``zone``, ``x_rel``, ``y_rel``,
            ``confidence``, ``seen_at`` — or ``None`` if no sighting exists.
        """
        assert self._db is not None, "Call open() before query_object()."

        async with self._db.execute(
            _SELECT_LATEST_FOR_LABEL,
            (label, self._session_id),
        ) as cur:
            row = await cur.fetchone()

        if row is None:
            return None

        return {
            "object":     row[0],
            "zone":       row[1],
            "x_rel":      row[2],
            "y_rel":      row[3],
            "confidence": row[4],
            "seen_at":    row[5],
        }

    async def recent_context(self, limit: int = 10) -> str:
        """Build a human-readable memory summary for injection into Claude prompts.

        Queries the *limit* most recent **distinct** object sightings (one row
        per object label) and formats them with relative timestamps.

        Args:
            limit: Maximum number of distinct objects to include.

        Returns:
            A multi-line string ready to embed verbatim in the Claude system
            prompt, for example::

                [Visual Memory — last 10 observations]
                - laptop: center zone, 2 minutes ago
                - phone: bottom-right zone, 8 minutes ago
                - keys: left zone, 14 minutes ago

            Returns a "no objects" notice when no sightings exist.
        """
        assert self._db is not None, "Call open() before recent_context()."

        rows: list[tuple] = []
        async with self._db.execute(
            _SELECT_RECENT_DISTINCT,
            (self._session_id, limit),
        ) as cur:
            async for row in cur:
                rows.append(row)

        if not rows:
            return "[Visual Memory]\nNo objects recorded in this session."

        lines = [f"[Visual Memory — last {len(rows)} observations]"]
        for row in rows:
            label, zone, *_, seen_at = row
            rel_time = _relative_time(seen_at)
            lines.append(f"- {label}: {zone} zone, {rel_time}")

        return "\n".join(lines)

    # ─────────────────────────── maintenance ──────────────────────────────────

    async def clear_session(self) -> None:
        """Delete all sighting rows for the current session and reset streaks.

        Useful when the user starts a fresh session or explicitly asks IRIS
        to forget everything it has seen.
        """
        assert self._db is not None, "Call open() before clear_session()."
        await self._db.execute(_DELETE_SESSION, (self._session_id,))
        await self._db.commit()
        self._streak = {}
        logger.info("VisualMemory cleared (session='%s').", self._session_id)

    def reset_streaks(self) -> None:
        """Clear the in-memory streak counters without touching the database.

        Called by the scene change detector on a ``force_reset()`` so that
        objects must re-accumulate their consecutive-frame streak before a
        new sighting is written.
        """
        self._streak = {}
        logger.debug("VisualMemory streaks reset (session='%s').", self._session_id)
