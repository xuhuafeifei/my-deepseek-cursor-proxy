from __future__ import annotations

from pathlib import Path
import sqlite3
import threading
import time
from typing import Any


CACHE_MAX_ROWS = 100
CACHE_MAX_AGE_SECONDS = 86400  # 24h


class ReasoningStore:
    def __init__(
        self,
        reasoning_content_path: str | Path,
        max_age_seconds: int | None = None,
        max_rows: int | None = None,
    ) -> None:
        self.max_age_seconds = max_age_seconds or CACHE_MAX_AGE_SECONDS
        self.max_rows = max_rows or CACHE_MAX_ROWS
        if str(reasoning_content_path) == ":memory:":
            self.reasoning_content_path: str | Path = ":memory:"
        else:
            self.reasoning_content_path = Path(reasoning_content_path).expanduser()
            self.reasoning_content_path.parent.mkdir(
                mode=0o700, parents=True, exist_ok=True
            )
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.reasoning_content_path, check_same_thread=False
        )
        if isinstance(self.reasoning_content_path, Path):
            self.reasoning_content_path.chmod(0o600)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reasoning_cache (
                key TEXT PRIMARY KEY,
                reasoning TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        # Migrate old schema if needed
        columns = [
            row[1]
            for row in self._conn.execute("PRAGMA table_info(reasoning_cache)")
        ]
        if "session" in columns:
            self._conn.execute(
                "ALTER TABLE reasoning_cache DROP COLUMN session"
            )
        if "message_json" in columns:
            self._conn.execute(
                "ALTER TABLE reasoning_cache DROP COLUMN message_json"
            )
        # Delete old-format keys from before tool_call_id refactor
        self._conn.execute(
            "DELETE FROM reasoning_cache WHERE key NOT LIKE 'tool:%'"
        )
        self._conn.commit()
        self.prune()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def store_assistant_message(self, message: dict[str, Any]) -> int:
        """Store reasoning_content keyed by the first tool_call_id in the message."""
        if message.get("role") != "assistant":
            return 0
        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str):
            return 0
        tool_calls = [
            tc for tc in (message.get("tool_calls") or [])
            if isinstance(tc, dict) and tc.get("id")
        ]
        if not tool_calls:
            return 0

        key = f"tool:{tool_calls[0]['id']}"
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO reasoning_cache(key, reasoning, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    reasoning = excluded.reasoning,
                    created_at = excluded.created_at
                """,
                (key, reasoning, time.time()),
            )
            self._prune_locked()
            self._conn.commit()
        return 1

    def lookup_for_message(self, message: dict[str, Any], session: str = "") -> str | None:
        """Look up reasoning_content by the first tool_call_id in the message."""
        tool_calls = [
            tc for tc in (message.get("tool_calls") or [])
            if isinstance(tc, dict) and tc.get("id")
        ]
        if not tool_calls:
            return None
        return self.get(f"tool:{tool_calls[0]['id']}")

    def list_entries(self) -> list[tuple[str, int, str]]:
        """Return (key, length, preview) for all cached entries."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, reasoning FROM reasoning_cache ORDER BY created_at ASC"
            ).fetchall()
        return [(r[0], len(r[1]), r[1][:100]) for r in rows]

    def clear(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM reasoning_cache").fetchone()
            count = int(row[0] if row else 0)
            self._conn.execute("DELETE FROM reasoning_cache")
            self._conn.commit()
        return count

    def prune(self) -> int:
        with self._lock:
            deleted = self._prune_locked()
            self._conn.commit()
        return deleted

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT reasoning FROM reasoning_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def _prune_locked(self) -> int:
        deleted = 0
        cutoff = time.time() - self.max_age_seconds
        cursor = self._conn.execute(
            "DELETE FROM reasoning_cache WHERE created_at < ?",
            (cutoff,),
        )
        deleted += cursor.rowcount if cursor.rowcount != -1 else 0

        cursor = self._conn.execute(
            """
            DELETE FROM reasoning_cache
            WHERE key NOT IN (
                SELECT key FROM reasoning_cache ORDER BY created_at DESC LIMIT ?
            )
            """,
            (self.max_rows,),
        )
        deleted += cursor.rowcount if cursor.rowcount != -1 else 0
        return deleted
