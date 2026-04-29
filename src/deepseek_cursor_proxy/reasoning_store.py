from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any


SESSION_MAX_THINKING = 5


def _normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function") or {}
    if not isinstance(function, dict):
        function = {}
    arguments = function.get("arguments", "")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    return {
        "id": tool_call.get("id"),
        "type": tool_call.get("type") or "function",
        "function": {
            "name": function.get("name") or "",
            "arguments": arguments,
        },
    }


def session_key(messages: list[dict[str, Any]], namespace: str = "") -> str:
    first_user = next(
        (m for m in messages if isinstance(m, dict) and m.get("role") == "user"),
        None,
    )
    if first_user is None:
        return ""
    content = first_user.get("content") or ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                parts.append(str(text))
            else:
                parts.append(str(item))
        content = "\n".join(parts)
    payload = {"namespace": namespace, "content": str(content)}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def assistant_message_signature(message: dict[str, Any]) -> str:
    tool_calls = [
        _normalize_tool_call(tc)
        for tc in (message.get("tool_calls") or [])
        if isinstance(tc, dict)
    ]
    payload = {
        "content": message.get("content") or "",
        "tool_calls": tool_calls,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ReasoningStore:
    def __init__(
        self,
        reasoning_content_path: str | Path,
        max_age_seconds: int | None = None,
        max_rows: int | None = None,
        session_max_thinking: int = SESSION_MAX_THINKING,
    ) -> None:
        self.max_age_seconds = max_age_seconds
        self.max_rows = max_rows
        self.session_max_thinking = session_max_thinking
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
                session TEXT NOT NULL DEFAULT '',
                reasoning TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        # Migrate old schema
        columns = [
            row[1]
            for row in self._conn.execute("PRAGMA table_info(reasoning_cache)")
        ]
        if "session" not in columns:
            self._conn.execute(
                "ALTER TABLE reasoning_cache ADD COLUMN session TEXT DEFAULT ''"
            )
        if "message_json" in columns:
            self._conn.execute(
                "ALTER TABLE reasoning_cache DROP COLUMN message_json"
            )
        self._conn.commit()
        self._conn.commit()
        self.prune()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def put(self, key: str, session: str, reasoning: str) -> None:
        if not isinstance(reasoning, str):
            return
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO reasoning_cache(key, session, reasoning, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    reasoning = excluded.reasoning,
                    created_at = excluded.created_at
                """,
                (key, session, reasoning, time.time()),
            )
            self._prune_locked()
            self._conn.commit()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT reasoning FROM reasoning_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def store_assistant_message(self, message: dict[str, Any], session: str) -> int:
        if message.get("role") != "assistant":
            return 0
        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str):
            return 0

        sig = assistant_message_signature(message)
        key = f"session:{session}:message:{sig}"
        self.put(key, session, reasoning)
        return 1

    def lookup_for_message(self, message: dict[str, Any], session: str) -> str | None:
        sig = assistant_message_signature(message)
        return self.get(f"session:{session}:message:{sig}")

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

    def _prune_locked(self) -> int:
        deleted = 0
        if self.max_age_seconds is not None and self.max_age_seconds > 0:
            cutoff = time.time() - self.max_age_seconds
            cursor = self._conn.execute(
                "DELETE FROM reasoning_cache WHERE created_at < ?",
                (cutoff,),
            )
            deleted += cursor.rowcount if cursor.rowcount != -1 else 0

        if self.max_rows is not None and self.max_rows > 0:
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

        # Per-session limit: keep latest N thinking entries per session
        if self.session_max_thinking is not None and self.session_max_thinking > 0:
            cursor = self._conn.execute(
                """
                DELETE FROM reasoning_cache
                WHERE rowid NOT IN (
                    SELECT rowid FROM (
                        SELECT rowid,
                               ROW_NUMBER() OVER (PARTITION BY session ORDER BY created_at DESC) AS rn
                        FROM reasoning_cache
                    ) WHERE rn <= ?
                )
                """,
                (self.session_max_thinking,),
            )
            deleted += cursor.rowcount if cursor.rowcount != -1 else 0
        return deleted
