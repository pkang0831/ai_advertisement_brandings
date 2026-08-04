"""Structured JSONL logging with redaction and local retention."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SECRET_KEY = re.compile(
    r"(access[_-]?token|authorization|app[_-]?secret|password|cookie)",
    re.IGNORECASE,
)
BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if SECRET_KEY.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return BEARER.sub("Bearer [REDACTED]", value)
    return value


class JsonLogger:
    def __init__(self, root: Path, retention_days: int = 14) -> None:
        self.root = root
        self.retention_days = retention_days
        self.root.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> Path:
        now = datetime.now(timezone.utc)
        path = self.root / f"orchestrator-{now.date().isoformat()}.jsonl"
        record = redact(
            {
                "timestamp": now.isoformat().replace("+00:00", "Z"),
                "event": event,
                **fields,
            }
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self.prune(now)
        return path

    def prune(self, now: datetime | None = None) -> int:
        cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=self.retention_days)
        removed = 0
        for path in self.root.glob("orchestrator-*.jsonl"):
            modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            if modified < cutoff:
                path.unlink()
                removed += 1
        return removed
