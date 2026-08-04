"""Fail-closed local-only mature_non_explicit storage.

This API intentionally has no import or export path to the platform manifest.
It never accepts publisher credentials and exposes no publication operation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
from pathlib import Path
from typing import Any

from .manifest import utc_now

RINA = Path(__file__).resolve().parents[1]
# Schema/presets live under private/ (gitignored); legacy factory/ path kept as fallback.
_SCHEMA_CANDIDATES = (
    RINA / "private" / "factory" / "mature_non_explicit" / "001_schema.sql",
    Path(__file__).with_name("mature_non_explicit") / "001_schema.sql",
)
SCHEMA = next((p for p in _SCHEMA_CANDIDATES if p.is_file()), _SCHEMA_CANDIDATES[0])
FORBIDDEN_TERMS = {
    "nude", "nudity", "naked", "visible nipples", "genitals", "sexual act",
    "sex act", "masturbation", "sexual fluids", "minor", "underage", "teen",
    "child", "coercion", "forced", "non-consensual",
}
ALLOWED_CONTEXT = {"adult", "lingerie", "swimwear", "sensual"}


class MatureStore:
    def __init__(self, db_path: os.PathLike[str] | str, root: os.PathLike[str] | str):
        self.db_path = Path(db_path)
        self.root = Path(root)
        if self.db_path.resolve() == self.root.resolve():
            raise ValueError("database and media root must be separate")

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA.read_text())

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    @staticmethod
    def validate_prompt(prompt: str) -> None:
        normalized = " ".join(prompt.lower().replace("_", " ").split())
        if re.search(r"\badult\b", normalized) is None:
            raise ValueError("mature prompt must explicitly identify an adult")
        matched = sorted(
            term for term in FORBIDDEN_TERMS
            if re.search(rf"\b{re.escape(term)}\b", normalized)
        )
        if matched:
            raise ValueError(f"forbidden mature content: {', '.join(matched)}")

    def register(
        self, local_id: str, path: os.PathLike[str] | str, media_type: str,
        expected_sha256: str | None = None,
    ) -> str:
        if not local_id.startswith("mne_"):
            raise ValueError("mature IDs must use the isolated mne_ namespace")
        candidate = Path(path)
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("symlinks are forbidden")
        resolved = candidate.resolve(strict=True)
        root = self.root.resolve(strict=True)
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("mature asset escapes isolated root") from exc
        info = resolved.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("mature asset must be a regular file")
        if info.st_nlink != 1:
            raise ValueError("hardlinks are forbidden")
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("asset hash does not match registration")
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO mature_assets VALUES(?,?,?,?,?,?)",
                (local_id, digest, str(relative), media_type,
                 "mature_non_explicit", utc_now()),
            )
            conn.execute(
                "INSERT INTO mature_audit(event_type,local_id,payload_json,created_at)"
                " VALUES(?,?,?,?)",
                ("asset_registered", local_id,
                 json.dumps({"sha256": digest}, sort_keys=True), utc_now()),
            )
        return digest

    def verify_registered(self, local_id: str) -> Path:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM mature_assets WHERE local_id=?", (local_id,)
            ).fetchone()
        if row is None:
            raise ValueError("unregistered mature asset")
        path = self.root / row["relative_path"]
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
            raise ValueError("registered asset link safety changed")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(self.root.resolve(strict=True))
        except ValueError as exc:
            raise ValueError("registered asset escaped isolated root") from exc
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError("registered asset hash changed")
        return resolved

    def assert_no_platform_context(
        self, platform_db: os.PathLike[str] | str | None = None,
        publisher_secret: str | None = None,
    ) -> None:
        if platform_db is not None or publisher_secret is not None:
            raise ValueError("mature process rejects all platform/publisher context")


def mature_profile(root: Path, db_path: Path) -> dict[str, Any]:
    return {
        "profile": "mature_non_explicit",
        "root": str(root),
        "database": str(db_path),
        "mode": "0700",
        "platform_upload": False,
        "allowed": ["adult lingerie", "adult swimwear", "sensual non-explicit"],
        "forbidden": sorted(FORBIDDEN_TERMS),
    }
