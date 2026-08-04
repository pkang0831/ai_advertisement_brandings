from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from .capabilities import ReviewProfile, assert_profile_isolation


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS review_posts (
    post_id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    caption TEXT NOT NULL DEFAULT '',
    location_label TEXT NOT NULL DEFAULT '',
    audience_tiers_json TEXT NOT NULL DEFAULT '[]',
    selected_asset_ids_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'needs_review',
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS review_candidates (
    asset_id TEXT PRIMARY KEY,
    post_id TEXT NOT NULL REFERENCES review_posts(post_id),
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    decision TEXT,
    reject_reason TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS review_approvals (
    approval_id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id TEXT NOT NULL REFERENCES review_posts(post_id),
    approval_type TEXT NOT NULL CHECK (approval_type IN ('content', 'schedule')),
    snapshot_hash TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS regeneration_requests (
    request_id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id TEXT NOT NULL REFERENCES review_posts(post_id),
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'requested'
);
CREATE TRIGGER IF NOT EXISTS review_approvals_no_update
BEFORE UPDATE ON review_approvals BEGIN SELECT RAISE(ABORT, 'approvals are immutable'); END;
CREATE TRIGGER IF NOT EXISTS review_approvals_no_delete
BEFORE DELETE ON review_approvals BEGIN SELECT RAISE(ABORT, 'approvals are immutable'); END;
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReviewStore:
    def __init__(self, db_path: Path, profile: ReviewProfile) -> None:
        assert_profile_isolation(profile, str(db_path))
        self.db_path = db_path.expanduser().resolve()
        self.profile = profile
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def upsert_post(
        self,
        post_id: str,
        platform: str,
        caption: str = "",
        location_label: str = "",
        audience_tiers: Sequence[str] = (),
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO review_posts
                    (post_id, platform, caption, location_label, audience_tiers_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(post_id) DO UPDATE SET
                    platform=excluded.platform, caption=excluded.caption,
                    location_label=excluded.location_label,
                    audience_tiers_json=excluded.audience_tiers_json,
                    updated_at=excluded.updated_at
                """,
                (
                    post_id,
                    platform,
                    caption,
                    location_label,
                    json.dumps(list(audience_tiers), separators=(",", ":")),
                    _now(),
                ),
            )

    def add_candidate(self, post_id: str, asset_id: str, path: Path) -> None:
        resolved = path.expanduser().resolve()
        digest = file_sha256(resolved)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO review_candidates
                    (asset_id, post_id, path, sha256, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (asset_id, post_id, str(resolved), digest, _now()),
            )

    def list_posts(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(connection.execute("SELECT * FROM review_posts ORDER BY post_id"))

    def get_post(self, post_id: str) -> sqlite3.Row:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM review_posts WHERE post_id=?", (post_id,)
            ).fetchone()
        if row is None:
            raise KeyError(post_id)
        return row

    def candidates(self, post_id: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM review_candidates WHERE post_id=? ORDER BY asset_id",
                    (post_id,),
                )
            )

    def decide_candidate(
        self, post_id: str, asset_id: str, decision: str, reason: str = ""
    ) -> None:
        if decision not in {"approved", "rejected"}:
            raise ValueError("decision must be approved or rejected")
        if decision == "rejected" and not reason.strip():
            raise ValueError("A rejection reason is required")
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE review_candidates SET decision=?, reject_reason=?
                WHERE post_id=? AND asset_id=?
                """,
                (decision, reason.strip() or None, post_id, asset_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(asset_id)
            selected = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT asset_id FROM review_candidates
                    WHERE post_id=? AND decision='approved' ORDER BY asset_id
                    """,
                    (post_id,),
                )
            ]
            connection.execute(
                """
                UPDATE review_posts
                SET selected_asset_ids_json=?, status='needs_review', updated_at=?
                WHERE post_id=?
                """,
                (json.dumps(selected, separators=(",", ":")), _now(), post_id),
            )

    def update_content(
        self,
        post_id: str,
        caption: str,
        location_label: str,
        audience_tiers: Sequence[str],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE review_posts SET caption=?, location_label=?,
                    audience_tiers_json=?, status='needs_review', updated_at=?
                WHERE post_id=?
                """,
                (
                    caption,
                    location_label,
                    json.dumps(list(audience_tiers), separators=(",", ":")),
                    _now(),
                    post_id,
                ),
            )

    def immutable_hash(self, post_id: str) -> str:
        post = self.get_post(post_id)
        selected = json.loads(post["selected_asset_ids_json"])
        candidates = {row["asset_id"]: row for row in self.candidates(post_id)}
        assets = []
        for asset_id in selected:
            row = candidates[asset_id]
            current_hash = file_sha256(Path(row["path"]))
            assets.append(
                {"asset_id": asset_id, "registered_sha256": row["sha256"], "sha256": current_hash}
            )
        snapshot = {
            "post_id": post_id,
            "platform": post["platform"],
            "caption": post["caption"],
            "location_label": post["location_label"],
            "audience_tiers": json.loads(post["audience_tiers_json"]),
            "assets": assets,
        }
        encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def asset_integrity_issues(self, post_id: str) -> list[str]:
        post = self.get_post(post_id)
        selected = json.loads(post["selected_asset_ids_json"])
        if not selected:
            return ["No candidate assets are selected"]
        candidates = {row["asset_id"]: row for row in self.candidates(post_id)}
        issues: list[str] = []
        for asset_id in selected:
            row = candidates.get(asset_id)
            if row is None:
                issues.append(f"{asset_id}: candidate record is missing")
                continue
            path = Path(row["path"])
            if not path.is_file():
                issues.append(f"{asset_id}: asset file is missing")
            elif file_sha256(path) != row["sha256"]:
                issues.append(f"{asset_id}: asset hash differs from its registered hash")
        return issues

    def approve(self, post_id: str, approval_type: str, reviewer: str) -> str:
        if approval_type not in {"content", "schedule"}:
            raise ValueError("Unknown approval type")
        if approval_type == "schedule" and self.profile is ReviewProfile.MATURE:
            raise PermissionError("Mature profile has no schedule approval capability")
        integrity_issues = self.asset_integrity_issues(post_id)
        if integrity_issues:
            raise ValueError("; ".join(integrity_issues))
        snapshot_hash = self.immutable_hash(post_id)
        if approval_type == "schedule" and not self.approval_valid(post_id, "content"):
            raise ValueError("Valid content approval is required before schedule approval")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO review_approvals
                    (post_id, approval_type, snapshot_hash, reviewer, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (post_id, approval_type, snapshot_hash, reviewer, _now()),
            )
            status = "schedule_approved" if approval_type == "schedule" else "content_approved"
            connection.execute(
                "UPDATE review_posts SET status=?, updated_at=? WHERE post_id=?",
                (status, _now(), post_id),
            )
        return snapshot_hash

    def approval_valid(self, post_id: str, approval_type: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT snapshot_hash FROM review_approvals
                WHERE post_id=? AND approval_type=?
                ORDER BY approval_id DESC LIMIT 1
                """,
                (post_id, approval_type),
            ).fetchone()
        if row is None or self.asset_integrity_issues(post_id):
            return False
        try:
            return row["snapshot_hash"] == self.immutable_hash(post_id)
        except (FileNotFoundError, KeyError):
            return False

    def request_regeneration(self, post_id: str, reason: str) -> None:
        if not reason.strip():
            raise ValueError("A regeneration reason is required")
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO regeneration_requests (post_id, reason, created_at) VALUES (?, ?, ?)",
                (post_id, reason.strip(), _now()),
            )
            connection.execute(
                "UPDATE review_posts SET status='needs_review', updated_at=? WHERE post_id=?",
                (_now(), post_id),
            )
