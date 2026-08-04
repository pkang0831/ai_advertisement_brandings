"""Transactional platform manifest API.

State machine:
draft -> generating -> assets_ready -> content_approved ->
schedule_approved -> packaged -> publishing -> published.
needs_review, failed and needs_reconciliation are explicit exception states;
the database migration contains the complete allowed transition graph.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

UTC = timezone.utc
MIGRATIONS = Path(__file__).with_name("migrations")
PLATFORM_TRACKS = {"ig", "patreon_a", "patreon_b", "patreon_c"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class Manifest:
    def __init__(self, db_path: os.PathLike[str] | str, asset_root: os.PathLike[str] | str):
        self.db_path = Path(db_path)
        self.asset_root = Path(asset_root).resolve()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = FULL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    @contextlib.contextmanager
    def transaction(self, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def migrate(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
            )
            applied = {
                row[0] for row in conn.execute("SELECT version FROM schema_migrations")
            }
            for path in sorted(MIGRATIONS.glob("[0-9][0-9][0-9]_*.sql")):
                version = int(path.name.split("_", 1)[0])
                if version in applied:
                    continue
                name = path.name.replace("'", "''")
                applied_at = utc_now().replace("'", "''")
                script = (
                    "BEGIN EXCLUSIVE;\n" + path.read_text() +
                    f"\nINSERT OR REPLACE INTO schema_migrations VALUES"
                    f"({version},'{name}','{applied_at}');\n"
                    f"PRAGMA user_version = {version};\nCOMMIT;\n"
                )
                conn.executescript(script)

    def add_post(self, post: Mapping[str, Any]) -> None:
        post_id = str(post["post_id"])
        if "mature" in post_id.lower() or post_id.lower().startswith("mne_"):
            raise ValueError("mature identifiers are forbidden in platform manifest")
        fields = {
            "post_id": post_id,
            "platform": post["platform"],
            "audience_tiers": json.dumps(post.get("audience_tiers", [])),
            "format": post["format"],
            "story_week": post.get("story_week"),
            "publish_at_local": post.get("publish_at_local"),
            "publish_at_utc": post.get("publish_at_utc"),
            "timezone": post.get("timezone", "America/Toronto"),
            "title": post.get("title", ""),
            "body": post.get("body", ""),
            "cta": post.get("cta", ""),
            "alt_text": post.get("alt_text", ""),
            "hashtags": json.dumps(post.get("hashtags", [])),
            "location_label": post.get("location_label", ""),
            "disclosure": post["disclosure"],
            "policy_version": post["policy_version"],
            "calendar_version": post.get("calendar_version", 1),
        }
        fields["content_hash"] = stable_hash(
            {k: fields[k] for k in (
                "title", "body", "cta", "alt_text", "hashtags",
                "location_label", "disclosure",
            )}
        )
        fields["created_at"] = fields["updated_at"] = utc_now()
        columns = ",".join(fields)
        placeholders = ",".join("?" for _ in fields)
        with self.transaction() as conn:
            conn.execute(
                f"INSERT INTO posts({columns}) VALUES({placeholders})",
                tuple(fields.values()),
            )
            self._audit(conn, "post", post_id, "created", "system", {})

    def update_content(self, post_id: str, **changes: Any) -> str:
        allowed = {
            "title", "body", "cta", "alt_text", "hashtags",
            "location_label", "disclosure",
        }
        if not changes or set(changes) - allowed:
            raise ValueError("unsupported or empty content update")
        if "hashtags" in changes and not isinstance(changes["hashtags"], str):
            changes["hashtags"] = json.dumps(changes["hashtags"])
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM posts WHERE post_id=?", (post_id,)).fetchone()
            if row is None:
                raise KeyError(post_id)
            snapshot = dict(row)
            snapshot.update(changes)
            content_hash = stable_hash(
                {k: snapshot[k] for k in (
                    "title", "body", "cta", "alt_text", "hashtags",
                    "location_label", "disclosure",
                )}
            )
            changes.update(content_hash=content_hash, updated_at=utc_now())
            assignments = ",".join(f"{key}=?" for key in changes)
            conn.execute(
                f"UPDATE posts SET {assignments} WHERE post_id=?",
                (*changes.values(), post_id),
            )
            return content_hash

    def transition(self, post_id: str, to_state: str, actor: str = "system") -> None:
        with self.transaction() as conn:
            old = conn.execute(
                "SELECT state FROM posts WHERE post_id=?", (post_id,)
            ).fetchone()
            if old is None:
                raise KeyError(post_id)
            conn.execute(
                "UPDATE posts SET state=?, updated_at=? WHERE post_id=?",
                (to_state, utc_now(), post_id),
            )
            self._audit(
                conn, "post", post_id, "state_transition", actor,
                {"from": old["state"], "to": to_state},
            )

    def approve(
        self, post_id: str, approval_type: str, approver: str,
        decision: str = "approved", reason: str | None = None,
    ) -> int:
        with self.transaction() as conn:
            post = conn.execute(
                "SELECT content_hash FROM posts WHERE post_id=?", (post_id,)
            ).fetchone()
            if post is None:
                raise KeyError(post_id)
            snapshot_hash = self._approval_snapshot_hash(conn, post_id)
            cursor = conn.execute(
                "INSERT INTO approvals(post_id,approval_type,decision,approver,"
                "reason,snapshot_hash,created_at) VALUES(?,?,?,?,?,?,?)",
                (post_id, approval_type, decision, approver, reason,
                 snapshot_hash, utc_now()),
            )
            return int(cursor.lastrowid)

    def approval_snapshot_hash(self, post_id: str) -> str:
        with self.connect() as conn:
            if conn.execute(
                "SELECT 1 FROM posts WHERE post_id=?", (post_id,)
            ).fetchone() is None:
                raise KeyError(post_id)
            return self._approval_snapshot_hash(conn, post_id)

    @staticmethod
    def _approval_snapshot_hash(conn: sqlite3.Connection, post_id: str) -> str:
        post = conn.execute(
            "SELECT content_hash,publish_at_local,publish_at_utc,timezone,"
            "calendar_version FROM posts WHERE post_id=?",
            (post_id,),
        ).fetchone()
        assets = conn.execute(
            "SELECT pa.asset_slot,pa.ordinal,a.asset_id,a.sha256 "
            "FROM post_assets pa JOIN assets a ON a.asset_id=pa.asset_id "
            "WHERE pa.post_id=? AND pa.selected=1 "
            "ORDER BY pa.asset_slot,pa.ordinal,a.asset_id",
            (post_id,),
        ).fetchall()
        return stable_hash(
            {
                "content_hash": post["content_hash"],
                "schedule": {
                    "publish_at_local": post["publish_at_local"],
                    "publish_at_utc": post["publish_at_utc"],
                    "timezone": post["timezone"],
                    "calendar_version": post["calendar_version"],
                },
                "assets": [dict(row) for row in assets],
            }
        )

    def register_asset(
        self, asset_id: str, path: os.PathLike[str] | str, *,
        media_type: str, prompt_hash: str, workflow_hash: str,
        model_hash: str, policy_version: str,
    ) -> str:
        candidate = Path(path)
        if candidate.is_symlink():
            raise ValueError("symlink assets are forbidden")
        resolved = candidate.resolve(strict=True)
        try:
            relative = resolved.relative_to(self.asset_root)
        except ValueError as exc:
            raise ValueError("asset escapes platform root") from exc
        stat = resolved.stat()
        if not resolved.is_file() or stat.st_nlink != 1:
            raise ValueError("asset must be a regular file with one hard link")
        relative_l = str(relative).lower()
        if (
            "mature" in asset_id.lower()
            or "nsfw" in asset_id.lower()
            or asset_id.lower().startswith("mne_")
            or "mature_non_explicit" in relative_l
            or "nsfw_test" in relative_l
        ):
            raise ValueError("mature assets are forbidden in platform manifest")
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO assets(asset_id,sha256,relative_path,media_type,"
                "prompt_hash,workflow_hash,model_hash,policy_version,created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (asset_id, digest, str(relative), media_type, prompt_hash,
                 workflow_hash, model_hash, policy_version, utc_now()),
            )
        return digest

    def attach_asset(
        self, post_id: str, asset_id: str, asset_slot: str,
        ordinal: int = 0, selected: bool = False,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO post_assets VALUES(?,?,?,?,?)",
                (post_id, asset_id, asset_slot, ordinal, int(selected)),
            )

    def enqueue_job(
        self, post_id: str, asset_slot: str, generation_version: int,
        candidate_index: int, track: str, hashes: Mapping[str, str],
    ) -> int:
        if track not in PLATFORM_TRACKS:
            raise ValueError("non-platform track rejected")
        now = utc_now()
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO generation_jobs(post_id,asset_slot,generation_version,"
                "candidate_index,track,prompt_hash,workflow_hash,model_hash,"
                "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (post_id, asset_slot, generation_version, candidate_index, track,
                 hashes["prompt"], hashes["workflow"], hashes["model"], now, now),
            )
            return int(cursor.lastrowid)

    def lease_job(self, owner: str, lease_seconds: int = 900) -> sqlite3.Row | None:
        now = utc_now()
        expires = (
            datetime.now(UTC) + timedelta(seconds=lease_seconds)
        ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM generation_jobs WHERE status='queued' "
                "OR (status='leased' AND lease_expires_at < ?) "
                "ORDER BY job_id LIMIT 1", (now,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE generation_jobs SET status='leased',lease_owner=?,"
                "lease_expires_at=?,updated_at=? WHERE job_id=?",
                (owner, expires, now, row["job_id"]),
            )
            return conn.execute(
                "SELECT * FROM generation_jobs WHERE job_id=?", (row["job_id"],)
            ).fetchone()

    def record_benchmark(
        self, component: str, workload: str, durations: Sequence[float],
        failures: int = 0, peak_memory_mb: float | None = None,
        model_id: str | None = None, metadata: Mapping[str, Any] | None = None,
    ) -> int:
        if not durations:
            raise ValueError("at least one duration is required")
        ordered = sorted(durations)
        percentile = lambda p: ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p))]
        total = len(durations) + failures
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO throughput_benchmarks(component,model_id,workload,"
                "samples,p50_seconds,p95_seconds,failure_rate,peak_memory_mb,"
                "metadata_json,recorded_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (component, model_id, workload, total, percentile(.50),
                 percentile(.95), failures / total, peak_memory_mb,
                 json.dumps(metadata or {}, sort_keys=True), utc_now()),
            )
            return int(cursor.lastrowid)

    def backup(self, destination: os.PathLike[str] | str) -> None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as source, sqlite3.connect(destination) as target:
            source.backup(target)

    @staticmethod
    def restore(source: os.PathLike[str] | str, destination: os.PathLike[str] | str) -> None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".restore")
        shutil.copy2(source, temporary)
        with sqlite3.connect(temporary) as conn:
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                temporary.unlink(missing_ok=True)
                raise ValueError("backup failed integrity check")
        os.replace(temporary, destination)

    @staticmethod
    def _audit(
        conn: sqlite3.Connection, entity_type: str, entity_id: str,
        event_type: str, actor: str, payload: Mapping[str, Any],
    ) -> None:
        conn.execute(
            "INSERT INTO audit_events(entity_type,entity_id,event_type,actor,"
            "payload_json,created_at) VALUES(?,?,?,?,?,?)",
            (entity_type, entity_id, event_type, actor,
             json.dumps(payload, sort_keys=True), utc_now()),
        )
