"""Explicit seed import and transactional calendar scheduling.

The CSV is read only by ``import_seed``. Once imported, callers must read
schedule data from manifest.db.
"""

from __future__ import annotations

import csv
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from rina_park.factory.manifest import Manifest, stable_hash, utc_now

TORONTO = ZoneInfo("America/Toronto")
SEED_START = date(2026, 8, 10)
DISCLOSURE = "Rina is a fictional virtual character. Visuals are AI-generated."


def _list_field(value: str) -> list[str]:
    stripped = value.strip()
    if stripped == "[]":
        return []
    if stripped.startswith("[") and stripped.endswith("]"):
        return [part.strip() for part in stripped[1:-1].split(",") if part.strip()]
    return stripped.split()


def _row_to_post(row: dict[str, str]) -> dict[str, object]:
    return {
        "post_id": row["post_id"],
        "platform": row["platform"],
        "audience_tiers": (
            ["A", "B", "C"] if row["audience_tiers"] == "[public]"
            else _list_field(row["audience_tiers"])
        ),
        "format": row["format"],
        "story_week": int(row["story_week"]),
        "publish_at_local": row["publish_at_local"],
        "publish_at_utc": row["publish_at_utc"],
        "timezone": "America/Toronto",
        "title": row["title"],
        "body": row["body"],
        "cta": row["cta"],
        "alt_text": row["alt_text"],
        "hashtags": row["hashtags"].split(),
        "location_label": row["location_label"],
        "disclosure": row["disclosure"],
        "policy_version": "platform-sfw-v1",
        "calendar_version": 1,
    }


def import_seed(manifest: Manifest, csv_path: Path) -> int:
    """Import the immutable seed exactly once into an empty runtime manifest."""
    manifest.migrate()
    raw = csv_path.read_bytes()
    seed_sha256 = stable_hash({"bytes_hex": raw.hex()})
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 56:
        raise ValueError("calendar seed must contain exactly 56 rows")
    posts = [_row_to_post(row) for row in rows]
    if any(post["disclosure"] != DISCLOSURE for post in posts):
        raise ValueError("calendar seed disclosure mismatch")

    with manifest.transaction() as conn:
        existing = conn.execute("SELECT count(*) FROM posts").fetchone()[0]
        setting = conn.execute(
            "SELECT value_json FROM runtime_settings WHERE setting_key='calendar_seed'"
        ).fetchone()
        if existing:
            if setting and json.loads(setting["value_json"]).get("sha256") == seed_sha256:
                return 0
            raise RuntimeError(
                "manifest already contains runtime posts; CSV re-import is forbidden"
            )
        now = utc_now()
        for post in posts:
            fields = {
                **post,
                "audience_tiers": json.dumps(post["audience_tiers"]),
                "hashtags": json.dumps(post["hashtags"]),
            }
            fields["content_hash"] = stable_hash(
                {
                    key: fields[key]
                    for key in (
                        "title", "body", "cta", "alt_text", "hashtags",
                        "location_label", "disclosure",
                    )
                }
            )
            fields["created_at"] = fields["updated_at"] = now
            columns = ",".join(fields)
            placeholders = ",".join("?" for _ in fields)
            conn.execute(
                f"INSERT INTO posts({columns}) VALUES({placeholders})",
                tuple(fields.values()),
            )
        conn.execute(
            "INSERT INTO runtime_settings(setting_key,value_json,updated_at) VALUES(?,?,?)",
            (
                "calendar_seed",
                json.dumps(
                    {"sha256": seed_sha256, "rows": len(posts), "imported_at": now},
                    sort_keys=True,
                ),
                now,
            ),
        )
    return len(posts)


def reschedule_calendar(manifest: Manifest, new_start: date) -> int:
    """Shift every post atomically and increment one shared calendar version."""
    delta = new_start - SEED_START
    with manifest.transaction() as conn:
        current = conn.execute(
            "SELECT COALESCE(MAX(calendar_version), 0) FROM posts"
        ).fetchone()[0]
        if not current:
            raise RuntimeError("calendar seed has not been imported")
        version = int(current) + 1
        rows = conn.execute(
            "SELECT post_id,publish_at_local FROM posts ORDER BY publish_at_utc"
        ).fetchall()
        now = utc_now()
        for row in rows:
            old_local = datetime.fromisoformat(row["publish_at_local"])
            wall = (old_local.astimezone(TORONTO) + delta).replace(tzinfo=None)
            shifted_local = wall.replace(tzinfo=TORONTO)
            shifted_utc = shifted_local.astimezone(timezone.utc)
            conn.execute(
                "UPDATE posts SET publish_at_local=?,publish_at_utc=?,timezone=?,"
                "calendar_version=?,updated_at=? WHERE post_id=?",
                (
                    shifted_local.isoformat(),
                    shifted_utc.isoformat().replace("+00:00", "Z"),
                    "America/Toronto",
                    version,
                    now,
                    row["post_id"],
                ),
            )
        conn.execute(
            "INSERT INTO runtime_settings(setting_key,value_json,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(setting_key) DO UPDATE SET "
            "value_json=excluded.value_json,updated_at=excluded.updated_at",
            (
                "calendar_schedule",
                json.dumps(
                    {"calendar_version": version, "start_date": new_start.isoformat()},
                    sort_keys=True,
                ),
                now,
            ),
        )
    return version


def due_posts(
    manifest: Manifest,
    now: datetime,
    *,
    max_lateness: timedelta = timedelta(hours=6),
) -> list[dict[str, object]]:
    """Return approved due work using UTC only; stale work requires HITL."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now_utc = now.astimezone(timezone.utc)
    cutoff = now_utc - max_lateness
    with manifest.transaction() as conn:
        stale = conn.execute(
            "SELECT post_id FROM posts WHERE publish_at_utc < ? "
            "AND state IN ('schedule_approved','packaged')",
            (cutoff.isoformat().replace("+00:00", "Z"),),
        ).fetchall()
        for row in stale:
            conn.execute(
                "UPDATE posts SET state='needs_review',updated_at=? WHERE post_id=?",
                (utc_now(), row["post_id"]),
            )
        rows = conn.execute(
            "SELECT * FROM posts WHERE publish_at_utc BETWEEN ? AND ? "
            "AND state IN ('schedule_approved','packaged') ORDER BY publish_at_utc",
            (
                cutoff.isoformat().replace("+00:00", "Z"),
                now_utc.isoformat().replace("+00:00", "Z"),
            ),
        ).fetchall()
    return [dict(row) for row in rows]
