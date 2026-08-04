"""Idempotent manual weekly CSV import into metrics.db."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

FIELDS = (
    "week_start",
    "platform",
    "post_id",
    "impressions",
    "reach",
    "likes",
    "comments",
    "saves",
    "shares",
    "clicks",
    "paid_members",
    "revenue_usd",
)

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS metric_imports(
  import_sha256 TEXT PRIMARY KEY,
  source_name TEXT NOT NULL,
  row_count INTEGER NOT NULL,
  imported_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS weekly_metrics(
  week_start TEXT NOT NULL,
  platform TEXT NOT NULL CHECK(platform IN ('instagram','patreon')),
  post_id TEXT NOT NULL,
  impressions INTEGER NOT NULL CHECK(impressions >= 0),
  reach INTEGER NOT NULL CHECK(reach >= 0),
  likes INTEGER NOT NULL CHECK(likes >= 0),
  comments INTEGER NOT NULL CHECK(comments >= 0),
  saves INTEGER NOT NULL CHECK(saves >= 0),
  shares INTEGER NOT NULL CHECK(shares >= 0),
  clicks INTEGER NOT NULL CHECK(clicks >= 0),
  paid_members INTEGER NOT NULL CHECK(paid_members >= 0),
  revenue_usd REAL NOT NULL CHECK(revenue_usd >= 0),
  import_sha256 TEXT NOT NULL REFERENCES metric_imports(import_sha256),
  raw_json TEXT NOT NULL,
  PRIMARY KEY(week_start, platform, post_id)
);
"""


class MetricsStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.executescript(SCHEMA)

    def import_csv(self, path: Path) -> int:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"metrics fields must be exactly {FIELDS}")
            rows = list(reader)
        normalized = []
        for row in rows:
            date.fromisoformat(row["week_start"])
            if row["platform"] not in {"instagram", "patreon"}:
                raise ValueError("unsupported metrics platform")
            if not row["post_id"] or "mature" in row["post_id"].lower():
                raise ValueError("invalid or mature post_id in platform metrics")
            integers = {
                field: int(row[field])
                for field in FIELDS[3:-1]
            }
            if any(value < 0 for value in integers.values()):
                raise ValueError("metrics cannot be negative")
            revenue = float(row["revenue_usd"])
            if revenue < 0:
                raise ValueError("revenue cannot be negative")
            normalized.append((row, integers, revenue))

        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with sqlite3.connect(self.path) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            if conn.execute(
                "SELECT 1 FROM metric_imports WHERE import_sha256=?", (digest,)
            ).fetchone():
                return 0
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO metric_imports VALUES(?,?,?,?)",
                (digest, path.name, len(rows), now),
            )
            for row, integers, revenue in normalized:
                conn.execute(
                    "INSERT INTO weekly_metrics VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(week_start,platform,post_id) DO UPDATE SET "
                    "impressions=excluded.impressions,reach=excluded.reach,"
                    "likes=excluded.likes,comments=excluded.comments,"
                    "saves=excluded.saves,shares=excluded.shares,"
                    "clicks=excluded.clicks,paid_members=excluded.paid_members,"
                    "revenue_usd=excluded.revenue_usd,"
                    "import_sha256=excluded.import_sha256,raw_json=excluded.raw_json",
                    (
                        row["week_start"], row["platform"], row["post_id"],
                        integers["impressions"], integers["reach"], integers["likes"],
                        integers["comments"], integers["saves"], integers["shares"],
                        integers["clicks"], integers["paid_members"], revenue,
                        digest, json.dumps(row, sort_keys=True),
                    ),
                )
        return len(rows)
