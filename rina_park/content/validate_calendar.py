#!/usr/bin/env python3
"""Read-only validation for the eight-week Rina Park seed calendar."""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
CALENDAR = ROOT / "calendar_8_weeks.csv"
HASHTAG_CONFIG = ROOT / "hashtag_sets.yml"
TORONTO = ZoneInfo("America/Toronto")
START_DATE = datetime(2026, 8, 10, tzinfo=TORONTO).date()
DISCLOSURE = "Rina is a fictional virtual character. Visuals are AI-generated."

FIELDS = [
    "post_id",
    "platform",
    "audience_tiers",
    "format",
    "story_week",
    "publish_at_local",
    "publish_at_utc",
    "title",
    "body",
    "cta",
    "alt_text",
    "hashtags",
    "location_label",
    "asset_ids",
    "disclosure",
    "content_approval",
    "schedule_approval",
    "status",
    "published_url",
]

IG_CADENCE = {
    0: (time(19, 30), "carousel"),
    2: (time(12, 15), "reel"),
    4: (time(20, 30), "carousel"),
    6: (time(10, 30), "reel"),
}
PATREON_CADENCE = {
    1: (time(21, 0), "[A,B,C]", "gallery_4_short_note"),
    3: (time(21, 0), "[B,C]", "gallery_6_alternate_editorial"),
    5: (time(22, 0), "[C]", "gallery_8_notes_archive"),
}
LOCATIONS = {
    "Toronto, Ontario",
    "Downtown Toronto",
    "Toronto Waterfront",
    "Toronto Parks",
    "Greater Toronto Area",
}
PLATFORM_COPY_BLOCKLIST = re.compile(
    r"\b(lingerie|nude|nudity|nipples?|genitals?|sexual acts?|"
    r"sexualized|seductive|sultry|explicit content|sexual services?)\b",
    re.IGNORECASE,
)


def fail(errors: list[str], row_number: int | None, message: str) -> None:
    prefix = f"row {row_number}: " if row_number is not None else ""
    errors.append(prefix + message)


def parse_hashtag_sets(path: Path) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    current: str | None = None
    in_sets = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line == "sets:":
            in_sets = True
            continue
        if raw_line == "rotation_rules:":
            break
        if not in_sets:
            continue
        key = re.fullmatch(r"  ([a-z_]+):", raw_line)
        if key:
            current = key.group(1)
            sets[current] = set()
            continue
        item = re.fullmatch(r'    - "(#[A-Za-z0-9]+)"', raw_line)
        if item and current:
            sets[current].add(item.group(1))
    return sets


def parse_local(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("local timestamp has no offset")
    return parsed


def parse_utc(value: str) -> datetime:
    if not value.endswith("Z"):
        raise ValueError("UTC timestamp must end in Z")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate() -> list[str]:
    errors: list[str] = []
    hashtag_sets = parse_hashtag_sets(HASHTAG_CONFIG)
    expected_sets = {"swim", "niche", "local", "virtual_creator"}
    if set(hashtag_sets) != expected_sets:
        fail(errors, None, f"hashtag sets must be exactly {sorted(expected_sets)}")
    known_tags = set().union(*hashtag_sets.values()) if hashtag_sets else set()

    with CALENDAR.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            fail(errors, None, f"CSV fields differ from required order: {reader.fieldnames}")
        rows = list(reader)

    if len(rows) != 56:
        fail(errors, None, f"expected 56 rows, found {len(rows)}")

    platform_counts = Counter(row["platform"] for row in rows)
    if platform_counts != Counter({"instagram": 32, "patreon": 24}):
        fail(errors, None, f"platform counts are {dict(platform_counts)}")

    seen_ids: set[str] = set()
    seen_local: set[str] = set()
    seen_utc: set[str] = set()
    unique_copy = {field: set() for field in ("title", "body", "cta", "alt_text")}
    week_platform = Counter()
    ig_tag_sets: set[frozenset[str]] = set()
    ig_tag_frequency: Counter[str] = Counter()

    for index, row in enumerate(rows, start=2):
        post_id = row["post_id"]
        if post_id in seen_ids:
            fail(errors, index, f"duplicate post_id {post_id}")
        seen_ids.add(post_id)

        for timestamp_field, seen in (
            ("publish_at_local", seen_local),
            ("publish_at_utc", seen_utc),
        ):
            value = row[timestamp_field]
            if value in seen:
                fail(errors, index, f"duplicate {timestamp_field} {value}")
            seen.add(value)

        try:
            local = parse_local(row["publish_at_local"])
            utc = parse_utc(row["publish_at_utc"])
        except ValueError as exc:
            fail(errors, index, str(exc))
            continue

        expected_local = local.astimezone(TORONTO)
        if local.utcoffset() != expected_local.utcoffset():
            fail(errors, index, "local offset does not match America/Toronto")
        if local.astimezone(timezone.utc) != utc:
            fail(errors, index, "local and UTC timestamps do not represent the same instant")

        expected_week = ((expected_local.date() - START_DATE).days // 7) + 1
        try:
            story_week = int(row["story_week"])
        except ValueError:
            fail(errors, index, f"invalid story_week {row['story_week']!r}")
            continue
        if story_week != expected_week or story_week not in range(1, 9):
            fail(
                errors,
                index,
                f"story_week {story_week} does not match date-derived week {expected_week}",
            )
        week_platform[(story_week, row["platform"])] += 1

        if row["platform"] == "instagram":
            cadence = IG_CADENCE.get(expected_local.weekday())
            if cadence is None:
                fail(errors, index, "Instagram post is on a non-cadence weekday")
            elif (expected_local.time(), row["format"]) != cadence:
                fail(errors, index, "Instagram time or format violates cadence")
            if row["audience_tiers"] != "[public]":
                fail(errors, index, "Instagram audience_tiers must be [public]")
            if row["location_label"] not in LOCATIONS:
                fail(errors, index, "Instagram location is not an approved broad label")

            tags = row["hashtags"].split()
            if len(tags) not in range(5, 9) or len(tags) != len(set(tags)):
                fail(errors, index, "Instagram must have 5–8 unique hashtags")
            if "#VirtualInfluencer" not in tags:
                fail(errors, index, "Instagram is missing #VirtualInfluencer")
            unknown = set(tags) - known_tags
            if unknown:
                fail(errors, index, f"unknown hashtags: {sorted(unknown)}")
            for set_name, allowed in hashtag_sets.items():
                if not set(tags) & allowed:
                    fail(errors, index, f"no hashtag from {set_name} set")
            frozen_tags = frozenset(tags)
            if frozen_tags in ig_tag_sets:
                fail(errors, index, "duplicate Instagram hashtag set")
            ig_tag_sets.add(frozen_tags)
            ig_tag_frequency.update(tag for tag in tags if tag != "#VirtualInfluencer")

        elif row["platform"] == "patreon":
            cadence = PATREON_CADENCE.get(expected_local.weekday())
            if cadence is None:
                fail(errors, index, "Patreon post is on a non-cadence weekday")
            elif (
                expected_local.time(),
                row["audience_tiers"],
                row["format"],
            ) != cadence:
                fail(errors, index, "Patreon time, audience_tiers, or format violates cadence")
            if row["hashtags"] or row["location_label"]:
                fail(errors, index, "Patreon hashtags and location must be empty")
        else:
            fail(errors, index, f"unknown platform {row['platform']!r}")

        for field in ("title", "body", "cta", "alt_text"):
            value = row[field]
            if not value or not re.search(r"[A-Za-z]", value):
                fail(errors, index, f"{field} must contain English text")
            if not value.isascii():
                fail(errors, index, f"{field} must use English/ASCII copy")
            if value in unique_copy[field]:
                fail(errors, index, f"{field} must be unique")
            unique_copy[field].add(value)
            if PLATFORM_COPY_BLOCKLIST.search(value):
                fail(errors, index, f"{field} contains blocked platform-SFW language")

        if row["disclosure"] != DISCLOSURE:
            fail(errors, index, "AI disclosure is missing or altered")
        if row["asset_ids"] != "[]":
            fail(errors, index, "seed asset_ids must be []")
        if (
            row["content_approval"],
            row["schedule_approval"],
            row["status"],
            row["published_url"],
        ) != ("pending", "pending", "draft", ""):
            fail(errors, index, "seed approval/status/publication fields are invalid")

    for week in range(1, 9):
        if week_platform[(week, "instagram")] != 4:
            fail(errors, None, f"week {week} must contain 4 Instagram posts")
        if week_platform[(week, "patreon")] != 3:
            fail(errors, None, f"week {week} must contain 3 Patreon posts")

    overused = {
        tag: count for tag, count in ig_tag_frequency.items() if count > 12
    }
    if overused:
        fail(errors, None, f"non-required hashtags exceed 12 uses: {overused}")

    return errors


if __name__ == "__main__":
    validation_errors = validate()
    if validation_errors:
        print(f"FAILED: {len(validation_errors)} validation error(s)")
        for validation_error in validation_errors:
            print(f"- {validation_error}")
        sys.exit(1)
    print(
        "PASS: 56 rows; 32 Instagram; 24 Patreon; "
        "8 weeks; cadence, UTC, copy, SFW, tiers, and hashtag rotation valid."
    )
