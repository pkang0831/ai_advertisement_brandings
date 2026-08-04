#!/usr/bin/env python3
"""Validate Rina Park launch copy without posting or modifying source files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_POLICY = BASE_DIR / "launch_policy.json"
DEFAULT_DECISION = BASE_DIR / "disclosure_decision.json"
DEFAULT_CALENDAR = BASE_DIR.parents[1] / "content" / "calendar_8_weeks.csv"

PROHIBITED_CLAIMS = {
    "real-time location claim": re.compile(
        r"\b(?:I(?:'m| am) currently (?:at|in)|"
        r"I(?:'m| am) (?:at|in) .{1,60}\b(?:right now|today)|"
        r"posting live from)\b",
        re.IGNORECASE,
    ),
    "real bodily achievement claim": re.compile(
        r"\bI (?:swam|completed|finished|trained for|ran) "
        r"(?:\d[\d,.]*|my first|the (?:swim|workout|race)|"
        r"(?:this|today's) (?:swim|workout|training))\b",
        re.IGNORECASE,
    ),
    "real weight or body result claim": re.compile(
        r"\bI (?:lost|gained) \d[\d,.]*\s*(?:pounds?|lbs?|kg|kilograms?)\b",
        re.IGNORECASE,
    ),
    "human identity claim": re.compile(
        r"\bI(?:'m| am) (?:a )?real (?:person|woman|human)\b", re.IGNORECASE
    ),
    "firsthand product claim": re.compile(
        r"\bI (?:personally )?(?:use|used|recommend|tested|bought) "
        r"(?:this|these|the|my)\b",
        re.IGNORECASE,
    ),
    "meeting or live-location invitation": re.compile(
        r"\b(?:meet me|come find me) at\b", re.IGNORECASE
    ),
}


def load_policy(path: Path = DEFAULT_POLICY) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_decision(path: Path = DEFAULT_DECISION) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def get_section(text: str, heading: str) -> str | None:
    """Return content under an exact Markdown heading, including subheadings."""
    lines = text.splitlines()
    target_index = None
    target_level = None

    for index, line in enumerate(lines):
        match = re.match(r"^(#{2,6})\s+(.+?)\s*$", line)
        if match and match.group(2) == heading:
            target_index = index
            target_level = len(match.group(1))
            break

    if target_index is None or target_level is None:
        return None

    end = len(lines)
    for index in range(target_index + 1, len(lines)):
        match = re.match(r"^(#{2,6})\s+", lines[index])
        if match and len(match.group(1)) <= target_level:
            end = index
            break
    return "\n".join(lines[target_index + 1 : end]).strip()


def validate_disclosure_unit(
    label: str, text: str | None, disclosure: str
) -> list[str]:
    if text is None:
        return [f"Missing section: {label}"]
    if disclosure not in text:
        return [f"Missing mandatory disclosure in: {label}"]
    return []


def validate_prohibited_claims(
    named_texts: Iterable[tuple[str, str]]
) -> list[str]:
    errors: list[str] = []
    for source_name, text in named_texts:
        for claim_name, pattern in PROHIBITED_CLAIMS.items():
            match = pattern.search(text)
            if match:
                errors.append(
                    f"{source_name}: prohibited {claim_name}: {match.group(0)!r}"
                )
    return errors


def validate_disclosure_decision(
    policy: dict,
    decision: dict,
    candidates_text: str,
    release_gate: bool = False,
) -> list[str]:
    """Validate decision state without selecting or applying any candidate."""
    errors: list[str] = []
    expected_ids = policy["disclosure_candidate_ids"]
    candidate_ids = decision.get("candidate_ids")

    if candidate_ids != expected_ids:
        errors.append("Disclosure candidate IDs do not match launch policy.")
        candidate_ids = candidate_ids if isinstance(candidate_ids, list) else []

    if decision.get("auto_apply") is not False:
        errors.append("Disclosure decision must keep auto_apply false.")

    selected = decision.get("selected")
    approved = decision.get("approved_by_user")
    if selected is not None and selected not in candidate_ids:
        errors.append("Selected disclosure ID is not an allowed candidate.")
    if not isinstance(approved, bool):
        errors.append("approved_by_user must be a boolean.")

    for candidate_id in expected_ids:
        section = get_section(candidates_text, f"`{candidate_id}`")
        if section is None:
            errors.append(f"Missing disclosure candidate: {candidate_id}")
            continue
        profile_match = re.search(
            r"^\*\*Profile line:\*\*\s+(.+)$", section, re.MULTILINE
        )
        post_match = re.search(
            r"^\*\*Post line:\*\*\s+(.+)$", section, re.MULTILINE
        )
        if not profile_match or not post_match:
            errors.append(
                f"{candidate_id}: profile and post lines are both required."
            )
            continue
        for line_type, copy in (
            ("profile", profile_match.group(1)),
            ("post", post_match.group(1)),
        ):
            if not (
                re.search(r"\bfictional\b", copy, re.IGNORECASE)
                and re.search(r"\bAI-generated\b", copy, re.IGNORECASE)
            ):
                errors.append(
                    f"{candidate_id}: {line_type} line must explicitly say "
                    "fictional and AI-generated."
                )

    resolved = selected in candidate_ids and approved is True
    if resolved:
        if decision.get("status") != "user_approved":
            errors.append("Resolved disclosure status must be user_approved.")
        if decision.get("release_gate") != "ready":
            errors.append("Resolved disclosure release_gate must be ready.")
    else:
        if decision.get("status") != "user_decision_required":
            errors.append(
                "Unresolved disclosure status must be user_decision_required."
            )
        if decision.get("release_gate") != "blocked":
            errors.append("Unresolved disclosure release_gate must be blocked.")
        if release_gate:
            errors.append(
                "USER DECISION REQUIRED: choose one disclosure candidate ID "
                "and explicitly approve it."
            )

    return errors


def proof_section(text: str, post_id: str) -> str | None:
    match = re.search(
        rf"^### `{re.escape(post_id)}`[^\n]*\n(?P<body>.*?)(?=^### `|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return match.group("body").strip() if match else None


def validate_calendar(policy: dict, calendar_path: Path) -> list[str]:
    errors: list[str] = []
    with calendar_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    expected = list(policy["proof_posts"])
    actual = [row["post_id"] for row in rows[:14]]
    if actual != expected:
        errors.append(
            "Two-week post IDs do not match the first 14 source-calendar rows."
        )

    by_id = {row["post_id"]: row for row in rows}
    for post_id, expected_location in policy["proof_posts"].items():
        row = by_id.get(post_id)
        if row is None:
            errors.append(f"Post ID is absent from source calendar: {post_id}")
            continue
        calendar_location = row["location_label"] or (
            "None — no location label in the source calendar."
        )
        if calendar_location != expected_location:
            errors.append(
                f"{post_id}: policy location {expected_location!r} does not "
                f"match calendar location {calendar_location!r}."
            )
    return errors


def validate_package(
    base_dir: Path = BASE_DIR,
    policy_path: Path = DEFAULT_POLICY,
    calendar_path: Path | None = DEFAULT_CALENDAR,
    decision_path: Path = DEFAULT_DECISION,
    release_gate: bool = False,
) -> list[str]:
    policy = load_policy(policy_path)
    disclosure = policy["draft_placeholder_disclosure"]
    errors: list[str] = []

    instagram_path = base_dir / "instagram.md"
    patreon_path = base_dir / "patreon.md"
    proof_path = base_dir / "two_week_copy_proof.md"
    candidates_path = base_dir / "disclosure_candidates.md"

    required_files = (
        instagram_path,
        patreon_path,
        proof_path,
        candidates_path,
        decision_path,
    )
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        return [f"Missing required file: {path.name}" for path in missing]

    instagram = instagram_path.read_text(encoding="utf-8")
    patreon = patreon_path.read_text(encoding="utf-8")
    proof = proof_path.read_text(encoding="utf-8")
    candidates = candidates_path.read_text(encoding="utf-8")
    decision = load_decision(decision_path)

    errors.extend(
        validate_disclosure_decision(
            policy, decision, candidates, release_gate=release_gate
        )
    )

    for heading in policy["bio_headings"]:
        section = get_section(instagram, heading)
        if section is None:
            errors.append(f"Missing section: {heading}")
        elif not (
            re.search(r"\bfictional\b", section, re.IGNORECASE)
            and re.search(r"\bAI-generated\b", section, re.IGNORECASE)
        ):
            errors.append(
                f"{heading}: bio must explicitly say fictional and AI-generated."
            )

    for heading in policy["pinned_post_headings"]:
        errors.extend(
            validate_disclosure_unit(
                heading, get_section(instagram, heading), disclosure
            )
        )

    for heading in policy["patreon_disclosure_headings"]:
        errors.extend(
            validate_disclosure_unit(
                heading, get_section(patreon, heading), disclosure
            )
        )

    for tier_name, price in policy["tiers"].items():
        heading = f"{tier_name} — {price}"
        if get_section(patreon, heading) is None:
            errors.append(f"Missing tier name or draft price: {heading}")

    for post_id, location in policy["proof_posts"].items():
        section = proof_section(proof, post_id)
        errors.extend(validate_disclosure_unit(post_id, section, disclosure))
        if section is None:
            continue
        if not re.search(r"^\*\*Alt text:\*\*\s+\S", section, re.MULTILINE):
            errors.append(f"{post_id}: missing non-empty alt text.")
        location_match = re.search(
            r"^\*\*Location label:\*\*\s+(.+?)\s*$", section, re.MULTILINE
        )
        if not location_match:
            errors.append(f"{post_id}: missing location label.")
        elif location_match.group(1) != location:
            errors.append(
                f"{post_id}: expected location {location!r}, found "
                f"{location_match.group(1)!r}."
            )
        hashtags = re.findall(r"(?<!\w)#[A-Za-z0-9_]+", section)
        if len(hashtags) > 3:
            errors.append(f"{post_id}: more than three hashtags in proof copy.")

    expected_ids = set(policy["proof_posts"])
    found_ids = set(re.findall(r"^### `([^`]+)`", proof, re.MULTILINE))
    extras = sorted(found_ids - expected_ids)
    if extras:
        errors.append(f"Unexpected post IDs in proof: {', '.join(extras)}")

    errors.extend(
        validate_prohibited_claims(
            (
                (instagram_path.name, instagram),
                (patreon_path.name, patreon),
                (proof_path.name, proof),
                (candidates_path.name, candidates),
            )
        )
    )

    if calendar_path is not None:
        if not calendar_path.is_file():
            errors.append(f"Source calendar not found: {calendar_path}")
        else:
            errors.extend(validate_calendar(policy, calendar_path))

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Rina Park launch copy disclosures and claims."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Launch-pack directory to validate.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
        help="Launch policy JSON path.",
    )
    parser.add_argument(
        "--calendar",
        type=Path,
        default=DEFAULT_CALENDAR,
        help="Read-only source calendar CSV path.",
    )
    parser.add_argument(
        "--decision",
        type=Path,
        default=DEFAULT_DECISION,
        help="Disclosure decision manifest path.",
    )
    parser.add_argument(
        "--release-gate",
        action="store_true",
        help="Fail unless the user selected and approved one disclosure candidate.",
    )
    args = parser.parse_args()

    errors = validate_package(
        base_dir=args.base_dir,
        policy_path=args.policy,
        calendar_path=args.calendar,
        decision_path=args.decision,
        release_gate=args.release_gate,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    gate_name = "release" if args.release_gate else "draft-package"
    print(f"Launch copy {gate_name} validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
