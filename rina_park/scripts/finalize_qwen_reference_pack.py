#!/usr/bin/env python3
"""Record explicit human review decisions for a Qwen identity reference pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

APPROVED = {
    "01_frontal_neutral",
    "02_frontal_genuine_teeth_smile",
    "03_three_quarter_soft_smile",
    "04_three_quarter_serious",
    "05_true_left_profile_neutral",
    "06_true_right_profile_neutral",
    "07_looking_down_thoughtful",
    "09_over_shoulder_alert",
    "10_candid_mid_laugh",
}
REJECTED = {"08_looking_up_away_curious"}
REJECTION_REASON = "exaggerated upward eyes and open mouth"


def finalize(pack: Path) -> None:
    sidecars = sorted((pack / "candidates").glob("*.png.json"))
    slugs = {path.name.removesuffix(".png.json") for path in sidecars}
    if slugs != APPROVED | REJECTED:
        raise RuntimeError(f"unexpected reference labels: {sorted(slugs)}")

    payloads: dict[str, dict[str, object]] = {}
    for sidecar in sidecars:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        slug = str(payload["slug"])
        if slug in APPROVED:
            payload["review_status"] = "approved_internal_production_reference"
            payload["production_status"] = "approved_internal_reference_not_master"
            payload["human_review"] = {
                "approved": True,
                "scope": "internal production reference",
                "master_replaced": False,
            }
        else:
            payload["review_status"] = "rejected_reference"
            payload["production_status"] = "rejected_not_promoted"
            payload["human_review"] = {
                "approved": False,
                "reason": REJECTION_REASON,
                "master_replaced": False,
            }
        sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payloads[slug] = payload

    summary_path = pack / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["review_status"] = "human_review_complete"
    summary["approved_internal_reference_count"] = len(APPROVED)
    summary["approved_internal_reference_labels"] = sorted(APPROVED)
    summary["rejected_reference_count"] = len(REJECTED)
    summary["rejected_reference_labels"] = sorted(REJECTED)
    summary["rejection_reasons"] = {slug: REJECTION_REASON for slug in sorted(REJECTED)}
    summary["production_references_promoted"] = True
    summary["promotion_scope"] = "internal production references only"
    summary["master_replaced"] = False
    summary["results"] = [payloads[str(item["slug"])] for item in summary["results"]]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.pack.resolve())


if __name__ == "__main__":
    main()
