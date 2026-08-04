"""Load and enforce the anatomy-lock pose whitelist / banlist."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

RINA = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = RINA / "ops" / "anatomy_lock" / "pose_catalog.yml"
DEFAULT_NSFW_CATALOG = RINA / "private" / "pose_catalog_nsfw.yml"


@dataclass(frozen=True)
class PoseEntry:
    id: str
    track: str
    regions: list[str]
    composition: str
    positive_extra: str
    camera: str
    expected_visible_hands: int
    hand_visibility: str
    foot_visibility: str
    difficulty: str = "medium"
    genital_visibility: str | None = None
    genital_angle: str | None = None
    local_only: bool = False
    raw: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PoseEntry":
        return cls(
            id=str(data["id"]),
            track=str(data["track"]),
            regions=[str(r) for r in data.get("regions", [])],
            composition=str(data.get("composition", "")).strip(),
            positive_extra=str(data.get("positive_extra", "")).strip(),
            camera=str(data.get("camera", "")).strip(),
            expected_visible_hands=int(data.get("expected_visible_hands", 0)),
            hand_visibility=str(data.get("hand_visibility", "unknown")),
            foot_visibility=str(data.get("foot_visibility", "unknown")),
            difficulty=str(data.get("difficulty", "medium")),
            genital_visibility=data.get("genital_visibility"),
            genital_angle=data.get("genital_angle"),
            local_only=bool(data.get("local_only", False)),
            raw=data,
        )


class PoseCatalog:
    def __init__(
        self,
        data: dict[str, Any],
        path: Path,
        *,
        nsfw_overlay_path: Path | None = None,
        nsfw_overlay_loaded: bool = False,
    ) -> None:
        self.path = path
        self.data = data
        self.nsfw_overlay_path = nsfw_overlay_path or DEFAULT_NSFW_CATALOG
        self.nsfw_overlay_loaded = nsfw_overlay_loaded
        self.whitelist = [PoseEntry.from_dict(x) for x in data.get("whitelist", [])]
        self.banned = list(data.get("banned", []))
        self.negatives = data.get("negatives", {})
        self.methods = data.get("methods", {})
        self.trigger = str(data.get("character_trigger", "rina_park_person"))

    def get(self, pose_id: str) -> PoseEntry:
        for entry in self.whitelist:
            if entry.id == pose_id:
                return entry
        if pose_id.startswith("nsfw_") and not self.nsfw_overlay_loaded:
            raise KeyError(
                f"pose not in whitelist: {pose_id} "
                f"(NSFW overlay missing or empty: {self.nsfw_overlay_path})"
            )
        raise KeyError(f"pose not in whitelist: {pose_id}")

    @property
    def by_id(self) -> dict[str, PoseEntry]:
        return {e.id: e for e in self.whitelist}

    def list_ids(self, track: str | None = None) -> list[str]:
        return [
            e.id
            for e in self.whitelist
            if track is None or e.track == track
        ]

    def reject_if_banned(self, text: str) -> list[str]:
        """Return matching ban ids if text contains banned keywords.

        Ignores negated phrases ('no typing', 'without keyboard') to avoid
        false positives from composition guidance text.
        """
        import re

        lowered = f" {text.lower()} "
        # Strip common negation windows so 'no typing' does not trip 'typing'
        cleaned = re.sub(
            r"\b(?:no|not|without|avoid|never)\s+[a-z0-9 \-]{0,40}",
            " ",
            lowered,
        )
        hits: list[str] = []
        for ban in self.banned:
            for kw in ban.get("match_keywords", []):
                needle = str(kw).lower().strip()
                if not needle:
                    continue
                # Word-boundary-ish: allow multi-word phrases
                pattern = r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])"
                if re.search(pattern, cleaned):
                    hits.append(str(ban["id"]))
                    break
        return hits

    def build_negative(self, track: str) -> str:
        # Keep short — CLIP neg also truncates ~77 tokens. Anatomy first.
        anatomy = (
            "mutated hands, extra fingers, fused fingers, bad hands, extra limbs, "
            "bad anatomy, mutated feet, extra toes, claw hands, melted hands"
        )
        quality = "plastic skin, CGI, anime, watermark, underage, teen"
        if track == "sfw":
            # Intentional SFW safety bans (keep in public tree).
            extra = "nude, explicit, nipples, genitals"
        else:
            # Prefer private overlay negatives; fallback is generic anatomy-only.
            extra = str(
                self.negatives.get("nsfw_anatomy_extra")
                or "mutated anatomy, melted anatomy, censored"
            ).replace("\n", " ").strip()
        return f"{anatomy}, {quality}, {extra}"

    def build_prompt(self, pose: PoseEntry, scene_glue: str = "") -> str:
        """Build a CLIP-friendly short prompt (keep critical tokens early)."""
        # SDXL CLIP truncates ~77 tokens — put identity + pose first.
        bits = [
            self.trigger,
            "adult Korean-Canadian woman 27, fair skin, long dark hair",
            "photoreal smartphone photo, natural skin",
            pose.positive_extra.replace("\n", " ").strip(),
            scene_glue.strip(),
            "coherent anatomy, careful hands, careful feet",
        ]
        prompt = ", ".join(b.strip().rstrip(",") for b in bits if b and b.strip())
        # Ban-check uses composition+glue (intent), not only final short prompt
        intent = " ".join(
            [
                pose.composition,
                pose.positive_extra,
                scene_glue,
            ]
        )
        bans = self.reject_if_banned(intent)
        if bans:
            raise ValueError(f"prompt hits banned poses {bans}: refuse generation")
        return prompt


def _merge_nsfw_overlay(data: dict[str, Any], overlay: dict[str, Any]) -> None:
    """Merge private NSFW whitelist + negatives into the public catalog dict (in place)."""
    base_wl = list(data.get("whitelist") or [])
    seen = {str(x.get("id")) for x in base_wl if isinstance(x, dict)}
    for entry in overlay.get("whitelist") or []:
        if not isinstance(entry, dict):
            continue
        eid = str(entry.get("id", ""))
        if not eid or eid in seen:
            continue
        base_wl.append(entry)
        seen.add(eid)
    data["whitelist"] = base_wl

    base_neg = dict(data.get("negatives") or {})
    for key, val in (overlay.get("negatives") or {}).items():
        if key not in base_neg:
            base_neg[key] = val
    data["negatives"] = base_neg


def load_pose_catalog(
    path: Path | None = None,
    nsfw_path: Path | None = None,
) -> PoseCatalog:
    catalog_path = path or DEFAULT_CATALOG
    data = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid pose catalog: {catalog_path}")

    overlay_path = DEFAULT_NSFW_CATALOG if nsfw_path is None else Path(nsfw_path)
    nsfw_loaded = False
    # Soft-merge when overlay exists; NSFW pose lookup fails clearly if absent.
    if overlay_path.is_file():
        overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
        if isinstance(overlay, dict):
            _merge_nsfw_overlay(data, overlay)
            nsfw_loaded = True

    return PoseCatalog(
        data,
        catalog_path,
        nsfw_overlay_path=overlay_path,
        nsfw_overlay_loaded=nsfw_loaded,
    )
