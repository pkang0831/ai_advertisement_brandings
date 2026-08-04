"""Soft-load private story overlays (home_mature) when present.

SFW YAML under this directory stays public. Mature/NSFW blocks live under
`rina_park/private/content/` and are merged only if those files exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

RINA = Path(__file__).resolve().parents[2]
PRIVATE_CONTENT = RINA / "private" / "content"
PUBLIC_DIR = Path(__file__).resolve().parent


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_scene_presets(path: Path | None = None) -> dict[str, Any]:
    base_path = path or (PUBLIC_DIR / "scene_presets.yml")
    data = _load_yaml(base_path)
    overlay = PRIVATE_CONTENT / "scene_presets_home_mature.yml"
    if overlay.is_file():
        extra = _load_yaml(overlay)
        presets = dict(data.get("presets") or {})
        presets.update(extra.get("presets") or {})
        data["presets"] = presets
    return data


def load_theme_pillars(path: Path | None = None) -> dict[str, Any]:
    base_path = path or (PUBLIC_DIR / "theme_pillars.yml")
    data = _load_yaml(base_path)
    overlay = PRIVATE_CONTENT / "theme_pillars_home_mature.yml"
    if overlay.is_file():
        extra = _load_yaml(overlay)
        pillars = dict(data.get("pillars") or {})
        pillars.update(extra.get("pillars") or {})
        data["pillars"] = pillars
    return data


def load_weekly_template(path: Path | None = None) -> dict[str, Any]:
    base_path = path or (PUBLIC_DIR / "weekly_template.yml")
    data = _load_yaml(base_path)
    overlay = PRIVATE_CONTENT / "weekly_home_mature_slots.yml"
    if overlay.is_file():
        extra = _load_yaml(overlay)
        days = dict(data.get("days") or {})
        for day_key, day_overlay in (extra.get("days") or {}).items():
            if not isinstance(day_overlay, dict):
                continue
            day = dict(days.get(day_key) or {})
            if "home_mature_slots" in day_overlay:
                day["home_mature_slots"] = day_overlay["home_mature_slots"]
            days[day_key] = day
        data["days"] = days
    return data
