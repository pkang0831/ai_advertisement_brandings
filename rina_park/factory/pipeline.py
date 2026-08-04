"""Queue derivation and lightweight video V0 scaffolding."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from .manifest import Manifest

TRACK_BY_PLATFORM = {
    ("instagram", None): "ig",
    ("patreon", "a"): "patreon_a",
    ("patreon", "b"): "patreon_b",
    ("patreon", "c"): "patreon_c",
}


def derive_generation_jobs(
    manifest: Manifest,
    post: Mapping[str, Any],
    asset_slots: Iterable[str],
    hashes: Mapping[str, str],
    generation_version: int = 1,
    candidate_count: int = 4,
) -> list[int]:
    """Derive idempotent candidate jobs from one calendar post.

    Existing UNIQUE keys are treated as already-derived work.
    """
    tiers = post.get("audience_tiers") or []
    tier = tiers[0].lower() if post["platform"] == "patreon" and tiers else None
    track = TRACK_BY_PLATFORM.get((post["platform"], tier))
    if track is None:
        raise ValueError("post does not map to an SFW platform track")
    ids: list[int] = []
    for slot in asset_slots:
        for candidate in range(candidate_count):
            try:
                ids.append(
                    manifest.enqueue_job(
                        post["post_id"], slot, generation_version,
                        candidate, track, hashes,
                    )
                )
            except Exception as exc:
                if "UNIQUE constraint failed" not in str(exc):
                    raise
    return ids


def ffmpeg_v0_command(
    source_image: str | Path,
    output_video: str | Path,
    duration_seconds: int = 6,
    fps: int = 30,
) -> list[str]:
    if duration_seconds < 5 or duration_seconds > 8:
        raise ValueError("V0 duration must be 5-8 seconds")
    frames = duration_seconds * fps
    # Deterministic subtle Ken Burns motion, vertical H.264, silent master.
    vf = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        f"zoompan=z='min(zoom+0.0005,1.04)':d={frames}:"
        "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"s=1080x1920:fps={fps},format=yuv420p"
    )
    return [
        "ffmpeg", "-y", "-loop", "1", "-i", str(source_image),
        "-vf", vf, "-t", str(duration_seconds), "-an",
        "-c:v", "libx264", "-profile:v", "high", "-movflags", "+faststart",
        str(output_video),
    ]


def render_ffmpeg_v0(
    source_image: str | Path,
    output_video: str | Path,
    duration_seconds: int = 6,
    timeout_seconds: int = 180,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ffmpeg_v0_command(source_image, output_video, duration_seconds),
        check=True, capture_output=True, text=True, timeout=timeout_seconds,
    )
