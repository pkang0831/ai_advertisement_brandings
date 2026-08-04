from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

from .common import PublisherError


def load_capabilities(path: Path | None = None) -> dict[str, Any]:
    source = path or Path(__file__).with_name("capabilities.yml")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublisherError("capabilities.yml is unavailable or malformed") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise PublisherError("unsupported capabilities schema")
    return value


def _image_size(path: Path) -> tuple[str, int, int]:
    data = path.read_bytes()
    if data.startswith(b"\xff\xd8"):
        offset = 2
        while offset + 9 < len(data):
            if data[offset] != 0xFF:
                offset += 1
                continue
            marker = data[offset + 1]
            length = struct.unpack(">H", data[offset + 2 : offset + 4])[0]
            if marker in range(0xC0, 0xC4):
                height, width = struct.unpack(">HH", data[offset + 5 : offset + 9])
                return "image/jpeg", width, height
            offset += 2 + length
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return "image/png", width, height
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        try:
            from PIL import Image

            with Image.open(path) as image:
                return "image/webp", image.width, image.height
        except (ImportError, OSError) as exc:
            raise PublisherError("valid WebP inspection requires Pillow") from exc
    raise PublisherError("unsupported or malformed image")


def validate_image(path: Path, rules: dict[str, Any]) -> dict[str, Any]:
    if path.suffix.lower() not in rules["extensions"] or path.stat().st_size > rules["max_bytes"]:
        raise PublisherError("image extension or file size violates capabilities")
    mime, width, height = _image_size(path)
    if mime not in rules["mime_types"] or width < rules["min_width"]:
        raise PublisherError("image MIME or dimensions violate capabilities")
    if rules.get("max_width") and width > rules["max_width"]:
        raise PublisherError("image width exceeds platform capability")
    ratio = width / height
    if ratio < rules.get("aspect_ratio_min", 0) or ratio > rules.get("aspect_ratio_max", 999):
        raise PublisherError("image aspect ratio violates capabilities")
    return {"mime_type": mime, "width": width, "height": height, "bytes": path.stat().st_size}


def validate_video(path: Path, rules: dict[str, Any]) -> dict[str, Any]:
    if path.suffix.lower() not in rules["extensions"] or path.stat().st_size > rules["max_bytes"]:
        raise PublisherError("video extension or file size violates capabilities")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise PublisherError("ffprobe is required to validate video; use the UI package fallback")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        probe = json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise PublisherError("ffprobe rejected or could not inspect video") from exc
    video = next((stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"), None)
    if not video or video.get("codec_name") not in rules["codecs"]:
        raise PublisherError("video codec violates capabilities")
    if audio and audio.get("codec_name") not in rules.get("audio_codecs", [audio.get("codec_name")]):
        raise PublisherError("audio codec violates capabilities")
    duration = float(probe.get("format", {}).get("duration", video.get("duration", 0)))
    width, height = int(video.get("width", 0)), int(video.get("height", 0))
    rate = video.get("avg_frame_rate", "0/1").split("/")
    fps = float(rate[0]) / float(rate[1]) if len(rate) == 2 and float(rate[1]) else 0
    ratio = width / height if height else 0
    checks = (
        duration >= rules.get("min_duration_seconds", 0),
        duration <= rules.get("max_duration_seconds", float("inf")),
        fps >= rules.get("min_fps", 0),
        fps <= rules.get("max_fps", float("inf")),
        ratio >= rules.get("aspect_ratio_min", 0),
        ratio <= rules.get("aspect_ratio_max", float("inf")),
    )
    if not all(checks):
        raise PublisherError("video duration, fps, or aspect ratio violates capabilities")
    return {
        "codec": video["codec_name"],
        "width": width,
        "height": height,
        "fps": fps,
        "duration_seconds": duration,
        "bytes": path.stat().st_size,
    }


def validate_asset(path: Path, media_type: str, platform: str) -> dict[str, Any]:
    capabilities = load_capabilities()[platform]
    if media_type == "image":
        return validate_image(path, capabilities["image"])
    video_key = "reel" if platform == "instagram" else "video"
    if media_type in {"video", "reel"}:
        return validate_video(path, capabilities[video_key])
    raise PublisherError("asset media_type is not supported")
