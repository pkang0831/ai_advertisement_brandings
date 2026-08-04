from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

from .models import AdapterResult, QCRequest


class QCAdapter(Protocol):
    """Contract for optional local model integrations."""

    name: str

    def inspect(self, image_path: Path, request: QCRequest) -> AdapterResult: ...


class UnavailableAdapter:
    def __init__(self, name: str, reason: str = "model is not configured") -> None:
        self.name = name
        self.reason = reason

    def inspect(self, image_path: Path, request: QCRequest) -> AdapterResult:
        return AdapterResult.unavailable(f"{self.name}: {self.reason}")


class CallableAdapter:
    """Small adapter useful for local model wrappers and deterministic tests."""

    def __init__(
        self,
        name: str,
        callback: Callable[[Path, QCRequest], AdapterResult],
    ) -> None:
        self.name = name
        self._callback = callback

    def inspect(self, image_path: Path, request: QCRequest) -> AdapterResult:
        return self._callback(image_path, request)


def default_adapters() -> dict[str, QCAdapter]:
    return {
        "face_present": UnavailableAdapter("face_present"),
        "identity_similarity": UnavailableAdapter("identity_similarity"),
        "frame_occupancy": UnavailableAdapter("frame_occupancy"),
        "text_watermark": UnavailableAdapter("text_watermark"),
    }
