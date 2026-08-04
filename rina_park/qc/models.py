from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    MANUAL_REVIEW = "manual_review"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"


@dataclass(frozen=True)
class AdapterResult:
    available: bool
    passed: bool | None = None
    score: float | None = None
    detail: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def unavailable(cls, detail: str) -> "AdapterResult":
        return cls(available=False, detail=detail)


@dataclass(frozen=True)
class CheckResult:
    check: str
    status: Status
    severity: Severity
    detail: str
    score: float | None = None
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QCRequest:
    asset_path: Path
    platform: str
    prompt: str = ""
    caption: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    expected_width: int | None = None
    expected_height: int | None = None
    allowed_aspect_ratios: Sequence[float] = field(default_factory=tuple)
    track: str = "platform"
    duplicate_hashes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class QCReport:
    asset_path: Path
    results: Sequence[CheckResult]
    perceptual_hash: str | None = None

    @property
    def passed(self) -> bool:
        return not any(
            result.status is Status.FAIL and result.severity is Severity.BLOCKING
            for result in self.results
        )

    @property
    def requires_manual_review(self) -> bool:
        return any(result.status is Status.MANUAL_REVIEW for result in self.results)
