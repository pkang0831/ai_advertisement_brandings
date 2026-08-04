from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ReviewProfile(str, Enum):
    PLATFORM = "platform"
    MATURE = "mature_non_explicit"


@dataclass(frozen=True)
class ReviewCapabilities:
    compare_candidates: bool = True
    edit_content: bool = True
    request_regeneration: bool = True
    content_approval: bool = True
    schedule_approval: bool = True
    export: bool = False
    package: bool = False
    publish: bool = False


def capabilities_for(profile: ReviewProfile) -> ReviewCapabilities:
    if profile is ReviewProfile.MATURE:
        return ReviewCapabilities(
            schedule_approval=False,
            export=False,
            package=False,
            publish=False,
        )
    return ReviewCapabilities()


def assert_profile_isolation(profile: ReviewProfile, db_path: str) -> None:
    lowered = db_path.lower()
    if profile is ReviewProfile.MATURE and "mature" not in lowered:
        raise ValueError("Mature review profile requires a dedicated DB path containing 'mature'")
    if profile is ReviewProfile.PLATFORM and "mature" in lowered:
        raise ValueError("Platform review must not open a mature DB")
