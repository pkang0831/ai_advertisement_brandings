from __future__ import annotations

import json
from pathlib import Path

import pytest

from rina_park.hyperreal.runners.hidream_o1 import (
    BakeoffRequest,
    CameraMetadata,
    HiDreamO1Runner,
    ModelNotReadyError,
    inspect_readiness,
)
from rina_park.hyperreal.runners.hidream_o1.readiness import (
    EXPECTED_REPO,
    EXPECTED_REVISION,
    REGISTRY_PATH,
    load_registry,
)


def test_registry_inventory_is_complete_and_pinned() -> None:
    component = load_registry()["component"]
    assert component["repo_id"] == EXPECTED_REPO
    assert component["revision"] == EXPECTED_REVISION
    assert component["file_count"] == len(component["files"]) == 24
    assert component["size_bytes"] == sum(
        entry["size_bytes"] for entry in component["files"].values()
    )
    assert all(
        len(entry["sha256"]) == 64 for entry in component["files"].values()
    )


def test_registry_has_no_required_external_text_encoder() -> None:
    licenses = load_registry()["component"]["text_encoder_licenses"]
    assert licenses["required_external_encoders"] == []
    assert licenses["optional_prompt_refiner"]["included_in_lane"] is False
    assert licenses["optional_prompt_refiner"]["base_model_terms"].startswith("Gemma")


def test_bakeoff_request_is_canonical_and_camera_aware() -> None:
    request = BakeoffRequest(
        case_id="portrait-01",
        prompt="A physically coherent studio portrait",
        seed=42,
        width=1024,
        height=1024,
        camera=CameraMetadata(
            camera_make="Sony",
            camera_model="ILCE-1",
            focal_length_mm=85.0,
            aperture_f=2.8,
            shutter_seconds=1 / 200,
            iso=100,
        ),
    )
    assert json.loads(request.canonical_json())["camera"]["focal_length_mm"] == 85.0
    assert request.request_sha256 == request.request_sha256
    assert len(request.request_sha256) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [("width", 1000), ("height", 1000), ("steps", 27)],
)
def test_request_rejects_recipe_drift(field: str, value: int) -> None:
    kwargs = {
        "case_id": "case",
        "prompt": "prompt",
        "seed": 0,
        "width": 1024,
        "height": 1024,
        "steps": 28,
        "camera": CameraMetadata(),
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        BakeoffRequest(**kwargs)


def test_readiness_fails_closed_without_substitution() -> None:
    report = inspect_readiness()
    assert report.status == "blocked_apple_exact_official_path_unverified"
    assert report.can_download is False
    assert report.can_load is False
    assert report.can_generate is False
    apple_check = next(
        check for check in report.checks if check.name == "apple_runtime_exact_official"
    )
    assert apple_check.passed is False


def test_registry_substitution_is_refused(tmp_path: Path) -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    payload["component"]["repo_id"] = "mlx-community/HiDream-O1-Image-Dev-mlx-q8"
    altered = tmp_path / "registry.yml"
    altered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="substitution refused"):
        load_registry(altered)


def test_runner_refuses_load_before_calling_adapter() -> None:
    called = False

    def loader(_path: Path) -> object:
        nonlocal called
        called = True
        return object()

    runner = HiDreamO1Runner(loader=loader)
    with pytest.raises(ModelNotReadyError, match="blocked_apple"):
        runner.load()
    assert called is False
