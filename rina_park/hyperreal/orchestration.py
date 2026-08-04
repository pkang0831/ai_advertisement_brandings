"""Serialized, readiness-gated orchestration for Phase-0 model adapters."""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import resource
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .prescreen import run_prescreen
from .spec import (
    DEFAULT_MANIFEST,
    LEGACY_MODEL_IDS,
    PHASE0_MODEL_IDS,
    load_manifest,
    write_blind_packet,
)

RINA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = RINA_ROOT / "out" / "hyperreal_phase0"
DEFAULT_MOODBOARD = RINA_ROOT / "moodboard"
DEFAULT_LOCK = Path("/tmp/rina-hyperreal-mlx-gpu.lock")
SIDECAR_SCHEMA = Path(__file__).resolve().parent / "sidecar.schema.v1.json"
ADAPTER_API_VERSION = "phase0-foundation-adapter-v1"
REQUIRED_READINESS_CHECKS = {
    "artifact_integrity",
    "runtime_backend",
    "license",
    "generation_smoke",
    "deterministic_request",
    "no_fallback",
}


@dataclass(frozen=True)
class AdapterRequest:
    """Model-neutral request containing every field required by the Qwen runner contract."""

    blind_id: str
    hypothesis_id: str
    model_id: str
    prompt: str
    seed: int
    width: int
    height: int
    camera_hypothesis: dict[str, Any]


class FoundationAdapter(Protocol):
    model_id: str
    adapter_api_version: str

    def readiness(self) -> dict[str, Any]:
        """Return a gate payload with ready(bool), checks(dict), and details."""

    def generate(self, request: AdapterRequest, output_path: Path) -> dict[str, Any] | None:
        """Generate exactly one image at output_path or raise; never substitute a model."""


def _memory_snapshot() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    snapshot: dict[str, Any] = {"ru_maxrss": int(usage.ru_maxrss)}
    try:
        import psutil  # type: ignore[import-not-found]

        process = psutil.Process(os.getpid())
        memory = process.memory_info()
        snapshot.update({"rss_bytes": int(memory.rss), "vms_bytes": int(memory.vms)})
    except ImportError:
        snapshot.update({"rss_bytes": None, "vms_bytes": None})
    return snapshot


def load_adapter(specification: str) -> FoundationAdapter:
    try:
        module_name, attribute_name = specification.split(":", 1)
    except ValueError as exc:
        raise ValueError("adapter must use module:attribute syntax") from exc
    module = importlib.import_module(module_name)
    value = getattr(module, attribute_name)
    adapter = value() if isinstance(value, type) else value
    for name in ("model_id", "adapter_api_version", "readiness", "generate"):
        if not hasattr(adapter, name):
            raise TypeError(f"adapter {specification} lacks {name}")
    return adapter


def require_all_ready(
    adapters: dict[str, FoundationAdapter], expected_models: list[str]
) -> dict[str, dict[str, Any]]:
    legacy = sorted(LEGACY_MODEL_IDS.intersection(adapters))
    if legacy:
        raise RuntimeError(
            f"legacy HiDream adapters are incompatible and cannot fallback: {legacy}; "
            f"required adapters are {list(PHASE0_MODEL_IDS)}"
        )
    if set(adapters) != set(expected_models):
        missing = sorted(set(expected_models).difference(adapters))
        unexpected = sorted(set(adapters).difference(expected_models))
        raise RuntimeError(f"exact model adapters required; missing={missing}, unexpected={unexpected}")
    reports: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for model_id in expected_models:
        adapter = adapters[model_id]
        if adapter.model_id != model_id:
            raise RuntimeError(f"adapter key/model mismatch: {model_id} != {adapter.model_id}")
        if adapter.adapter_api_version != ADAPTER_API_VERSION:
            raise RuntimeError(
                f"{model_id} adapter API is incompatible: {adapter.adapter_api_version!r}; "
                f"expected {ADAPTER_API_VERSION!r}"
            )
        report = adapter.readiness()
        if not isinstance(report, dict):
            raise RuntimeError(f"{model_id} readiness must return a mapping")
        if report.get("model_id") != model_id:
            raise RuntimeError(
                f"{model_id} readiness model_id mismatch: {report.get('model_id')!r}"
            )
        if report.get("adapter_api_version") != ADAPTER_API_VERSION:
            raise RuntimeError(
                f"{model_id} readiness API mismatch: {report.get('adapter_api_version')!r}"
            )
        checks = report.get("checks")
        if not isinstance(checks, dict):
            raise RuntimeError(f"{model_id} readiness checks must be a mapping")
        missing_checks = sorted(REQUIRED_READINESS_CHECKS.difference(checks))
        if missing_checks:
            raise RuntimeError(f"{model_id} readiness missing checks: {missing_checks}")
        reports[model_id] = report
        checks_pass = all(checks[name] is True for name in REQUIRED_READINESS_CHECKS)
        if report.get("ready") is not True or not checks_pass:
            failures.append(model_id)
    if failures:
        raise RuntimeError(
            "generation blocked: all model readiness gates must pass; failed=" + ",".join(failures)
        )
    return reports


@contextmanager
def serialized_gpu(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _camera_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in manifest["camera_hypotheses"]}


def validate_sidecar(sidecar: dict[str, Any]) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - required runtime dependency
        raise RuntimeError("jsonschema is required for Phase-0 sidecar validation") from exc
    schema = json.loads(SIDECAR_SCHEMA.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(sidecar), key=lambda item: list(item.path))
    if errors:
        details = "; ".join(error.message for error in errors[:5])
        raise ValueError(f"invalid Phase-0 sidecar: {details}")


def _labeled_tile(path: Path, label: str, size: tuple[int, int]) -> Image.Image:
    image_height = size[1] - 42
    tile = Image.new("RGB", size, "white")
    with Image.open(path) as source:
        fitted = ImageOps.contain(source.convert("RGB"), (size[0], image_height), Image.Resampling.LANCZOS)
    x = (size[0] - fitted.width) // 2
    tile.paste(fitted, (x, 0))
    draw = ImageDraw.Draw(tile)
    draw.text((10, image_height + 12), label, fill="black", font=ImageFont.load_default())
    return tile


def make_blinded_contact_sheets(
    blinded_outputs: dict[str, Path],
    review_directory: Path,
) -> tuple[list[str], str]:
    review_directory.mkdir(parents=True, exist_ok=False)
    pair_paths: list[str] = []
    for index in range(1, 7):
        hypothesis_id = f"H{index:02d}"
        pair = Image.new("RGB", (768, 536), "white")
        for arm_index, arm in enumerate(("A", "B")):
            blind_id = f"{hypothesis_id}_{arm}"
            pair.paste(_labeled_tile(blinded_outputs[blind_id], f"{hypothesis_id} · {arm}", (384, 536)), (arm_index * 384, 0))
        path = review_directory / f"{hypothesis_id}_pair_blinded.jpg"
        pair.save(path, quality=92, subsampling=0)
        pair_paths.append(str(path))

    overall = Image.new("RGB", (4 * 288, 3 * 402), "white")
    for position, blind_id in enumerate(sorted(blinded_outputs)):
        tile = _labeled_tile(blinded_outputs[blind_id], blind_id.replace("_", " · "), (288, 402))
        overall.paste(tile, ((position % 4) * 288, (position // 4) * 402))
    overall_path = review_directory / "contact_sheet_all_blinded.jpg"
    overall.save(overall_path, quality=92, subsampling=0)
    return pair_paths, str(overall_path)


def run_bakeoff(
    adapters: dict[str, FoundationAdapter],
    *,
    run_directory: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
    moodboard_directory: Path = DEFAULT_MOODBOARD,
    lock_path: Path = DEFAULT_LOCK,
    create_run_directory: bool = True,
) -> Path:
    manifest = load_manifest(manifest_path)
    readiness = require_all_ready(adapters, manifest["models"])
    moodboards = [moodboard_directory / name for name in manifest["moodboard_policy"]["references_reviewed"]]
    missing_moodboards = [str(path) for path in moodboards if not path.is_file()]
    if missing_moodboards:
        raise FileNotFoundError(f"moodboard copy checks require all references: {missing_moodboards}")

    slots = write_blind_packet(
        run_directory,
        manifest,
        create_run_directory=create_run_directory,
    )
    raw_outputs = run_directory / "private" / "raw_outputs"
    outputs = run_directory / "blinded"
    screens = run_directory / "prescreen"
    private_sidecars = run_directory / "private" / "sidecars"
    outputs.mkdir()
    raw_outputs.mkdir()
    screens.mkdir()
    private_sidecars.mkdir()
    (run_directory / "private" / "readiness.json").write_text(
        json.dumps(readiness, indent=2) + "\n", encoding="utf-8"
    )
    cameras = _camera_by_id(manifest)
    completed_outputs: list[Path] = []
    blinded_outputs: dict[str, Path] = {}
    rejects: list[str] = []
    run_started = time.monotonic()

    try:
        with serialized_gpu(lock_path):
            for slot in slots:
                adapter = adapters[slot.model_id]
                raw_output_path = raw_outputs / slot.model_id / slot.filename
                output_path = outputs / slot.filename
                request = AdapterRequest(
                    blind_id=slot.blind_id,
                    hypothesis_id=slot.hypothesis_id,
                    model_id=slot.model_id,
                    prompt=slot.prompt,
                    seed=slot.seed,
                    width=int(manifest["output"]["width"]),
                    height=int(manifest["output"]["height"]),
                    camera_hypothesis=cameras[slot.hypothesis_id],
                )
                started_wall = datetime.now(timezone.utc)
                started = time.monotonic()
                memory_before = _memory_snapshot()
                adapter_metadata = adapter.generate(request, raw_output_path)
                elapsed = time.monotonic() - started
                memory_after = _memory_snapshot()
                if not raw_output_path.is_file():
                    raise RuntimeError(
                        f"{slot.model_id} adapter returned without required output: {raw_output_path}"
                    )
                shutil.copy2(raw_output_path, output_path)
                sidecar = {
                    "schema_version": "1.0.0",
                    "experiment_id": manifest["experiment_id"],
                    "blind_id": slot.blind_id,
                    "hypothesis_id": slot.hypothesis_id,
                    "model_id": slot.model_id,
                    "seed": slot.seed,
                    "prompt": slot.prompt,
                    "camera_hypothesis": cameras[slot.hypothesis_id],
                    "output": {
                        "path": str(output_path),
                        "dimensions": [manifest["output"]["width"], manifest["output"]["height"]],
                        "format": manifest["output"]["format"],
                    },
                    "timing": {
                        "started_at_utc": started_wall.isoformat(),
                        "generation_seconds": round(elapsed, 3),
                    },
                    "memory": {"before": memory_before, "after": memory_after},
                    "adapter_metadata": {
                        **(adapter_metadata or {}),
                        "raw_output_path": str(raw_output_path),
                    },
                    "publication_allowed": False,
                    "published": False,
                }
                screen = run_prescreen(
                    output_path,
                    sidecar,
                    expected_dimensions=(
                        int(manifest["output"]["width"]),
                        int(manifest["output"]["height"]),
                    ),
                    moodboard_paths=moodboards,
                    comparison_outputs=completed_outputs,
                )
                sidecar["prescreen"] = screen
                validate_sidecar(sidecar)
                (private_sidecars / f"{slot.blind_id}.json").write_text(
                    json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
                )
                (screens / f"{slot.blind_id}.json").write_text(
                    json.dumps(screen, indent=2) + "\n", encoding="utf-8"
                )
                if not screen["passed"]:
                    rejects.append(slot.blind_id)
                completed_outputs.append(output_path)
                blinded_outputs[slot.blind_id] = output_path
    finally:
        cleanup: dict[str, Any] = {}
        for model_id, adapter in adapters.items():
            cleanup_method = getattr(adapter, "cleanup", None)
            if callable(cleanup_method):
                cleanup[model_id] = cleanup_method()

    paired_contact_sheets, overall_contact_sheet = make_blinded_contact_sheets(
        blinded_outputs,
        run_directory / "review",
    )

    summary = {
        "schema_version": "1.0.0",
        "experiment_id": manifest["experiment_id"],
        "planned": len(slots),
        "generated": len(completed_outputs),
        "all_readiness_gates_passed_before_generation": True,
        "gpu_serialized": True,
        "fallback_allowed": False,
        "publication_allowed": False,
        "total_seconds": round(time.monotonic() - run_started, 3),
        "prescreen_rejects": rejects,
        "cleanup_memory": cleanup,
        "review_sheet": "review_sheet.json",
        "mapping_key": "private/mapping_key.json",
        "paired_contact_sheets": paired_contact_sheets,
        "overall_blinded_contact_sheet": overall_contact_sheet,
    }
    (run_directory / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return run_directory


def _adapters_from_arguments(values: list[str]) -> dict[str, FoundationAdapter]:
    adapters: dict[str, FoundationAdapter] = {}
    for value in values:
        adapter = load_adapter(value)
        if adapter.model_id in adapters:
            raise ValueError(f"duplicate adapter model_id: {adapter.model_id}")
        adapters[adapter.model_id] = adapter
    return adapters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--moodboard-directory", type=Path, default=DEFAULT_MOODBOARD)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        help="Required twice as module:attribute, once for each manifest model.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    adapters = _adapters_from_arguments(args.adapter)
    run = run_bakeoff(
        adapters,
        run_directory=args.output_root / args.run_id,
        manifest_path=args.manifest,
        moodboard_directory=args.moodboard_directory,
    )
    print(json.dumps({"run_directory": str(run), "published": False}))


if __name__ == "__main__":
    main()
