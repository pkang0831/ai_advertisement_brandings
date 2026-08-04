"""Sequential, offline runner shell for a future verified Apple adapter."""

from __future__ import annotations

import contextlib
import fcntl
import os
import resource
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .contract import BakeoffRequest, RunMetrics, RunRecord
from .readiness import EXPECTED_REVISION, ReadinessReport, inspect_readiness, load_registry


class ModelNotReadyError(RuntimeError):
    """Raised instead of silently selecting another model or runtime."""


@contextlib.contextmanager
def _offline_environment() -> Iterator[None]:
    keys = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1",
    }
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update(keys)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def _exclusive_accelerator(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _accelerator_bytes() -> int | None:
    try:
        import torch

        if torch.backends.mps.is_available():
            return int(torch.mps.driver_allocated_memory())
    except (AttributeError, ImportError, RuntimeError):
        return None
    return None


@contextlib.contextmanager
def _measure() -> Iterator[dict[str, Any]]:
    start = time.perf_counter()
    accelerator_start = _accelerator_bytes()
    values: dict[str, Any] = {}
    try:
        yield values
    finally:
        accelerator_end = _accelerator_bytes()
        accelerator_peak = None
        if accelerator_start is not None or accelerator_end is not None:
            accelerator_peak = max(accelerator_start or 0, accelerator_end or 0)
        values["metrics"] = RunMetrics(
            wall_seconds=time.perf_counter() - start,
            process_max_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            accelerator_peak_bytes=accelerator_peak,
        )


class HiDreamO1Runner:
    """Adapter boundary that cannot run until the exact Apple path is approved."""

    def __init__(
        self,
        *,
        loader: Callable[[Path], Any] | None = None,
        generator: Callable[[Any, BakeoffRequest, Path], None] | None = None,
        generation_enabled: bool = False,
    ) -> None:
        registry = load_registry()
        component = registry["component"]
        self.model_path = (
            Path(__file__).resolve().parents[3]
            / "models"
            / component["local_path"]
        )
        self.code_revision = component["code_revision"]
        self.loader = loader
        self.generator = generator
        self.generation_enabled = generation_enabled
        self.lock_path = (
            Path(__file__).resolve().parents[3]
            / "models"
            / "hidream_o1"
            / ".accelerator.lock"
        )

    def readiness(self) -> ReadinessReport:
        return inspect_readiness()

    def load(self) -> tuple[Any, RunMetrics]:
        report = self.readiness()
        if not report.can_load:
            raise ModelNotReadyError(report.status)
        if self.loader is None:
            raise ModelNotReadyError("verified Apple model loader is not configured")
        measured: dict[str, Any] = {}
        with _exclusive_accelerator(self.lock_path), _offline_environment(), _measure() as measured:
            model = self.loader(self.model_path)
        return model, measured["metrics"]

    def run(
        self,
        request: BakeoffRequest,
        output_path: Path,
        record_path: Path,
    ) -> RunRecord:
        """Run one request sequentially and offline after a separate approval wave."""

        report = self.readiness()
        if not report.can_load:
            raise ModelNotReadyError(report.status)
        if not self.generation_enabled:
            raise ModelNotReadyError("image generation is disabled for this readiness wave")
        if self.loader is None or self.generator is None:
            raise ModelNotReadyError("verified Apple loader/generator adapter is not configured")
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite {output_path}")

        measured: dict[str, Any] = {}
        with _exclusive_accelerator(self.lock_path), _offline_environment(), _measure() as measured:
            model = self.loader(self.model_path)
            self.generator(model, request, output_path)
        if not output_path.is_file():
            raise RuntimeError("generator returned without creating the requested output")

        record = RunRecord(
            request_sha256=request.request_sha256,
            model_revision=EXPECTED_REVISION,
            code_revision=self.code_revision,
            runtime="apple_adapter_pending_verification",
            offline=True,
            metrics=measured["metrics"],
            output_path=str(output_path),
            status="succeeded",
        )
        record.write(record_path)
        return record
