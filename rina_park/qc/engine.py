from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from .adapters import QCAdapter, default_adapters
from .models import AdapterResult, CheckResult, QCReport, QCRequest, Severity, Status
from .policy import inspect_text


def _hamming(left: str, right: str) -> int:
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except (TypeError, ValueError):
        return 10_000


class QCEngine:
    """Runs deterministic checks and optional model-backed adapters."""

    def __init__(
        self,
        adapters: Mapping[str, QCAdapter] | None = None,
        *,
        black_mean_threshold: float = 3.0,
        blur_variance_threshold: float = 40.0,
        duplicate_hamming_threshold: int = 5,
        identity_threshold: float = 0.75,
    ) -> None:
        self.adapters = dict(default_adapters())
        if adapters:
            self.adapters.update(adapters)
        self.black_mean_threshold = black_mean_threshold
        self.blur_variance_threshold = blur_variance_threshold
        self.duplicate_hamming_threshold = duplicate_hamming_threshold
        self.identity_threshold = identity_threshold

    def run(self, request: QCRequest) -> QCReport:
        results: list[CheckResult] = [
            inspect_text(request.prompt, request.caption, request.track, request.metadata)
        ]
        image, load_result = self._load_image(request.asset_path)
        results.append(load_result)
        perceptual_hash: str | None = None

        if image is not None:
            results.extend(self._image_checks(image, request))
            perceptual_hash = self._dhash(image)
            results.append(self._duplicate_check(perceptual_hash, request.duplicate_hashes))

        results.extend(
            (
                self._adapter_check("face_present", request, blocking_when_false=True),
                self._identity_check(request),
                self._occupancy_check(request),
                self._adapter_check("text_watermark", request, blocking_when_false=True),
            )
        )
        return QCReport(request.asset_path, tuple(results), perceptual_hash)

    @staticmethod
    def _load_image(path: Path):
        try:
            from PIL import Image
        except ImportError:
            return None, CheckResult(
                "image_decode",
                Status.MANUAL_REVIEW,
                Severity.WARNING,
                "Pillow is not installed; deterministic image checks were skipped",
            )
        try:
            image = Image.open(path)
            image.load()
            return image.convert("RGB"), CheckResult(
                "image_decode", Status.PASS, Severity.BLOCKING, "Image decoded successfully"
            )
        except Exception as exc:
            return None, CheckResult(
                "image_decode",
                Status.FAIL,
                Severity.BLOCKING,
                f"Image could not be decoded: {type(exc).__name__}",
            )

    def _image_checks(self, image, request: QCRequest) -> list[CheckResult]:
        width, height = image.size
        resolution_ok = (
            (request.expected_width is None or width == request.expected_width)
            and (request.expected_height is None or height == request.expected_height)
        )
        resolution = CheckResult(
            "resolution",
            Status.PASS if resolution_ok else Status.FAIL,
            Severity.BLOCKING,
            f"Actual resolution is {width}x{height}",
            data={"width": width, "height": height},
        )

        actual_ratio = width / height
        ratio_ok = not request.allowed_aspect_ratios or any(
            math.isclose(actual_ratio, allowed, rel_tol=0.02, abs_tol=0.02)
            for allowed in request.allowed_aspect_ratios
        )
        aspect = CheckResult(
            "aspect_ratio",
            Status.PASS if ratio_ok else Status.FAIL,
            Severity.BLOCKING,
            f"Aspect ratio is {actual_ratio:.4f}",
            score=actual_ratio,
        )

        pixels = list(image.resize((32, 32)).convert("L").getdata())
        mean = sum(pixels) / len(pixels)
        black = CheckResult(
            "black_frame",
            Status.FAIL if mean <= self.black_mean_threshold else Status.PASS,
            Severity.BLOCKING,
            f"Mean luminance is {mean:.2f}",
            score=mean,
        )
        blur = self._blur_check(image)
        return [resolution, aspect, black, blur]

    def _blur_check(self, image) -> CheckResult:
        try:
            import cv2
            import numpy as np

            variance = float(cv2.Laplacian(np.asarray(image.convert("L")), cv2.CV_64F).var())
        except ImportError:
            from PIL import ImageFilter, ImageStat

            edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
            if edges.width > 2 and edges.height > 2:
                edges = edges.crop((1, 1, edges.width - 1, edges.height - 1))
            variance = float(ImageStat.Stat(edges).var[0])
            method = "Pillow edge variance"
        else:
            method = "Laplacian variance"
        return CheckResult(
            "blur",
            Status.PASS if variance >= self.blur_variance_threshold else Status.FAIL,
            Severity.BLOCKING,
            f"{method} is {variance:.2f}",
            score=variance,
        )

    @staticmethod
    def _dhash(image) -> str:
        gray = image.convert("L").resize((9, 8))
        pixels = list(gray.getdata())
        bits = 0
        for row in range(8):
            for column in range(8):
                bits = (bits << 1) | (
                    pixels[row * 9 + column] > pixels[row * 9 + column + 1]
                )
        return f"{bits:016x}"

    def _duplicate_check(
        self, current_hash: str, known_hashes: Mapping[str, str]
    ) -> CheckResult:
        duplicates = [
            asset_id
            for asset_id, known_hash in known_hashes.items()
            if _hamming(current_hash, known_hash) <= self.duplicate_hamming_threshold
        ]
        return CheckResult(
            "perceptual_duplicate",
            Status.FAIL if duplicates else Status.PASS,
            Severity.BLOCKING,
            f"Near-duplicate assets: {', '.join(duplicates)}" if duplicates else "No near duplicate",
            data={"asset_ids": duplicates, "hash": current_hash},
        )

    def _inspect_adapter(self, name: str, request: QCRequest) -> AdapterResult:
        try:
            return self.adapters[name].inspect(request.asset_path, request)
        except Exception as exc:
            return AdapterResult.unavailable(f"{name} adapter error: {type(exc).__name__}")

    def _adapter_check(
        self, name: str, request: QCRequest, *, blocking_when_false: bool
    ) -> CheckResult:
        result = self._inspect_adapter(name, request)
        if not result.available or result.passed is None:
            return CheckResult(
                name, Status.MANUAL_REVIEW, Severity.WARNING, result.detail, result.score, result.data
            )
        return CheckResult(
            name,
            Status.PASS if result.passed else Status.FAIL,
            Severity.BLOCKING if blocking_when_false else Severity.WARNING,
            result.detail,
            result.score,
            result.data,
        )

    def _identity_check(self, request: QCRequest) -> CheckResult:
        result = self._inspect_adapter("identity_similarity", request)
        if not result.available or result.score is None:
            return CheckResult(
                "identity_similarity",
                Status.MANUAL_REVIEW,
                Severity.WARNING,
                result.detail or "Identity score unavailable",
            )
        passed = result.passed is not False and result.score >= self.identity_threshold
        return CheckResult(
            "identity_similarity",
            Status.PASS if passed else Status.FAIL,
            Severity.BLOCKING,
            result.detail or f"Identity similarity is {result.score:.3f}",
            result.score,
            result.data,
        )

    def _occupancy_check(self, request: QCRequest) -> CheckResult:
        result = self._inspect_adapter("frame_occupancy", request)
        if not result.available or result.score is None:
            return CheckResult(
                "frame_occupancy",
                Status.MANUAL_REVIEW,
                Severity.WARNING,
                result.detail or "Subject occupancy unavailable; close-up must be checked manually",
            )
        occupancy = result.score
        if occupancy > 0.65:
            status, severity, detail = (
                Status.FAIL,
                Severity.BLOCKING,
                "Excessive close-up: subject occupancy exceeds 65%",
            )
        elif occupancy > 0.55:
            status, severity, detail = (
                Status.MANUAL_REVIEW,
                Severity.WARNING,
                "Occupancy exceeds the normal 30–55% range",
            )
        elif occupancy < 0.30:
            status, severity, detail = (
                Status.MANUAL_REVIEW,
                Severity.WARNING,
                "Occupancy is below the normal 30–55% range",
            )
        else:
            status, severity, detail = Status.PASS, Severity.INFO, "Occupancy is within 30–55%"
        return CheckResult(
            "frame_occupancy", status, severity, detail, occupancy, result.data
        )
