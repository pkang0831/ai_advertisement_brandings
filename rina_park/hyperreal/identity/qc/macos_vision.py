"""Local macOS Vision OCR adapter with explicit fail-closed availability."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MacOSVisionOCRAdapter:
    detector_id = "macos-vision-ocr"

    @staticmethod
    def _unavailable(reason: str) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "available": False,
            "detector_id": MacOSVisionOCRAdapter.detector_id,
            "detected": None,
            "regions": [],
            "confidence": None,
            "reason": reason,
        }

    def detect(self, image_path: Path) -> dict[str, Any]:
        try:
            from Foundation import NSURL  # type: ignore[import-not-found]
            from Quartz import (  # type: ignore[import-not-found]
                CGImageSourceCreateImageAtIndex,
                CGImageSourceCreateWithURL,
            )
            from Vision import (  # type: ignore[import-not-found]
                VNImageRequestHandler,
                VNRecognizeTextRequest,
                VNRequestTextRecognitionLevelAccurate,
            )
        except ImportError:
            return self._unavailable(
                "pyobjc Foundation/Quartz/Vision frameworks not installed in project venv"
            )

        try:
            url = NSURL.fileURLWithPath_(str(image_path.resolve()))
            source = CGImageSourceCreateWithURL(url, None)
            cg_image = CGImageSourceCreateImageAtIndex(source, 0, None) if source else None
            if cg_image is None:
                raise ValueError("could_not_decode_image")
            request = VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
            request.setUsesLanguageCorrection_(False)
            handler = VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, {})
            success, error = handler.performRequests_error_([request], None)
            if not success:
                raise RuntimeError(str(error or "Vision request failed"))
            regions: list[dict[str, Any]] = []
            for observation in request.results() or []:
                candidates = observation.topCandidates_(1)
                if not candidates:
                    continue
                candidate = candidates[0]
                box = observation.boundingBox()
                regions.append(
                    {
                        "text": str(candidate.string()),
                        "confidence": round(float(candidate.confidence()), 5),
                        "bbox_normalized": [
                            round(float(box.origin.x), 6),
                            round(float(box.origin.y), 6),
                            round(float(box.size.width), 6),
                            round(float(box.size.height), 6),
                        ],
                    }
                )
            confidence = max((item["confidence"] for item in regions), default=0.0)
            return {
                "status": "complete",
                "available": True,
                "detector_id": self.detector_id,
                "detected": bool(regions),
                "regions": regions,
                "confidence": confidence,
                "scope": "visible_text_candidate; watermark semantics require human review",
            }
        except Exception as error:  # Vision failures must remain approval blockers.
            return {
                **self._unavailable(f"vision_ocr_failed:{type(error).__name__}:{error}"),
                "status": "error",
                "available": True,
            }
