"""Pinned, offline validation providers for Phase-3 calibration."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np


RINA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RINA_ROOT.parent))
from rina_park.runtime_device import get_torch_device_str  # noqa: E402

LPIPS_ROOT = RINA_ROOT / "models/lpips/official-0.1.4"
LPIPS_LINEAR_PATH = LPIPS_ROOT / "alex-v0.1.pth"
ALEXNET_PATH = LPIPS_ROOT / "alexnet-owt-7be5be79.pth"


class OfflineAlexLpips:
    """Official LPIPS v0.1 linear head with an explicitly local AlexNet trunk."""

    provider_id = "lpips-0.1.4-alex-v0.1-local"

    def __init__(self) -> None:
        if not LPIPS_LINEAR_PATH.is_file() or not ALEXNET_PATH.is_file():
            raise RuntimeError("pinned_local_lpips_assets_missing")
        os.environ["TORCH_HOME"] = str(LPIPS_ROOT / "offline_torch_home")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import lpips
        import torch
        from lpips import pretrained_networks
        from torch import nn
        from torchvision.models import alexnet

        device = (
            torch.device(get_torch_device_str())
            if get_torch_device_str() != "cpu"
            else torch.device("cpu")
        )
        trunk = alexnet(weights=None)
        state = torch.load(ALEXNET_PATH, map_location="cpu", weights_only=True)
        trunk.load_state_dict(state, strict=True)
        features = list(trunk.features.children())
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The parameter 'pretrained' is deprecated",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Arguments other than a weight enum",
                category=UserWarning,
            )
            wrapped = pretrained_networks.alexnet(
                pretrained=False,
                requires_grad=False,
            )
        wrapped.slice1 = nn.Sequential(*features[0:2])
        wrapped.slice2 = nn.Sequential(*features[2:5])
        wrapped.slice3 = nn.Sequential(*features[5:8])
        wrapped.slice4 = nn.Sequential(*features[8:10])
        wrapped.slice5 = nn.Sequential(*features[10:12])

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The parameter 'pretrained' is deprecated",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Arguments other than a weight enum",
                category=UserWarning,
            )
            model = lpips.LPIPS(
                net="alex",
                version="0.1",
                pretrained=True,
                pnet_rand=True,
                model_path=str(LPIPS_LINEAR_PATH),
                use_dropout=False,
                eval_mode=True,
                verbose=False,
            )
        model.net = wrapped
        model.requires_grad_(False)
        self._torch = torch
        self._device = device
        self._model = model.to(device).eval()

    @property
    def device(self) -> str:
        return self._device.type

    def __call__(self, before: np.ndarray, after: np.ndarray) -> float:
        if before.shape != after.shape:
            raise ValueError("lpips_inputs_must_have_equal_shape")
        if before.ndim != 3 or before.shape[2] != 3:
            raise ValueError("lpips_inputs_must_be_HxWx3")
        first = self._tensor(before)
        second = self._tensor(after)
        with self._torch.inference_mode():
            value = self._model(first, second, normalize=True)
            if self._device.type == "cuda":
                self._torch.cuda.synchronize()
            elif self._device.type == "mps":
                self._torch.mps.synchronize()
        # Official linear LPIPS heads can produce tiny negative floating-point
        # values around identical inputs; perceptual distance is reported as zero.
        return max(0.0, float(value.detach().cpu().item()))

    def _tensor(self, image: np.ndarray):
        array = np.asarray(image, dtype=np.float32)
        array = np.clip(array, 0.0, 1.0)
        tensor = self._torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self._device)


class LocalFaceLandmarks:
    """MediaPipe Face Landmarker using the existing pinned Apache-2.0 task."""

    provider_id = "mediapipe-face-landmarker-local-0.10.35"

    def __init__(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        from hyperreal.identity.qc.registry import verified_model_paths

        face_model = verified_model_paths()["face"]
        self._mp = mp
        self._detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(face_model)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=2,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
        )

    def __call__(self, path: Path) -> list[np.ndarray]:
        image = self._mp.Image.create_from_file(str(path))
        result = self._detector.detect(image)
        return [
            np.asarray([(float(point.x), float(point.y)) for point in face], dtype=np.float32)
            for face in result.face_landmarks
        ]

    def close(self) -> None:
        self._detector.close()

    def __enter__(self) -> "LocalFaceLandmarks":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()
