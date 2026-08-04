from __future__ import annotations

import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rina_park.hyperreal.runners.z_image import readiness
from rina_park.hyperreal.runners.z_image.runner import (
    DEFAULT_STEPS,
    build_parser,
    load_bakeoff_manifest,
    manifest_plan,
)


class ManifestTests(unittest.TestCase):
    def write_manifest(self, root: str, payload: dict[str, object]) -> Path:
        path = Path(root) / "phase0.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_shared_camera_manifest_derives_seed_and_preserves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_manifest(
                directory,
                {
                    "schema_version": "1.0.0",
                    "output": {"width": 896, "height": 1152},
                    "seed_policy": {"base_seed": 26072600},
                    "camera_hypotheses": [
                        {
                            "id": "H01",
                            "prompt": "A coherent location photograph",
                            "equivalent_focal_length_mm": 35,
                            "camera_height_m": 1.3,
                            "lighting": {"key": "late afternoon side light"},
                        }
                    ],
                },
            )
            original_read_text = Path.read_text
            with patch.object(
                Path,
                "read_text",
                autospec=True,
                side_effect=lambda target, *args, **kwargs: original_read_text(
                    target, *args, **kwargs
                ),
            ) as read_text:
                plan = manifest_plan(path)
            self.assertEqual(read_text.call_count, 1)
            case = plan["cases"][0]
            self.assertEqual(case["seed"], 26072601)
            self.assertEqual(case["steps"], DEFAULT_STEPS)
            self.assertEqual(case["width"], 896)
            self.assertEqual(case["height"], 1152)
            self.assertEqual(
                case["camera_hypothesis"]["equivalent_focal_length_mm"], 35
            )

    def test_manifest_rejects_steps_outside_base_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_manifest(
                directory,
                {
                    "cases": [
                        {
                            "id": "bad",
                            "prompt": "photo",
                            "seed": 1,
                            "steps": 8,
                            "camera_hypothesis": {"lens_mm": 35},
                        }
                    ]
                },
            )
            with self.assertRaisesRegex(ValueError, "28 through 50"):
                load_bakeoff_manifest(path)

    def test_generation_requires_explicit_switch(self) -> None:
        args = build_parser().parse_args(["--manifest", "phase0.json"])
        self.assertFalse(args.allow_image_generation)
        self.assertIsNone(args.output_dir)


class ReadinessTests(unittest.TestCase):
    def test_no_network_sets_offline_mode_and_blocks_sockets(self) -> None:
        original = socket.create_connection
        with readiness.no_network():
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                socket.create_connection(("example.com", 443))
        self.assertIs(socket.create_connection, original)

    def test_artifact_inventory_size_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            artifact = model / "config.json"
            artifact.write_text("{}", encoding="utf-8")
            registry = {
                "size_bytes": 2,
                "artifacts": {
                    "config.json": {
                        "size_bytes": 2,
                        "sha256": readiness.sha256_file(artifact),
                    }
                },
            }
            report = readiness.verify_artifacts(registry, model_path=model)
            self.assertEqual(report["file_count"], 1)
            self.assertTrue(report["hashes_verified"])


if __name__ == "__main__":
    unittest.main()
