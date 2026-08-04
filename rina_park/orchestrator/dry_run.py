#!/usr/bin/env python3
"""One-week end-to-end rehearsal using tiny synthetic fixtures and no network."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from rina_park.factory.manifest import Manifest, utc_now
from rina_park.publisher.attestation import PromotionVerifier
from rina_park.publisher.common import sha256_file
from rina_park.publisher.instagram_package import build_instagram_package
from rina_park.publisher.patreon_package import build_patreon_package
from rina_park.publisher.promotion import promote, readiness
from rina_park.qc.engine import QCEngine
from rina_park.qc.models import AdapterResult, QCRequest

from .calendar import import_seed
from .runner import HeartbeatRunner, PipelineHooks, RunnerConfig

ROOT = Path(__file__).resolve().parents[1]
HASHES = {"prompt": "0" * 64, "workflow": "0" * 64, "model": "0" * 64}


class FixtureAdapter:
    def __init__(self, score: float | None = None) -> None:
        self.score = score

    def inspect(self, _path: Path, _request: QCRequest) -> AdapterResult:
        return AdapterResult(
            available=True,
            passed=True,
            score=self.score,
            detail="deterministic fixture adapter passed",
        )


class DryRunHooks:
    def __init__(
        self,
        root: Path,
        verifier: PromotionVerifier,
        attest: Callable[[Path, str, str], object],
    ) -> None:
        self.root = root
        self.verifier = verifier
        self.attest = attest
        self.assets = root / "assets"
        self.exports = root / "approved_exports"
        self.packages = root / "packages"
        self.assets.mkdir(parents=True, exist_ok=True)
        self.exports.mkdir(parents=True, exist_ok=True)
        (self.exports / "index.json").write_text(
            '{"version":1,"posts":{}}\n', encoding="utf-8"
        )

    def generate(self, manifest: Manifest, job: dict[str, object]) -> bool:
        post_id = str(job["post_id"])
        slot = str(job["asset_slot"])
        asset_id = f"{post_id}-{slot}"
        color_seed = hashlib.sha256(asset_id.encode()).hexdigest()[:6]
        if slot == "reel":
            path = self.assets / f"{asset_id}.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"color=c=0x{color_seed}:s=320x576:r=24:d=5",
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
                ],
                check=True,
                timeout=60,
            )
            media_type = "video"
        else:
            from PIL import Image, ImageDraw

            path = self.assets / f"{asset_id}.jpg"
            image = Image.new(
                "RGB",
                (400, 500),
                tuple(int(color_seed[index : index + 2], 16) for index in (0, 2, 4)),
            )
            draw = ImageDraw.Draw(image)
            draw.rectangle((30, 30, 370, 470), outline=(240, 240, 240), width=8)
            draw.text((55, 220), f"DRY RUN {asset_id}", fill=(255, 255, 255))
            image.save(path, "JPEG", quality=88)
            media_type = "image"
        manifest.register_asset(
            asset_id,
            path,
            media_type=media_type,
            prompt_hash=HASHES["prompt"],
            workflow_hash=HASHES["workflow"],
            model_hash=HASHES["model"],
            policy_version="fixture-only-v1",
        )
        manifest.attach_asset(post_id, asset_id, slot, selected=True)
        return True

    def qc(self, manifest: Manifest, post_id: str) -> bool:
        adapters = {
            "face_present": FixtureAdapter(),
            "identity_similarity": FixtureAdapter(0.99),
            "frame_occupancy": FixtureAdapter(0.45),
            "text_watermark": FixtureAdapter(),
        }
        engine = QCEngine(adapters=adapters, blur_variance_threshold=0)
        passed = True
        with manifest.connect() as conn:
            rows = conn.execute(
                "SELECT a.* FROM assets a JOIN post_assets pa ON pa.asset_id=a.asset_id "
                "WHERE pa.post_id=? AND pa.selected=1 ORDER BY pa.ordinal,a.asset_id",
                (post_id,),
            ).fetchall()
            post = conn.execute(
                "SELECT platform,body FROM posts WHERE post_id=?", (post_id,)
            ).fetchone()
        for row in rows:
            path = manifest.asset_root / row["relative_path"]
            if row["media_type"] == "image":
                report = engine.run(
                    QCRequest(
                        path,
                        str(post["platform"]),
                        prompt="fictional adult virtual swim diary",
                        caption=str(post["body"]),
                        allowed_aspect_ratios=(0.8,),
                    )
                )
                checks = report.results
                passed = passed and report.passed
            else:
                subprocess.run(
                    ["ffprobe", "-v", "error", "-show_format", str(path)],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                checks = ()
            with manifest.transaction() as conn:
                if checks:
                    for check in checks:
                        conn.execute(
                            "INSERT OR REPLACE INTO qc_results("
                            "asset_id,check_name,passed,score,details_json,"
                            "policy_version,checked_at) VALUES(?,?,?,?,?,?,?)",
                            (
                                row["asset_id"],
                                check.check,
                                int(check.status.value != "fail"),
                                check.score,
                                json.dumps({"detail": check.detail}),
                                "fixture-qc-v1",
                                utc_now(),
                            ),
                        )
                else:
                    conn.execute(
                        "INSERT OR REPLACE INTO qc_results VALUES("
                        "NULL,?,?,?,?,?,?,?)",
                        (
                            row["asset_id"], "ffprobe", 1, None, "{}",
                            "fixture-qc-v1", utc_now(),
                        ),
                    )
        return passed

    def _write_export(self, manifest: Manifest, post_id: str) -> tuple[str, Path]:
        with manifest.connect() as conn:
            post = dict(
                conn.execute("SELECT * FROM posts WHERE post_id=?", (post_id,)).fetchone()
            )
            assets = conn.execute(
                "SELECT a.* FROM assets a JOIN post_assets pa ON pa.asset_id=a.asset_id "
                "WHERE pa.post_id=? AND pa.selected=1 ORDER BY pa.asset_slot,pa.ordinal",
                (post_id,),
            ).fetchall()
        platform = str(post["platform"])
        destination = self.exports / post_id
        destination.mkdir(exist_ok=False)
        records = []
        for row in assets:
            source = manifest.asset_root / row["relative_path"]
            target = destination / source.name
            shutil.copy2(source, target)
            records.append(
                {
                    "asset_id": row["asset_id"],
                    "path": f"{post_id}/{target.name}",
                    "sha256": row["sha256"],
                    "media_type": (
                        "reel"
                        if platform == "instagram" and row["media_type"] == "video"
                        else row["media_type"]
                    ),
                }
            )
        post.update(
            audience_tiers=json.loads(post["audience_tiers"]),
            hashtags=" ".join(json.loads(post["hashtags"])),
            content_approval="approved",
            schedule_approval="approved",
            review_status="approved",
            production_status="approved",
            queue_id=f"dry-run-{post_id}",
            track=(
                "ig"
                if platform == "instagram"
                else f"patreon_{json.loads(post['audience_tiers'])[0].lower()}"
            ),
            assets=records,
        )
        qc_report = destination / "qc_report.json"
        qc_report.write_text(
            json.dumps({"fixture_only": True, "post_id": post_id, "passed": True}),
            encoding="utf-8",
        )
        post["qc_report_path"] = f"{post_id}/qc_report.json"
        post["qc_report_sha256"] = sha256_file(qc_report)
        manifest_path = destination / "manifest.json"
        manifest_path.write_text(
            json.dumps(post, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        index_path = self.exports / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["posts"][post_id] = {
            "manifest": f"{post_id}/manifest.json",
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "asset_ids": [record["asset_id"] for record in records],
        }
        index_path.write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return platform, manifest_path

    def package(self, manifest: Manifest, post_id: str) -> bool:
        platform, _ = self._write_export(manifest, post_id)
        self.attest(self.exports, post_id, platform)
        output = self.packages / platform
        if platform == "instagram":
            build_instagram_package(
                self.exports, post_id, output, verifier=self.verifier
            )
        else:
            build_patreon_package(
                self.exports, post_id, output, verifier=self.verifier
            )
        return True

    def simulate_publish(self, manifest: Manifest, post_id: str) -> bool:
        remote_id = f"dry-{post_id}"
        now = utc_now()
        with manifest.transaction() as conn:
            conn.execute(
                "INSERT INTO publish_attempts(platform,post_id,attempt_no,"
                "container_id,request_hash,response_json,status,lease_owner,"
                "lease_expires_at,error_class,created_at,updated_at) "
                "VALUES('instagram',?,1,?,?,?,'succeeded',NULL,NULL,NULL,?,?)",
                (
                    post_id,
                    f"fixture-container-{post_id}",
                    hashlib.sha256(post_id.encode()).hexdigest(),
                    json.dumps({"mocked": True, "remote_media_id": remote_id}),
                    now,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO publication_records(platform,post_id,remote_media_id,"
                "remote_url,published_at,created_at) VALUES('instagram',?,?,?,?,?)",
                (post_id, remote_id, f"mock://instagram/{remote_id}", now, now),
            )
        return True


def run(
    output: Path,
    *,
    verifier: PromotionVerifier,
    attest: Callable[[Path, str, str], object],
) -> dict[str, object]:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    hooks = DryRunHooks(output, verifier, attest)
    database = output / "manifest.db"
    manifest = Manifest(database, hooks.assets)
    imported = import_seed(manifest, ROOT / "content" / "calendar_8_weeks.csv")
    runner = HeartbeatRunner(
        RunnerConfig(
            manifest_db=database,
            asset_root=hooks.assets,
            log_root=output / "logs",
            generation_horizon_days=0,
            max_publish_lateness_hours=24 * 7,
            generation_enabled=True,
            dry_run=True,
        ),
        PipelineHooks(hooks.generate, hooks.qc, hooks.package, hooks.simulate_publish),
    )
    now = datetime(2026, 8, 16, 15, 0, tzinfo=timezone.utc)
    for _ in range(80):
        runner.run_once(now)
        with manifest.connect() as conn:
            week = conn.execute(
                "SELECT state,count(*) count FROM posts WHERE story_week=1 GROUP BY state"
            ).fetchall()
        states = {row["state"]: row["count"] for row in week}
        if states.get("published") == 4 and states.get("packaged") == 3:
            break
        with manifest.connect() as conn:
            ready = conn.execute(
                "SELECT post_id,state FROM posts WHERE story_week=1 "
                "AND state IN ('assets_ready','content_approved')"
            ).fetchall()
        for row in ready:
            if row["state"] == "assets_ready":
                manifest.approve(row["post_id"], "content", "dry-run-hitl")
            else:
                manifest.approve(row["post_id"], "schedule", "dry-run-hitl")
    else:
        raise RuntimeError("dry run did not converge")

    with sqlite3.connect(database) as conn:
        conn.row_factory = sqlite3.Row
        state_rows = conn.execute(
            "SELECT state,count(*) count FROM posts WHERE story_week=1 GROUP BY state"
        ).fetchall()
        counts = {
            "assets": conn.execute("SELECT count(*) FROM assets").fetchone()[0],
            "qc_results": conn.execute("SELECT count(*) FROM qc_results").fetchone()[0],
            "approvals": conn.execute("SELECT count(*) FROM approvals").fetchone()[0],
            "publication_records": conn.execute(
                "SELECT count(*) FROM publication_records"
            ).fetchone()[0],
        }
    summary = {
        "mode": "DRY_RUN_MOCKS_ONLY",
        "seed_rows_imported": imported,
        "week_1_states": {row["state"]: row["count"] for row in state_rows},
        **counts,
        "graph_api_called": False,
        "patreon_mode": "manual_package",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "dry_run_week_1",
    )
    args = parser.parse_args()
    report = readiness()
    if not report["ready"]:
        raise RuntimeError(
            "production promotion is blocked: " + "; ".join(report["reasons"])
        )
    verifier = PromotionVerifier()
    print(
        json.dumps(
            run(
                args.output.resolve(),
                verifier=verifier,
                attest=lambda root, post_id, platform: promote(
                    root,
                    post_id,
                    platform,
                    "package",
                ),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
