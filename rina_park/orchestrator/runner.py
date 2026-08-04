"""Five-minute heartbeat runner for the local platform manifest."""

from __future__ import annotations

import json
import os
import socket
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from rina_park.factory.manifest import Manifest, utc_now
from rina_park.factory.pipeline import derive_generation_jobs

from .calendar import due_posts
from .gates import DEFAULT_GRAPH_GATES, GraphGates
from .logging import JsonLogger

UTC = timezone.utc


@dataclass(frozen=True)
class RunnerConfig:
    manifest_db: Path
    asset_root: Path
    log_root: Path
    heartbeat_seconds: int = 300
    lease_seconds: int = 270
    gpu_lease_seconds: int = 3600
    generation_horizon_days: int = 14
    max_publish_lateness_hours: int = 6
    graph_gates: GraphGates = DEFAULT_GRAPH_GATES
    generation_enabled: bool = False
    dry_run: bool = False


@dataclass
class PipelineHooks:
    generate: Callable[[Manifest, dict[str, object]], bool] = lambda _m, _j: False
    qc: Callable[[Manifest, str], bool] = lambda _m, _p: False
    package: Callable[[Manifest, str], bool] = lambda _m, _p: False
    simulate_publish: Callable[[Manifest, str], bool] = lambda _m, _p: False
    publish_graph: Callable[[Manifest, str], bool] = lambda _m, _p: False


class HeartbeatRunner:
    def __init__(
        self,
        config: RunnerConfig,
        hooks: PipelineHooks | None = None,
        owner: str | None = None,
    ) -> None:
        if config.heartbeat_seconds != 300:
            raise ValueError("launchd heartbeat must remain five minutes")
        self.config = config
        self.manifest = Manifest(config.manifest_db, config.asset_root)
        self.manifest.migrate()
        self.hooks = hooks or PipelineHooks()
        self.owner = owner or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self.log = JsonLogger(config.log_root)

    @staticmethod
    def _stamp(moment: datetime) -> str:
        return moment.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _acquire(self, name: str, seconds: int, now: datetime) -> bool:
        expires = self._stamp(now + timedelta(seconds=seconds))
        stamp = self._stamp(now)
        with self.manifest.transaction() as conn:
            row = conn.execute(
                "SELECT owner,expires_at FROM orchestrator_leases WHERE lease_name=?",
                (name,),
            ).fetchone()
            if row and row["owner"] != self.owner and row["expires_at"] >= stamp:
                return False
            conn.execute(
                "INSERT INTO orchestrator_leases(lease_name,owner,expires_at,updated_at) "
                "VALUES(?,?,?,?) ON CONFLICT(lease_name) DO UPDATE SET "
                "owner=excluded.owner,expires_at=excluded.expires_at,"
                "updated_at=excluded.updated_at",
                (name, self.owner, expires, stamp),
            )
        return True

    def _release(self, name: str) -> None:
        with self.manifest.transaction() as conn:
            conn.execute(
                "DELETE FROM orchestrator_leases WHERE lease_name=? AND owner=?",
                (name, self.owner),
            )

    def _approval_valid(self, post_id: str, approval_type: str) -> bool:
        with self.manifest.connect() as conn:
            row = conn.execute(
                "SELECT a.decision,a.snapshot_hash FROM approvals a "
                "WHERE a.post_id=? AND a.approval_type=? "
                "ORDER BY a.approval_id DESC LIMIT 1",
                (post_id, approval_type),
            ).fetchone()
        return bool(
            row
            and row["decision"] == "approved"
            and row["snapshot_hash"] == self.manifest.approval_snapshot_hash(post_id)
        )

    def _derive_due_generation(self, now: datetime) -> int:
        horizon = self._stamp(now + timedelta(days=self.config.generation_horizon_days))
        with self.manifest.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM posts WHERE state='draft' AND publish_at_utc<=? "
                "ORDER BY publish_at_utc",
                (horizon,),
            ).fetchall()
        count = 0
        hashes = {"prompt": "0" * 64, "workflow": "0" * 64, "model": "0" * 64}
        for row in rows:
            post = dict(row)
            post["audience_tiers"] = json.loads(str(post["audience_tiers"]))
            post_format = str(post["format"]).lower()
            if post_format == "reel":
                slots = ["reel"]
            elif post["platform"] == "instagram":
                slots = ["carousel_1", "carousel_2"]
            else:
                slot_count = (
                    4 if post_format.startswith("gallery_4")
                    else 6 if post_format.startswith("gallery_6")
                    else 8
                )
                slots = [f"gallery_{index}" for index in range(1, slot_count + 1)]
            count += len(
                derive_generation_jobs(
                    self.manifest, post, slots, hashes,
                    candidate_count=1 if self.config.dry_run else 4,
                )
            )
            self.manifest.transition(str(post["post_id"]), "generating", self.owner)
        return count

    def _run_one_gpu_job(self, now: datetime) -> int:
        if not self._acquire("gpu", self.config.gpu_lease_seconds, now):
            return 0
        try:
            job = self.manifest.lease_job(self.owner, self.config.gpu_lease_seconds)
            if job is None:
                return 0
            succeeded = self.hooks.generate(self.manifest, dict(job))
            with self.manifest.transaction() as conn:
                conn.execute(
                    "UPDATE generation_jobs SET status=?,lease_owner=NULL,"
                    "lease_expires_at=NULL,updated_at=? WHERE job_id=?",
                    ("succeeded" if succeeded else "failed", utc_now(), job["job_id"]),
                )
            return 1
        finally:
            self._release("gpu")

    def _route_qc_and_approvals(self) -> dict[str, int]:
        counts = {"qc": 0, "content": 0, "schedule": 0}
        with self.manifest.connect() as conn:
            generating = conn.execute(
                "SELECT p.post_id FROM posts p WHERE p.state='generating' "
                "AND EXISTS(SELECT 1 FROM generation_jobs j WHERE j.post_id=p.post_id "
                "AND j.status='succeeded') "
                "AND NOT EXISTS(SELECT 1 FROM generation_jobs j WHERE j.post_id=p.post_id "
                "AND j.status IN ('queued','leased'))"
            ).fetchall()
        for row in generating:
            post_id = str(row["post_id"])
            self.manifest.transition(
                post_id,
                "assets_ready" if self.hooks.qc(self.manifest, post_id) else "needs_review",
                self.owner,
            )
            counts["qc"] += 1

        with self.manifest.connect() as conn:
            assets_ready = conn.execute(
                "SELECT post_id FROM posts WHERE state='assets_ready'"
            ).fetchall()
            content_approved = conn.execute(
                "SELECT post_id FROM posts WHERE state='content_approved'"
            ).fetchall()
        for row in assets_ready:
            post_id = str(row["post_id"])
            if self._approval_valid(post_id, "content"):
                self.manifest.transition(post_id, "content_approved", self.owner)
                counts["content"] += 1
        for row in content_approved:
            post_id = str(row["post_id"])
            if self._approval_valid(post_id, "schedule"):
                self.manifest.transition(post_id, "schedule_approved", self.owner)
                counts["schedule"] += 1
        return counts

    def _route_packages_and_publish(self, now: datetime) -> dict[str, int]:
        counts = {"packaged": 0, "published": 0, "manual": 0}
        with self.manifest.connect() as conn:
            approved = conn.execute(
                "SELECT post_id FROM posts WHERE state='schedule_approved'"
            ).fetchall()
        for row in approved:
            post_id = str(row["post_id"])
            if not (
                self._approval_valid(post_id, "content")
                and self._approval_valid(post_id, "schedule")
            ):
                continue
            if self.hooks.package(self.manifest, post_id):
                self.manifest.transition(post_id, "packaged", self.owner)
                counts["packaged"] += 1

        due = due_posts(
            self.manifest,
            now,
            max_lateness=timedelta(hours=self.config.max_publish_lateness_hours),
        )
        for post in due:
            post_id = str(post["post_id"])
            if post["state"] != "packaged":
                continue
            if not (
                self._approval_valid(post_id, "content")
                and self._approval_valid(post_id, "schedule")
            ):
                continue
            if post["platform"] == "patreon":
                counts["manual"] += 1
                continue
            if not self.config.dry_run and not self.config.graph_gates.ready:
                counts["manual"] += 1
                continue
            self.manifest.transition(post_id, "publishing", self.owner)
            publish = (
                self.hooks.simulate_publish
                if self.config.dry_run else self.hooks.publish_graph
            )
            if publish(self.manifest, post_id):
                self.manifest.transition(post_id, "published", self.owner)
                counts["published"] += 1
            else:
                self.manifest.transition(post_id, "needs_reconciliation", self.owner)
        return counts

    def run_once(self, now: datetime | None = None) -> dict[str, object]:
        now = (now or datetime.now(UTC)).astimezone(UTC)
        if not self._acquire("heartbeat", self.config.lease_seconds, now):
            self.log.emit("heartbeat_skipped", reason="single_instance_lease")
            return {"status": "lease_busy"}
        try:
            result: dict[str, object] = {
                "status": "ok",
                "jobs_created": (
                    self._derive_due_generation(now)
                    if self.config.generation_enabled else 0
                ),
                "gpu_jobs": (
                    self._run_one_gpu_job(now)
                    if self.config.generation_enabled else 0
                ),
            }
            result.update(self._route_qc_and_approvals())
            result.update(self._route_packages_and_publish(now))
            self.log.emit("heartbeat_complete", **result)
            return result
        except Exception as exc:
            self.log.emit(
                "heartbeat_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        finally:
            self._release("heartbeat")
