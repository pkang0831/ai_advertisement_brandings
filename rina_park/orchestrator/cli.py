#!/usr/bin/env python3
"""Operator CLI. All mutating actions are explicit and local."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from rina_park.analytics.metrics import MetricsStore
from rina_park.factory.manifest import Manifest

from .calendar import import_seed, reschedule_calendar
from .runner import HeartbeatRunner, RunnerConfig

ROOT = Path(__file__).resolve().parents[1]


def config() -> RunnerConfig:
    return RunnerConfig(
        manifest_db=ROOT / "factory" / "manifest.db",
        asset_root=ROOT / "out",
        log_root=ROOT / "logs",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("heartbeat")
    seed = commands.add_parser("import-seed")
    seed.add_argument("--csv", type=Path, default=ROOT / "content" / "calendar_8_weeks.csv")
    shift = commands.add_parser("reschedule")
    shift.add_argument("start_date", type=date.fromisoformat)
    metrics = commands.add_parser("import-metrics")
    metrics.add_argument("csv", type=Path)
    args = parser.parse_args()

    runtime = config()
    if args.command == "heartbeat":
        print(json.dumps(HeartbeatRunner(runtime).run_once(), sort_keys=True))
        return
    if args.command == "import-metrics":
        count = MetricsStore(ROOT / "analytics" / "metrics.db").import_csv(args.csv)
        print(f"imported {count} metrics row(s)")
        return
    manifest = Manifest(runtime.manifest_db, runtime.asset_root)
    if args.command == "import-seed":
        print(f"imported {import_seed(manifest, args.csv)} calendar row(s)")
    else:
        manifest.migrate()
        print(f"calendar_version={reschedule_calendar(manifest, args.start_date)}")


if __name__ == "__main__":
    main()
