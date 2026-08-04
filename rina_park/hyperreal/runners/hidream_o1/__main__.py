"""CLI for non-generating readiness inspection."""

from __future__ import annotations

import argparse
import sys

from .readiness import inspect_readiness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("readiness", "download"),
        nargs="?",
        default="readiness",
    )
    args = parser.parse_args()
    report = inspect_readiness()
    print(report.to_json())
    if args.command == "download":
        if not report.can_download:
            print(
                "refused: exact official 2604 Apple load path is not verified",
                file=sys.stderr,
            )
            return 2
        print("download gate passed, but downloader is intentionally absent in this wave")
        return 3
    return 0 if report.can_load else 1


if __name__ == "__main__":
    raise SystemExit(main())
