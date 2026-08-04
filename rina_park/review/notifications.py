from __future__ import annotations

import subprocess
import webbrowser
from urllib.parse import urlparse


def _escape_applescript(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def notify_review_ready(
    message: str,
    *,
    title: str = "Rina review ready",
    review_url: str | None = None,
    open_browser: bool = False,
) -> None:
    """Send a GUI-session notification without invoking a shell."""
    script = (
        f'display notification "{_escape_applescript(message)}" '
        f'with title "{_escape_applescript(title)}"'
    )
    subprocess.run(
        ["/usr/bin/osascript", "-e", script],
        check=True,
        timeout=10,
        capture_output=True,
        text=True,
    )
    if review_url and open_browser:
        parsed = urlparse(review_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("Only a local HTTP review URL may be opened")
        webbrowser.open(review_url)
