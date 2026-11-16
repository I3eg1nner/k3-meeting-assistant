"""Production startup and authenticated readiness checks; never print secrets."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import time
import urllib.error
import urllib.request
from pathlib import Path


def validate_config(environ: dict[str, str]) -> None:
    if not environ.get("MEETING_API_TOKEN", "").strip():
        raise RuntimeError("MEETING_API_TOKEN is required for the LAN service")
    for name in ("MEETING_DATA_DIR", "MEETING_STATIC_DIR"):
        value = environ.get(name, "")
        if not value or not Path(value).is_absolute():
            raise RuntimeError(f"{name} must be an absolute path")


def read_token(path: Path) -> str:
    token = ""
    for line in path.read_text().splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and key == "MEETING_API_TOKEN":
            parts = shlex.split(value, comments=False)
            token = parts[0] if len(parts) == 1 else ""
    if not token.strip():
        raise RuntimeError("Production API token is missing")
    return token


def check_health(token: str, url: str = "http://127.0.0.1:8000/api/health") -> None:
    try:
        with urllib.request.urlopen(url, timeout=2):
            raise RuntimeError("Health endpoint permits unauthenticated access")
    except urllib.error.HTTPError as exc:
        if exc.code != 401:
            raise RuntimeError("Unexpected unauthenticated health response") from exc
    request = urllib.request.Request(url, headers={"X-Meeting-Token": token})
    with urllib.request.urlopen(request, timeout=2) as response:
        if response.status != 200 or json.load(response).get("status") != "ok":
            raise RuntimeError("Authenticated health check failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args()
    if args.check_config:
        validate_config(dict(os.environ))
    if args.health:
        token = read_token(Path("/etc/meeting-assistant.env"))
        for attempt in range(30):
            try:
                check_health(token)
                print("Service readiness and authentication checks passed")
                return
            except (OSError, RuntimeError, ValueError):
                if attempt == 29:
                    raise RuntimeError("Service readiness/authentication check failed after 60 seconds") from None
                time.sleep(2)


if __name__ == "__main__":
    main()
