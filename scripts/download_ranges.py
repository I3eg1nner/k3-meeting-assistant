#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def remote_size(url: str, attempts: int = 5, timeout: int = 60) -> int:
    """Read range metadata with bounded retries for unstable board networks."""
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                size = int(response.headers["Content-Length"])
                if "bytes" not in response.headers.get("Accept-Ranges", "").lower():
                    raise RuntimeError("server does not advertise byte ranges")
                return size
        except RuntimeError:
            raise
        except (OSError, urllib.error.URLError, KeyError, ValueError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(15, 2 ** attempt))
    raise RuntimeError(f"failed to read remote size after {attempts} attempts: {last_error}") from last_error


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable parallel HTTP range downloader")
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--parts", type=int, default=12)
    parser.add_argument("--metadata-attempts", type=int, default=5)
    parser.add_argument("--metadata-timeout", type=int, default=60)
    args = parser.parse_args()

    if args.metadata_attempts < 1 or args.metadata_timeout < 1:
        parser.error("metadata attempts and timeout must be positive")
    size = remote_size(args.url, args.metadata_attempts, args.metadata_timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    part_dir = args.output.with_suffix(args.output.suffix + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    width = (size + args.parts - 1) // args.parts

    def download(index: int) -> Path:
        start = index * width
        end = min(size - 1, start + width - 1)
        expected = end - start + 1
        path = part_dir / f"part-{index:03d}"
        if path.is_file() and path.stat().st_size == expected:
            return path
        present = path.stat().st_size if path.is_file() else 0
        if present > expected:
            path.unlink()
            present = 0
        tail = part_dir / f"part-{index:03d}.resume"
        tail.unlink(missing_ok=True)
        subprocess.run(
            [
                "curl", "-fL", "--retry", "30", "--retry-all-errors",
                "--retry-delay", "2", "--silent", "--show-error",
                "--range", f"{start + present}-{end}", "--output", str(tail), args.url,
            ],
            check=True,
        )
        if tail.stat().st_size != expected - present:
            raise RuntimeError(
                f"part {index} resume has {tail.stat().st_size} bytes, expected {expected - present}"
            )
        with path.open("ab") as destination, tail.open("rb") as source:
            while block := source.read(8 * 1024 * 1024):
                destination.write(block)
            destination.flush()
            os.fsync(destination.fileno())
        tail.unlink()
        if path.stat().st_size != expected:
            raise RuntimeError(
                f"part {index} has {path.stat().st_size} bytes, expected {expected}"
            )
        print(f"part {index + 1}/{args.parts} complete", flush=True)
        return path

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parts) as pool:
        parts = list(pool.map(download, range(args.parts)))
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as destination:
        for part in parts:
            with part.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    destination.write(block)
        destination.flush()
        os.fsync(destination.fileno())
    if temporary.stat().st_size != size:
        raise RuntimeError("assembled file size mismatch")
    temporary.replace(args.output)
    print(f"downloaded {size} bytes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
