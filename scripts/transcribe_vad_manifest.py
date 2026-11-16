#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from qwen3_asr_client import transcribe


def atomic_write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe every VAD chunk with resumable checkpoints"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:18085/v1/chat/completions"
    )
    parser.add_argument("--model", default="qwen3-asr")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--server-pid", type=int)
    args = parser.parse_args()

    source = json.loads(args.manifest.read_text())
    chunks = source.get("chunks") or []
    if not chunks:
        raise ValueError("VAD manifest has no chunks")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any]
    if args.output.is_file():
        payload = json.loads(args.output.read_text())
    else:
        payload = {
            "source_manifest": str(args.manifest),
            "source_audio": source.get("input"),
            "results": [],
        }
    completed = {
        int(item["chunk_index"]): item
        for item in payload.get("results", [])
        if "chunk_index" in item and not item.get("error")
    }
    peak_rss_mb = int(payload.get("peak_server_rss_mb") or 0)
    started = time.monotonic()

    def rss_mb() -> int:
        if not args.server_pid:
            return 0
        try:
            for line in Path(f"/proc/{args.server_pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
        except FileNotFoundError:
            pass
        return 0

    for index, chunk in enumerate(chunks):
        if index in completed:
            continue
        result: dict[str, Any] = {
            "chunk_index": index,
            "path": str(chunk["path"]),
            "start": chunk["start"],
            "end": chunk["end"],
            "forced_split": bool(chunk.get("forced_split")),
        }
        try:
            output = transcribe(
                Path(chunk["path"]),
                args.endpoint,
                args.model,
                args.language,
                args.timeout,
            )
            result.update(
                text=output["text"],
                duration_seconds=output["duration_seconds"],
                processing_seconds=output["processing_seconds"],
                rtf=output["rtf"],
            )
            if not output["text"]:
                result["error"] = "empty transcript"
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        payload.setdefault("results", []).append(result)
        peak_rss_mb = max(peak_rss_mb, rss_mb())
        successful = [item for item in payload["results"] if not item.get("error")]
        audio_seconds = sum(float(item.get("duration_seconds") or 0) for item in successful)
        processing_seconds = sum(
            float(item.get("processing_seconds") or 0) for item in successful
        )
        payload.update(
            requested_chunks=len(chunks),
            completed_chunks=len(payload["results"]),
            successful_chunks=len(successful),
            failure_count=len(payload["results"]) - len(successful),
            audio_seconds=round(audio_seconds, 3),
            processing_seconds=round(processing_seconds, 3),
            rtf=round(processing_seconds / audio_seconds, 4) if audio_seconds else None,
            peak_server_rss_mb=peak_rss_mb or None,
            wall_seconds_this_run=round(time.monotonic() - started, 3),
        )
        atomic_write(args.output, payload)
        print(
            f"[{len(payload['results'])}/{len(chunks)}] "
            f"{result.get('rtf', '-')} {result.get('text') or result.get('error')}",
            flush=True,
        )

    failures = [item for item in payload["results"] if item.get("error")]
    return 0 if not failures and len(payload["results"]) >= len(chunks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
