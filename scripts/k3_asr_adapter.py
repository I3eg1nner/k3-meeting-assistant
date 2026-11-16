#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from qwen3_asr_client import transcribe


def mem_available_mb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except FileNotFoundError:
        pass
    return None


def process_rss_mb(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except FileNotFoundError:
        pass
    return None


def assert_memory(pid: int, min_available_mb: int, max_rss_mb: int) -> tuple[int | None, int | None]:
    available = mem_available_mb()
    rss = process_rss_mb(pid)
    if available is not None and available < min_available_mb:
        raise RuntimeError(
            f"MemAvailable {available}MiB is below the safety threshold {min_available_mb}MiB"
        )
    if rss is not None and rss > max_rss_mb:
        raise RuntimeError(f"ASR server RSS {rss}MiB exceeds the limit {max_rss_mb}MiB")
    return available, rss


def health(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def normalized_with_positions(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    positions: list[int] = []
    for index, character in enumerate(text):
        if character.isalnum() or "\u4e00" <= character <= "\u9fff":
            chars.append(character.lower())
            positions.append(index)
    return "".join(chars), positions


def trim_overlap(previous: str, current: str, overlap_seconds: float) -> str:
    if overlap_seconds <= 0 or not previous or not current:
        return current
    left, _ = normalized_with_positions(previous)
    right, positions = normalized_with_positions(current)
    maximum = min(100, len(left), len(right))
    match = 0
    for length in range(maximum, 1, -1):
        if left[-length:] == right[:length]:
            match = length
            break
    if not match:
        return current
    # Two Chinese characters are meaningful; Latin-only overlaps need a longer match.
    matched = right[:match]
    if not any("\u4e00" <= char <= "\u9fff" for char in matched) and match < 5:
        return current
    cut = positions[match - 1] + 1
    return current[cut:].lstrip(" ，,。.!！？?；;：:")


def transcript_issue(text: str, duration: float) -> str | None:
    normalized, _ = normalized_with_positions(text)
    if not normalized:
        return "empty-transcript"
    if duration >= 10 and len(normalized) / duration < 0.5:
        return "implausibly-short-transcript"
    if len(normalized) >= 100:
        four_grams = [normalized[index : index + 4] for index in range(len(normalized) - 3)]
        if len(set(four_grams)) / len(four_grams) < 0.2:
            return "repetition-loop"
    return None


def terminate(process: subprocess.Popen[bytes] | None) -> None:
    if not process or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="K3 Silero VAD + Qwen3-ASR adapter")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.getenv("K3_ASR_MODEL_DIR", "/var/lib/meeting-assistant/models/asr/qwen3-asr-0.6B-dynq-q40")),
    )
    parser.add_argument("--port", type=int, default=18085)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--llama-server", default=os.getenv("LLAMA_SERVER_BIN", "llama-server"))
    parser.add_argument(
        "--min-mem-available-mb",
        type=int,
        default=int(os.getenv("MIN_MEM_AVAILABLE_MB", "512")),
    )
    parser.add_argument(
        "--max-model-rss-mb",
        type=int,
        default=int(os.getenv("MAX_MODEL_RSS_MB", "6349")),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    args.output.parent.mkdir(parents=True, exist_ok=True)
    vad_manifest = args.output.with_suffix(".vad.json")
    chunks_dir = args.output.with_suffix(".vad-chunks")
    subprocess.run(
        [
            sys.executable,
            str(script_dir / "vad_segment.py"),
            "--input", str(args.input),
            "--output", str(vad_manifest),
            "--output-dir", str(chunks_dir),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    vad = json.loads(vad_manifest.read_text())
    text_model = next(args.model_dir.glob("*text*.gguf"), None)
    if not text_model:
        raise FileNotFoundError(f"ASR text GGUF not found in {args.model_dir}")

    endpoint = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    log_path = args.output.with_suffix(".server.log")
    log = log_path.open("ab")
    process: subprocess.Popen[bytes] | None = None
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            [
                args.llama_server, "-m", str(text_model),
                "--media-backend", "smt", "--smt-config-dir", str(args.model_dir),
                "--host", "127.0.0.1", "--port", str(args.port),
                "--alias", "qwen3-asr", "-t", "8", "-c", "4096",
                "--parallel", "1", "--cache-ram", "0",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not health(f"http://127.0.0.1:{args.port}/health"):
            if process.poll() is not None:
                raise RuntimeError(f"Qwen3-ASR server exited; see {log_path}")
            assert_memory(process.pid, args.min_mem_available_mb, args.max_model_rss_mb)
            time.sleep(0.5)
        if not health(f"http://127.0.0.1:{args.port}/health"):
            raise TimeoutError("Qwen3-ASR server did not become healthy")

        segments = []
        failed_segments = []
        ignored_non_speech = []
        previous_text = ""
        total_processing = 0.0
        peak_rss_mb = 0
        min_mem_available_mb: int | None = None
        for index, chunk in enumerate(vad["chunks"]):
            available, rss = assert_memory(process.pid, args.min_mem_available_mb, args.max_model_rss_mb)
            output = transcribe(Path(chunk["path"]), endpoint, "qwen3-asr", args.language, 600)
            raw_text = output["text"].strip()
            total_processing += float(output["processing_seconds"])
            available, rss = assert_memory(process.pid, args.min_mem_available_mb, args.max_model_rss_mb)
            if available is not None:
                min_mem_available_mb = (
                    available if min_mem_available_mb is None else min(min_mem_available_mb, available)
                )
            if rss is not None:
                peak_rss_mb = max(peak_rss_mb, rss)
            issue = transcript_issue(raw_text, float(chunk.get("duration") or 0))
            if issue:
                record = {
                    "chunk_index": index,
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "mean_probability": chunk.get("mean_probability"),
                    "reason": issue,
                }
                if issue == "empty-transcript" and float(chunk.get("duration") or 0) <= 2.0 and float(chunk.get("mean_probability") or 0) < 0.3:
                    ignored_non_speech.append(record)
                else:
                    failed_segments.append(record)
                continue
            overlap = max(0.0, float(segments[-1]["end"]) - float(chunk["start"])) if segments else 0.0
            corrected = trim_overlap(previous_text, raw_text, overlap)
            if not corrected:
                ignored_non_speech.append(
                    {"chunk_index": index, "start": chunk["start"], "end": chunk["end"], "reason": "overlap-only"}
                )
                continue
            segments.append(
                {
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "raw_text": raw_text,
                    "text": corrected,
                    "confidence": 0.75,
                    "chunk_index": index,
                }
            )
            previous_text = raw_text
        audio_seconds = sum(float(chunk.get("duration") or 0) for chunk in vad["chunks"])
        payload = {
            "text": "\n".join(segment["text"] for segment in segments),
            "segments": segments,
            "failed_segments": failed_segments,
            "ignored_non_speech": ignored_non_speech,
            "vad": {key: vad.get(key) for key in ("duration_seconds", "speech_seconds", "speech_ratio", "interval_count", "chunk_count")},
            "metrics": {
                "audio_seconds": round(audio_seconds, 3),
                "processing_seconds": round(total_processing, 3),
                "rtf": round(total_processing / audio_seconds, 4) if audio_seconds else None,
                "wall_seconds": round(time.monotonic() - started, 3),
                "peak_server_rss_mb": peak_rss_mb,
                "min_mem_available_mb": min_mem_available_mb,
            },
        }
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        temporary.replace(args.output)
        print(json.dumps({key: value for key, value in payload.items() if key not in {"text", "segments"}}, ensure_ascii=False))
        return 0
    finally:
        terminate(process)
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
