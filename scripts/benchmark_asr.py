#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from qwen3_asr_client import transcribe


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, 1):
        current = [row]
        for column, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def normalized_characters(text: str) -> list[str]:
    return list(re.sub(r"[^\w\u4e00-\u9fff]+", "", text.lower()))


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


def error_rate(hypothesis: str, reference: str, unit: str) -> float:
    if unit == "word":
        hyp, ref = normalized_words(hypothesis), normalized_words(reference)
    else:
        hyp, ref = normalized_characters(hypothesis), normalized_characters(reference)
    return edit_distance(hyp, ref) / max(1, len(ref))


def read_manifest(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not value.get("audio"):
            raise ValueError(f"{path}:{line_number} 缺少 audio")
        audio = Path(str(value["audio"]))
        if not audio.is_file():
            raise FileNotFoundError(audio)
        samples.append({**value, "audio": audio})
    if not samples:
        raise ValueError("样本清单为空")
    return samples


def process_rss_mb(pid: int | None) -> int | None:
    if not pid:
        return None
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except FileNotFoundError:
        pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark K3 Qwen3-ASR accuracy and stability")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18085/v1/chat/completions")
    parser.add_argument("--model", default="qwen3-asr")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    samples = read_manifest(args.manifest)
    results: list[dict[str, Any]] = []
    peak_rss_mb = process_rss_mb(args.server_pid) or 0
    for round_index in range(args.rounds):
        for sample_index, sample in enumerate(samples):
            result: dict[str, Any] = {
                "round": round_index + 1,
                "sample": str(sample.get("id") or sample_index + 1),
            }
            try:
                output = transcribe(
                    sample["audio"],
                    args.endpoint,
                    args.model,
                    str(sample.get("language") or "auto"),
                    args.timeout,
                )
                text = str(output["text"]).strip()
                characters = normalized_characters(text)
                if not characters:
                    raise ValueError("ASR returned an empty transcript")
                if float(output["duration_seconds"]) >= 10 and len(characters) / float(output["duration_seconds"]) < 0.5:
                    raise ValueError("ASR transcript is implausibly short for the audio duration")
                lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
                if len(lines) >= 10 and len(set(lines)) / len(lines) < 0.3:
                    raise ValueError("ASR output entered a repetition loop")
                if len(characters) >= 100:
                    four_grams = [tuple(characters[index : index + 4]) for index in range(len(characters) - 3)]
                    if len(set(four_grams)) / len(four_grams) < 0.2:
                        raise ValueError("ASR output entered a character repetition loop")
                result.update(
                    text=text,
                    rtf=output["rtf"],
                    duration_seconds=output["duration_seconds"],
                    processing_seconds=output["processing_seconds"],
                )
                reference = sample.get("reference")
                if isinstance(reference, str) and reference:
                    result["cer"] = round(error_rate(output["text"], reference, "character"), 4)
                    result["wer"] = round(error_rate(output["text"], reference, "word"), 4)
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            results.append(result)
            peak_rss_mb = max(peak_rss_mb, process_rss_mb(args.server_pid) or 0)

    successful = [item for item in results if "error" not in item]
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for item in successful:
        by_sample.setdefault(item["sample"], []).append(item)
    consistency = {
        sample: round(Counter(item["text"] for item in items).most_common(1)[0][1] / len(items), 4)
        for sample, items in by_sample.items()
    }
    rtf_values = [float(item["rtf"]) for item in successful if item.get("rtf") is not None]
    cer_values = [float(item["cer"]) for item in successful if item.get("cer") is not None]
    wer_values = [float(item["wer"]) for item in successful if item.get("wer") is not None]
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "rounds": args.rounds,
        "requested_runs": args.rounds * len(samples),
        "successful_runs": len(successful),
        "failure_rate": round(1 - len(successful) / max(1, len(results)), 4),
        "peak_server_rss_mb": peak_rss_mb or None,
        "rtf_mean": round(statistics.mean(rtf_values), 4) if rtf_values else None,
        "rtf_max": round(max(rtf_values), 4) if rtf_values else None,
        "cer_mean": round(statistics.mean(cer_values), 4) if cer_values else None,
        "wer_mean": round(statistics.mean(wer_values), 4) if wer_values else None,
        "output_consistency_by_sample": consistency,
        "results": results,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded)
    return 0 if len(successful) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
