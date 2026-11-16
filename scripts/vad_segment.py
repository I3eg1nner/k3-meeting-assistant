#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import wave
from pathlib import Path
from typing import Any

import numpy as np


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("VAD 输入必须是 16-bit 单声道 WAV")
        sample_rate = wav.getframerate()
        if sample_rate != 16000:
            raise ValueError("VAD 输入必须是 16kHz WAV")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy()
    return samples, sample_rate


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.astype("<i2", copy=False).tobytes())


def speech_intervals(
    samples: np.ndarray,
    sample_rate: int,
    trigger_threshold: float,
    stop_threshold: float,
    redemption_seconds: float,
    pre_roll_seconds: float,
    post_roll_seconds: float,
    min_speech_seconds: float,
) -> tuple[list[dict[str, Any]], list[float]]:
    import spacemit_vad

    config = (
        spacemit_vad.VadConfig.preset("silero")
        .with_trigger_threshold(trigger_threshold)
        .with_stop_threshold(stop_threshold)
        .with_smoothing_window(10)
        .with_min_speech_duration(int(min_speech_seconds * 1000))
    )
    if model_dir := os.getenv("MEETING_VAD_MODEL_DIR"):
        config.model_dir = model_dir
    engine = spacemit_vad.VadEngine(config)
    if not engine.is_initialized:
        raise RuntimeError("Silero VAD 初始化失败")

    window_size = 512
    probabilities: list[float] = []
    intervals: list[dict[str, Any]] = []
    active_start: float | None = None
    last_voice: float | None = None
    active_probs: list[float] = []
    duration = len(samples) / sample_rate

    for offset in range(0, len(samples), window_size):
        frame = samples[offset : offset + window_size]
        if len(frame) < window_size:
            frame = np.pad(frame, (0, window_size - len(frame)))
        result = engine.detect(frame, sample_rate)
        probability = float(result.smoothed_probability)
        probabilities.append(probability)
        time_seconds = offset / sample_rate

        if active_start is None:
            if probability >= trigger_threshold:
                active_start = max(0.0, time_seconds - pre_roll_seconds)
                last_voice = time_seconds
                active_probs = [probability]
            continue

        active_probs.append(probability)
        if probability >= stop_threshold:
            last_voice = time_seconds
        if last_voice is not None and time_seconds - last_voice >= redemption_seconds:
            end = min(duration, last_voice + post_roll_seconds)
            if end - active_start >= min_speech_seconds:
                intervals.append(
                    {
                        "start": round(active_start, 3),
                        "end": round(end, 3),
                        "mean_probability": round(float(np.mean(active_probs)), 4),
                        "max_probability": round(max(active_probs), 4),
                    }
                )
            active_start = None
            last_voice = None
            active_probs = []

    if active_start is not None:
        end = duration if last_voice is None else min(duration, last_voice + post_roll_seconds)
        if end - active_start >= min_speech_seconds:
            intervals.append(
                {
                    "start": round(active_start, 3),
                    "end": round(end, 3),
                    "mean_probability": round(float(np.mean(active_probs)), 4),
                    "max_probability": round(max(active_probs), 4),
                }
            )
    return intervals, probabilities


def split_intervals(
    intervals: list[dict[str, Any]],
    max_seconds: float,
    overlap_seconds: float,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for interval_index, interval in enumerate(intervals):
        start = float(interval["start"])
        end = float(interval["end"])
        cursor = start
        part = 0
        while cursor < end:
            chunk_end = min(end, cursor + max_seconds)
            chunks.append(
                {
                    **interval,
                    "interval_index": interval_index,
                    "part": part,
                    "start": round(cursor, 3),
                    "end": round(chunk_end, 3),
                    "forced_split": chunk_end < end,
                }
            )
            if chunk_end >= end:
                break
            cursor = max(cursor + 0.1, chunk_end - overlap_seconds)
            part += 1
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Segment 16kHz meeting audio with K3 Silero VAD")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trigger-threshold", type=float, default=0.5)
    parser.add_argument("--stop-threshold", type=float, default=0.35)
    parser.add_argument("--redemption-seconds", type=float, default=2.0)
    parser.add_argument("--pre-roll-seconds", type=float, default=0.3)
    parser.add_argument("--post-roll-seconds", type=float, default=0.4)
    parser.add_argument("--min-speech-seconds", type=float, default=0.25)
    parser.add_argument("--max-seconds", type=float, default=25.0)
    parser.add_argument("--overlap-seconds", type=float, default=0.8)
    args = parser.parse_args()

    samples, sample_rate = load_wav(args.input)
    intervals, probabilities = speech_intervals(
        samples,
        sample_rate,
        args.trigger_threshold,
        args.stop_threshold,
        args.redemption_seconds,
        args.pre_roll_seconds,
        args.post_roll_seconds,
        args.min_speech_seconds,
    )
    chunks = split_intervals(intervals, args.max_seconds, args.overlap_seconds)
    output_dir = args.output_dir or args.output.with_suffix("")
    for index, chunk in enumerate(chunks):
        start_sample = max(0, math.floor(float(chunk["start"]) * sample_rate))
        end_sample = min(len(samples), math.ceil(float(chunk["end"]) * sample_rate))
        chunk_path = output_dir / f"chunk-{index:05d}-{start_sample:010d}.wav"
        write_wav(chunk_path, samples[start_sample:end_sample], sample_rate)
        chunk["path"] = str(chunk_path)
        chunk["duration"] = round((end_sample - start_sample) / sample_rate, 3)

    duration = len(samples) / sample_rate
    speech_seconds = sum(float(item["end"]) - float(item["start"]) for item in intervals)
    payload = {
        "input": str(args.input.resolve()),
        "duration_seconds": round(duration, 3),
        "speech_seconds": round(speech_seconds, 3),
        "speech_ratio": round(speech_seconds / duration, 4) if duration else 0,
        "interval_count": len(intervals),
        "chunk_count": len(chunks),
        "thresholds": {
            "trigger": args.trigger_threshold,
            "stop": args.stop_threshold,
            "redemption_seconds": args.redemption_seconds,
            "pre_roll_seconds": args.pre_roll_seconds,
            "post_roll_seconds": args.post_roll_seconds,
        },
        "probability_summary": {
            "mean": round(float(np.mean(probabilities)), 4) if probabilities else None,
            "p95": round(float(np.percentile(probabilities, 95)), 4) if probabilities else None,
            "max": round(max(probabilities), 4) if probabilities else None,
        },
        "intervals": intervals,
        "chunks": chunks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in payload.items() if key not in {"intervals", "chunks"}}, ensure_ascii=False, indent=2))
    return 0 if chunks else 2


if __name__ == "__main__":
    raise SystemExit(main())
