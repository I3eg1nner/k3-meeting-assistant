#!/usr/bin/env python3
"""K3 offline diarization: Silero VAD -> CAM++ ONNX -> spectral clustering."""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vad_segment import load_wav, speech_intervals, write_wav  # noqa: E402


def subsegments(intervals: list[dict[str, Any]], duration: float = 1.5, step: float = 0.75) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for interval in intervals:
        start, end = float(interval["start"]), float(interval["end"])
        cursor = start
        while cursor + duration < end + step:
            result.append((round(cursor, 3), round(min(cursor + duration, end), 3)))
            cursor += step
    return result


def kmeans(x: np.ndarray, k: int, iterations: int = 100) -> np.ndarray:
    """Small deterministic k-means implementation for K3 (no sklearn dependency)."""
    rng = np.random.default_rng(7)
    centers = [x[rng.integers(0, len(x))]]
    for _ in range(1, k):
        distance = np.min(
            np.stack([np.sum((x - center) ** 2, axis=1) for center in centers]), axis=0
        )
        if float(distance.sum()) == 0:
            centers.append(x[rng.integers(0, len(x))])
        else:
            centers.append(x[rng.choice(len(x), p=distance / distance.sum())])
    centers_array = np.stack(centers)
    labels = np.zeros(len(x), dtype=np.int32)
    for _ in range(iterations):
        next_labels = np.argmin(
            np.stack([np.sum((x - center) ** 2, axis=1) for center in centers_array]), axis=0
        ).astype(np.int32)
        if np.array_equal(next_labels, labels):
            break
        labels = next_labels
        for index in range(k):
            if np.any(labels == index):
                centers_array[index] = x[labels == index].mean(axis=0)
    return labels


def spectral_cluster(embeddings: np.ndarray, speakers: int | None, minimum: int, maximum: int) -> tuple[np.ndarray, int]:
    if len(embeddings) == 1:
        return np.zeros(1, dtype=np.int32), 1
    normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    affinity = normalized @ normalized.T
    keep = max(6, int(math.ceil(len(affinity) * 0.02)))
    keep = min(len(affinity), keep)
    pruned = np.zeros_like(affinity)
    for row in range(len(affinity)):
        indices = np.argpartition(affinity[row], -keep)[-keep:]
        pruned[row, indices] = affinity[row, indices]
    affinity = (pruned + pruned.T) * 0.5
    np.fill_diagonal(affinity, 0)
    laplacian = np.diag(np.sum(np.abs(affinity), axis=1)) - affinity
    upper = min(maximum + 1, len(embeddings) - 1)
    if upper < 2:
        return np.zeros(len(embeddings), dtype=np.int32), 1
    values, vectors = eigsh(laplacian, k=upper, which="SM")
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    if speakers is None:
        lo = max(0, minimum - 1)
        hi = min(len(values) - 1, maximum)
        gaps = np.diff(values[lo : hi + 1])
        speakers = int(np.argmax(gaps) + minimum) if len(gaps) else minimum
    speakers = max(1, min(int(speakers), len(embeddings)))
    return kmeans(vectors[:, :speakers], speakers), speakers


def refine_clusters(
    labels: np.ndarray,
    embeddings: np.ndarray,
    min_cluster_size: int = 4,
    merge_cosine: float = 0.8,
) -> np.ndarray:
    """Apply the minor-cluster removal and centroid merge used by 3D-Speaker."""
    labels = labels.copy()
    clusters, sizes = np.unique(labels, return_counts=True)
    minor = clusters[sizes <= min_cluster_size]
    major = clusters[sizes > min_cluster_size]
    if len(minor):
        if not len(major):
            labels[:] = 0
        else:
            centers = np.stack([embeddings[labels == cluster].mean(axis=0) for cluster in major])
            centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
            normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
            for index in np.flatnonzero(np.isin(labels, minor)):
                labels[index] = int(major[np.argmax(normalized[index] @ centers.T)])
    while True:
        clusters = np.unique(labels)
        if len(clusters) <= 1:
            break
        centers = np.stack([embeddings[labels == cluster].mean(axis=0) for cluster in clusters])
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
        affinity = np.triu(centers @ centers.T, 1)
        left, right = np.unravel_index(np.argmax(affinity), affinity.shape)
        if float(affinity[left, right]) < merge_cosine:
            break
        labels[labels == clusters[right]] = clusters[left]
    return labels


def frame_vote(
    windows: list[tuple[float, float]], labels: np.ndarray, intervals: list[dict[str, Any]], frame: float = 0.05
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for interval in intervals:
        start, end = float(interval["start"]), float(interval["end"])
        count = max(1, int(math.ceil((end - start) / frame)))
        votes: list[list[int]] = [[] for _ in range(count)]
        for (win_start, win_end), label in zip(windows, labels):
            if win_end <= start or win_start >= end:
                continue
            left = max(0, int((max(start, win_start) - start) / frame))
            right = min(count, int(math.ceil((min(end, win_end) - start) / frame)))
            for index in range(left, right):
                votes[index].append(int(label))
        timeline: list[int] = []
        previous = 0
        for values in votes:
            if values:
                counts = np.bincount(values)
                previous = int(np.argmax(counts))
            timeline.append(previous)
        turn_start = start
        current = timeline[0]
        for index, label in enumerate(timeline[1:], 1):
            if label != current:
                boundary = min(end, start + index * frame)
                output.append({"start": round(turn_start, 3), "end": round(boundary, 3), "speaker": f"Speaker {current + 1}"})
                turn_start, current = boundary, label
        output.append({"start": round(turn_start, 3), "end": round(end, 3), "speaker": f"Speaker {current + 1}"})
    return output


def read_rttm(path: Path) -> list[dict[str, Any]]:
    result = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[0] == "SPEAKER":
            start, duration = float(fields[3]), float(fields[4])
            result.append({"start": start, "end": start + duration, "speaker": fields[7]})
    return result


def der(reference: list[dict[str, Any]], predicted: list[dict[str, Any]], frame: float = 0.01) -> dict[str, float]:
    end = max([float(x["end"]) for x in reference + predicted], default=0)
    frames = int(math.ceil(end / frame))
    ref_names = sorted({str(x["speaker"]) for x in reference})
    hyp_names = sorted({str(x["speaker"]) for x in predicted})
    ref = np.full(frames, -1, dtype=np.int32)
    hyp = np.full(frames, -1, dtype=np.int32)
    for item in reference:
        ref[int(float(item["start"]) / frame) : int(math.ceil(float(item["end"]) / frame))] = ref_names.index(str(item["speaker"]))
    for item in predicted:
        hyp[int(float(item["start"]) / frame) : int(math.ceil(float(item["end"]) / frame))] = hyp_names.index(str(item["speaker"]))
    overlap = np.zeros((len(ref_names), len(hyp_names)), dtype=np.int64)
    for i in range(len(ref_names)):
        for j in range(len(hyp_names)):
            overlap[i, j] = int(np.sum((ref == i) & (hyp == j)))
    mapping: dict[int, int] = {}
    if overlap.size:
        rows, cols = linear_sum_assignment(-overlap)
        mapping = {int(col): int(row) for row, col in zip(rows, cols)}
    mapped = np.array([mapping.get(int(value), -2) if value >= 0 else -1 for value in hyp], dtype=np.int32)
    ref_speech = ref >= 0
    hyp_speech = hyp >= 0
    denominator = max(1, int(ref_speech.sum()))
    miss = int(np.sum(ref_speech & ~hyp_speech))
    false_alarm = int(np.sum(~ref_speech & hyp_speech))
    confusion = int(np.sum(ref_speech & hyp_speech & (ref != mapped)))
    return {
        "der": round((miss + false_alarm + confusion) / denominator, 4),
        "miss": round(miss / denominator, 4),
        "false_alarm": round(false_alarm / denominator, 4),
        "confusion": round(confusion / denominator, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path("/var/lib/meeting-assistant/models/campplus_cn_common.onnx"))
    parser.add_argument("--runtime", type=Path, default=Path("/opt/meeting-assistant/vendor/onnxruntime/build/bin/extract_speaker_embedding"))
    parser.add_argument("--fbank-config", type=Path, default=Path("/opt/meeting-assistant/vendor/onnxruntime/assets/fbank_config.json"))
    parser.add_argument("--speakers", type=int)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=8)
    parser.add_argument("--reference-rttm", type=Path)
    parser.add_argument("--window-duration", type=float, default=1.5)
    parser.add_argument("--window-step", type=float, default=1.5)
    parser.add_argument("--min-cluster-size", type=int, default=4)
    parser.add_argument("--merge-cosine", type=float, default=0.8)
    args = parser.parse_args()

    started = time.monotonic()
    samples, sample_rate = load_wav(args.input)
    intervals, _ = speech_intervals(samples, sample_rate, 0.5, 0.35, 2.0, 0.3, 0.4, 0.25)
    windows = subsegments(intervals, args.window_duration, args.window_step)
    if not windows:
        raise RuntimeError("VAD did not find speech")
    with tempfile.TemporaryDirectory(prefix="meeting-diar-") as temporary:
        root = Path(temporary)
        clips = root / "clips"
        embeddings_dir = root / "embeddings"
        clips.mkdir()
        embeddings_dir.mkdir()
        scp_lines = []
        for index, (start, end) in enumerate(windows):
            left, right = int(start * sample_rate), int(math.ceil(end * sample_rate))
            audio = samples[left:right]
            target = int(args.window_duration * sample_rate)
            if len(audio) < target:
                audio = np.resize(audio, target)
            path = clips / f"seg-{index:06d}.wav"
            write_wav(path, audio, sample_rate)
            scp_lines.append(f"seg-{index:06d} {path}")
        wav_scp = root / "wav.scp"
        embedding_scp = root / "embedding.scp"
        wav_scp.write_text("\n".join(scp_lines) + "\n")
        subprocess.run(
            [str(args.runtime), str(args.fbank_config), str(args.model), str(wav_scp), str(embedding_scp), str(embeddings_dir)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        by_id = {}
        for line in embedding_scp.read_text().splitlines():
            identifier, path = line.split(maxsplit=1)
            by_id[identifier] = np.loadtxt(path, dtype=np.float32)
        embeddings = np.stack([by_id[f"seg-{index:06d}"] for index in range(len(windows))])
    labels, _estimated_speakers = spectral_cluster(embeddings, args.speakers, args.min_speakers, args.max_speakers)
    labels = refine_clusters(labels, embeddings, args.min_cluster_size, args.merge_cosine)
    active_labels = sorted(int(value) for value in np.unique(labels))
    label_map = {label: index for index, label in enumerate(active_labels)}
    labels = np.array([label_map[int(value)] for value in labels], dtype=np.int32)
    speaker_count = len(active_labels)
    segments = frame_vote(windows, labels, intervals)
    elapsed = time.monotonic() - started
    metrics: dict[str, Any] = {
        "duration_seconds": round(len(samples) / sample_rate, 3),
        "speech_seconds": round(sum(float(x["end"]) - float(x["start"]) for x in intervals), 3),
        "vad_intervals": len(intervals),
        "embedding_windows": len(windows),
        "speakers": speaker_count,
        "elapsed_seconds": round(elapsed, 3),
        "rtf": round(elapsed / (len(samples) / sample_rate), 4),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    if args.reference_rttm:
        metrics.update(der(read_rttm(args.reference_rttm), segments))
    payload = {"segments": segments, "metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
