#!/usr/bin/env python3
"""Measure visible key-line recall from structured VLM slide extraction."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from evaluate_ocr_recall import best_similarity


def strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for value in value for item in strings(value)]
    if isinstance(value, dict):
        return [item for value in value.values() for item in strings(value)]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("vlm_result", type=Path)
    parser.add_argument("--ocr-results", type=Path)
    parser.add_argument("--threshold", type=float, default=0.72)
    args = parser.parse_args()
    truth = json.loads(args.ground_truth.read_text())
    results = json.loads(args.vlm_result.read_text()).get("results", [])
    by_image: dict[str, list[str]] = {}
    invalid_images = []
    for row in results:
        key = os.path.basename(row["image"]).split(".")[0]
        if row.get("json_valid") and isinstance(row.get("parsed"), dict):
            by_image[key] = strings(row["parsed"])
        else:
            invalid_images.append(key)
    if args.ocr_results:
        ocr_rows = json.loads(args.ocr_results.read_text()).get("results", [])
        for row in ocr_rows:
            key = os.path.basename(row.get("image", row.get("input", ""))).split(".")[0]
            lines = [str(line.get("text", "")) for line in row.get("lines", [])]
            by_image.setdefault(key, []).extend(lines)
    rows = []
    covered_images = sorted(set(truth) & set(by_image))
    tested_images = sorted(set(truth) & {os.path.basename(row["image"]).split(".")[0] for row in results})
    for image in tested_images:
        for reference in truth[image]:
            score = best_similarity(reference, by_image.get(image, []))
            rows.append(
                {
                    "image": image,
                    "reference": reference,
                    "similarity": round(score, 4),
                    "recalled": score >= args.threshold,
                }
            )
    recalled = sum(row["recalled"] for row in rows)
    payload = {
        "threshold": args.threshold,
        "tested_ground_truth_images": tested_images,
        "valid_ground_truth_images": covered_images,
        "invalid_images": invalid_images,
        "key_lines": len(rows),
        "recalled": recalled,
        "recall": round(recalled / len(rows), 4) if rows else 0,
        "results": rows,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
