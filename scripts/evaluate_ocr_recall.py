#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", value.lower())


def distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def best_similarity(reference: str, lines: list[str]) -> float:
    target = normalize(reference)
    candidates = [normalize(line) for line in lines if normalize(line)]
    candidates.extend(
        normalize(" ".join(lines[start : start + width]))
        for width in range(2, min(6, len(lines)) + 1)
        for start in range(len(lines) - width + 1)
    )
    if not target or not candidates:
        return 0.0
    return max(1 - distance(target, value) / max(len(target), len(value)) for value in candidates)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate OCR key-line recall")
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("ocr_result", type=Path)
    parser.add_argument("--threshold", type=float, default=0.72)
    args = parser.parse_args()
    truth = json.loads(args.ground_truth.read_text())
    results = json.loads(args.ocr_result.read_text())["results"]
    by_image = {}
    for row in results:
        key = os.path.basename(row["image"]).split(".")[0]
        lines = row.get("lines") or []
        lines.sort(
            key=lambda line: (
                min((point[1] for point in line.get("polygon", [])), default=0),
                min((point[0] for point in line.get("polygon", [])), default=0),
            )
        )
        by_image[key] = [line.get("text", "") for line in lines]
    rows = []
    for image, references in truth.items():
        for reference in references:
            score = best_similarity(reference, by_image.get(image, []))
            rows.append(
                {
                    "image": image,
                    "reference": reference,
                    "similarity": round(score, 4),
                    "recalled": score >= args.threshold,
                }
            )
    payload = {
        "threshold": args.threshold,
        "key_lines": len(rows),
        "recalled": sum(row["recalled"] for row in rows),
        "recall": round(sum(row["recalled"] for row in rows) / len(rows), 4),
        "results": rows,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
