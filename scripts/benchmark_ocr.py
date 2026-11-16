#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import resource
import statistics
import time
from pathlib import Path

from spacemit_vision import VisionServiceNative, VisionServiceStatus


def result_json(item: object) -> dict:
    polygon = getattr(item, "polygon", []) or []
    return {
        "text": str(getattr(item, "text", "") or ""),
        "score": round(float(getattr(item, "score", 0.0) or 0.0), 4),
        "polygon": [
            [round(float(getattr(point, "x", 0.0)), 2), round(float(getattr(point, "y", 0.0)), 2)]
            for point in polygon
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark K3 PP-OCRv5 on real images")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--images", required=True, help="Image glob")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    images = [Path(path) for path in sorted(glob.glob(args.images))]
    if not images:
        raise ValueError(f"No images matched: {args.images}")

    load_started = time.monotonic()
    service = VisionServiceNative.create(str(args.config))
    load_seconds = time.monotonic() - load_started
    rows = []
    for index, image in enumerate(images):
        started = time.monotonic()
        try:
            status, results = service.infer_image(str(image))
            elapsed = time.monotonic() - started
            if status != VisionServiceStatus.OK:
                raise RuntimeError(service.last_error())
            lines = [result_json(item) for item in results]
            rows.append(
                {
                    "index": index,
                    "image": str(image),
                    "elapsed_seconds": round(elapsed, 4),
                    "line_count": len(lines),
                    "mean_score": round(
                        statistics.mean(line["score"] for line in lines), 4
                    )
                    if lines
                    else None,
                    "text": "\n".join(line["text"] for line in lines),
                    "lines": lines,
                }
            )
        except Exception as exc:
            elapsed = time.monotonic() - started
            rows.append(
                {
                    "index": index,
                    "image": str(image),
                    "elapsed_seconds": round(elapsed, 4),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        print(
            f"[{index + 1}/{len(images)}] {rows[-1].get('elapsed_seconds')}s "
            f"lines={rows[-1].get('line_count', 0)} {rows[-1].get('error', '')}",
            flush=True,
        )

    successful = [row for row in rows if not row.get("error")]
    payload = {
        "config": str(args.config),
        "requested_images": len(images),
        "successful_images": len(successful),
        "failure_rate": round(1 - len(successful) / len(images), 4),
        "load_seconds": round(load_seconds, 4),
        "mean_inference_seconds": round(
            statistics.mean(row["elapsed_seconds"] for row in successful), 4
        )
        if successful
        else None,
        "max_inference_seconds": round(
            max(row["elapsed_seconds"] for row in successful), 4
        )
        if successful
        else None,
        "empty_image_count": sum(not row.get("lines") for row in successful),
        "peak_process_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}, ensure_ascii=False))
    return 0 if len(successful) == len(images) else 1


if __name__ == "__main__":
    raise SystemExit(main())
