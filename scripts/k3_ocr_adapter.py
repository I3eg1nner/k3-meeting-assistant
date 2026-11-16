#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from spacemit_vision import VisionServiceNative, VisionServiceStatus


def main() -> int:
    parser = argparse.ArgumentParser(description="K3 PP-OCRv5 command adapter")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.getenv("K3_OCR_CONFIG", "/var/lib/meeting-assistant/models/ocr/ppocr.yaml")),
    )
    args = parser.parse_args()

    loaded = time.monotonic()
    service = VisionServiceNative.create(str(args.config))
    load_seconds = time.monotonic() - loaded
    started = time.monotonic()
    status, results = service.infer_image(str(args.input))
    if status != VisionServiceStatus.OK:
        raise RuntimeError(service.last_error())
    inference_seconds = time.monotonic() - started

    regions = []
    for result in results:
        polygon = [
            [round(float(point.x), 2), round(float(point.y), 2)]
            for point in (getattr(result, "polygon", []) or [])
        ]
        regions.append(
            {
                "text": str(getattr(result, "text", "") or "").strip(),
                "confidence": round(float(getattr(result, "score", 0.0) or 0.0), 4),
                "polygon": polygon,
            }
        )
    regions.sort(
        key=lambda item: (
            min((point[1] for point in item["polygon"]), default=0),
            min((point[0] for point in item["polygon"]), default=0),
        )
    )
    scores = [region["confidence"] for region in regions if region["text"]]
    payload = {
        "text": "\n".join(region["text"] for region in regions if region["text"]),
        "regions": regions,
        "confidence": round(statistics.mean(scores), 4) if scores else 0.0,
        "load_seconds": round(load_seconds, 4),
        "inference_seconds": round(inference_seconds, 4),
        "source_backend": "PP-OCRv5-mobile-K3",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
