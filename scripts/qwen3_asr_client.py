#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
import wave
from pathlib import Path
from typing import Any


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getframerate() != 16000:
            raise ValueError("Qwen3-ASR 输入必须是 16kHz 单声道 WAV")
        return wav.getnframes() / wav.getframerate()


def response_text(response: dict[str, Any]) -> str:
    content = response["choices"][0]["message"]["content"]
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            str(item.get("text") or "") for item in content if isinstance(item, dict)
        )
    else:
        text = str(content or "")
    text = text.strip()
    for tag in ("<asr_text>", "</asr_text>"):
        text = text.replace(tag, "")
    return text.strip()


def transcribe(path: Path, endpoint: str, model: str, language: str, timeout: int) -> dict[str, Any]:
    duration = wav_duration(path)
    encoded = base64.b64encode(path.read_bytes()).decode()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded, "format": "wav"},
                    },
                    {"type": "text", "text": f"language {language}<asr_text>"},
                ],
            }
        ],
        "max_tokens": 512,
        "temperature": 0,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    elapsed = time.monotonic() - started
    text = response_text(result)
    return {
        "text": text,
        "duration_seconds": round(duration, 3),
        "processing_seconds": round(elapsed, 3),
        "rtf": round(elapsed / duration, 4) if duration else None,
        "timings": result.get("timings") or {},
        "segments": [
            {
                "start": 0.0,
                "end": round(duration, 3),
                "text": text,
                "confidence": 0.75,
            }
        ] if text else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Call a K3 Qwen3-ASR llama-server")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:18085/v1/chat/completions",
    )
    parser.add_argument("--model", default="qwen3-asr")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    result = transcribe(args.input, args.endpoint, args.model, args.language, args.timeout)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded)
    return 0 if result["segments"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
