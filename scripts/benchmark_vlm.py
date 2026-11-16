#!/usr/bin/env python3
"""Benchmark a K3 SMT vision model on real meeting images.

The benchmark deliberately records raw responses as well as parsed JSON so a
malformed answer can never be silently counted as a successful extraction.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


def memory(pid: int) -> tuple[int, int]:
    rss = 0
    available = 0
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) // 1024
                break
    except OSError:
        pass
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) // 1024
            break
    return rss, available


def post_json(url: str, payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def parse_object(content: str) -> dict[str, Any]:
    value = content.strip()
    if value.startswith("```"):
        value = "\n".join(value.splitlines()[1:-1]).strip()
        if value.startswith("json"):
            value = value[4:].lstrip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    return parsed


def validate_semantics(parsed: dict[str, Any]) -> dict[str, Any]:
    """Mirror the application-side evidence gate for generative image output."""
    value = dict(parsed)
    metrics = value.get("metrics") or []
    if isinstance(metrics, list):
        value["metrics"] = [
            str(item) for item in metrics[:5]
            if item and re.search(
                r"\d|[%％]|(?:GB|MB|GHz|MHz|fps|帧|秒|分钟|小时|万元|亿元|台)",
                str(item), re.I,
            )
        ]
    unreadable = value.get("unreadable") or []
    if isinstance(unreadable, list):
        value["unreadable"] = [
            str(item) for item in unreadable[:3]
            if item and not re.search(r"人物|人脸|上半身|下半身|观众|会场|背景", str(item))
        ]
    chart = str(value.get("chart_or_table") or "").strip()
    if chart.lower() in {"text", "none", "n/a", "无", "无图表", "无表格"}:
        value["chart_or_table"] = ""
    return value


def wait_ready(url: str, process: subprocess.Popen[bytes], deadline: float) -> None:
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with {process.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/models", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError("llama-server did not become ready")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--server", default="llama-server")
    parser.add_argument("--port", type=int, default=18086)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--ocr-results", type=Path)
    parser.add_argument("--max-rss-mb", type=int, default=6349)
    parser.add_argument("--min-mem-available-mb", type=int, default=512)
    args = parser.parse_args()

    images = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )[: args.limit]
    if not images:
        raise RuntimeError("no benchmark images found")
    ocr_by_name: dict[str, str] = {}
    if args.ocr_results:
        data = json.loads(args.ocr_results.read_text())
        for item in data.get("results", data if isinstance(data, list) else []):
            ocr_by_name[Path(item.get("input", item.get("image", ""))).name] = str(item.get("text", ""))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".server.log")
    log = log_path.open("wb")
    base_url = f"http://127.0.0.1:{args.port}/v1"
    command = [
        args.server, "-m", str(args.model), "--host", "127.0.0.1", "--port", str(args.port),
        "--ctx-size", "4096", "--parallel", "1", "--cache-ram", "0", "--reasoning-budget", "0",
        "--batch-size", "512", "--ubatch-size", "128",
        "--threads", "8", "--metrics", "--jinja", "--flash-attn", "on", "--no-warmup",
        "--vision-backend", "smt", "--smt-config-dir", str(args.model_dir),
    ]
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    peak_rss = 0
    min_available = 1 << 30
    resource_violation: list[str] = []
    monitor_stop = threading.Event()

    def monitor_resources() -> None:
        nonlocal peak_rss, min_available
        while not monitor_stop.wait(0.1) and process.poll() is None:
            rss, available = memory(process.pid)
            peak_rss = max(peak_rss, rss)
            min_available = min(min_available, available)
            if rss > args.max_rss_mb:
                resource_violation.append(f"RSS {rss}MiB exceeds {args.max_rss_mb}MiB")
            elif available < args.min_mem_available_mb:
                resource_violation.append(
                    f"MemAvailable {available}MiB is below {args.min_mem_available_mb}MiB"
                )
            if resource_violation:
                args.output.with_suffix(".resource-violation.json").write_text(
                    json.dumps(
                        {
                            "error": resource_violation[0],
                            "peak_rss_mb": peak_rss,
                            "min_mem_available_mb": min_available,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                return

    monitor = threading.Thread(target=monitor_resources, daemon=True)
    monitor.start()
    results: list[dict[str, Any]] = []
    try:
        wait_ready(base_url, process, time.monotonic() + 90)
        loaded = time.monotonic() - started
        for index, image in enumerate(images):
            encoded = base64.b64encode(image.read_bytes()).decode()
            mime = "jpeg" if image.suffix.lower() in {".jpg", ".jpeg"} else image.suffix.lower()[1:]
            instruction = (
                "核对这张会议PPT照片。只输出JSON对象，字段为title字符串、key_points字符串数组、"
                "metrics字符串数组、chart_or_table字符串、unreadable字符串数组。只提取图中直接可见信息；"
                "模糊、遮挡或无法确认的幻灯片文字/图表区域必须放入unreadable，不得猜测。"
                "metrics只能逐字记录带阿拉伯数字、百分号或明确单位的可见指标，无数字指标时必须返回[]；"
                "chart_or_table只概括确实可见的图表或表格，没有则返回空字符串；unreadable不要描述人物、会场或背景。"
                "key_points最多5项、metrics最多5项、unreadable最多3项，每项不超过100字；不要重复，不要解释。"
            )
            ocr = ocr_by_name.get(image.name, "")
            payload = {
                "model": args.model.stem,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": f"{instruction}\nOCR候选（可能有错）：{ocr}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{encoded}"}},
                ]}],
                "temperature": 0.0,
                "max_tokens": 384,
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning_budget": 0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "slide_evidence",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "maxLength": 100},
                                "key_points": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 100}},
                                "metrics": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 100}},
                                "chart_or_table": {"type": "string", "maxLength": 200},
                                "unreadable": {"type": "array", "maxItems": 5, "items": {"type": "string", "maxLength": 100}},
                            },
                            "required": ["title", "key_points", "metrics", "chart_or_table", "unreadable"],
                            "additionalProperties": False,
                        },
                    },
                },
            }
            sample_started = time.monotonic()
            record: dict[str, Any] = {"index": index, "image": str(image)}
            try:
                response = post_json(f"{base_url}/chat/completions", payload)
                content = str(response["choices"][0]["message"]["content"])
                record["raw"] = content
                parsed = validate_semantics(parse_object(content))
                record.update(
                    elapsed_seconds=round(time.monotonic() - sample_started, 4),
                    parsed=parsed,
                    json_valid=True,
                    attempts=1,
                    usage=response.get("usage", {}),
                )
            except Exception as exc:
                first_error = f"{type(exc).__name__}: {exc}"
                try:
                    retry = json.loads(json.dumps(payload, ensure_ascii=False))
                    retry["messages"][0]["content"][0]["text"] = (
                        "上一次输出因重复或过长而无效。重新观察图片并只输出简短JSON：title；"
                        "key_points最多3条；metrics只保留带可见数字/单位的原文，最多3条；无图表则chart_or_table为空；"
                        "unreadable最多2条且只描述幻灯片文字/图表，不描述人物或会场。"
                        "每条不超过60字，不重复，不解释，不猜测。\n" +
                        retry["messages"][0]["content"][0]["text"]
                    )
                    retry["max_tokens"] = 384
                    retry["repeat_penalty"] = 1.15
                    response = post_json(f"{base_url}/chat/completions", retry)
                    content = str(response["choices"][0]["message"]["content"])
                    record["retry_raw"] = content
                    parsed = validate_semantics(parse_object(content))
                    record.update(
                        elapsed_seconds=round(time.monotonic() - sample_started, 4),
                        parsed=parsed,
                        json_valid=True,
                        attempts=2,
                        first_error=first_error,
                        usage=response.get("usage", {}),
                    )
                except Exception as retry_exc:
                    record.update(
                        elapsed_seconds=round(time.monotonic() - sample_started, 4),
                        json_valid=False,
                        attempts=2,
                        error=f"{type(retry_exc).__name__}: {retry_exc}",
                        first_error=first_error,
                    )
            rss, available = memory(process.pid)
            peak_rss = max(peak_rss, rss)
            min_available = min(min_available, available)
            if resource_violation:
                raise RuntimeError(resource_violation[0])
            results.append(record)
            args.output.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2))
        valid = sum(bool(item["json_valid"]) for item in results)
        elapsed = [float(item["elapsed_seconds"]) for item in results]
        report = {
            "model": str(args.model),
            "model_dir": str(args.model_dir),
            "samples": len(results),
            "json_valid": valid,
            "json_valid_rate": valid / len(results),
            "load_seconds": round(loaded, 4),
            "mean_seconds": round(sum(elapsed) / len(elapsed), 4),
            "max_seconds": round(max(elapsed), 4),
            "peak_rss_mb": peak_rss,
            "min_mem_available_mb": min_available,
            "resource_violation": resource_violation[0] if resource_violation else None,
            "results": results,
        }
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False))
        return 0 if valid == len(results) else 2
    finally:
        monitor_stop.set()
        monitor.join(timeout=2)
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except Exception:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                pass
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
