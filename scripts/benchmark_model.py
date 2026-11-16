#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_CASES = [
    {
        "system": "你是会议事实抽取器。只提取证据直接支持的内容。",
        "prompt": (
            "证据 ev-a：李明说：本周五前完成 K3 部署脚本。\n"
            "证据 ev-b：王芳说：验收时间定在 7 月 18 日。\n"
            "返回 JSON：{\"facts\":[{\"kind\":\"action|decision|metric|risk|question|topic|term\","
            "\"statement\":\"...\",\"evidence_ids\":[\"...\"],\"confidence\":0.0}]}"
        ),
        "expected": {"ev-a": ["action"], "ev-b": ["decision", "action"]},
    },
    {
        "system": "你是会议事实抽取器。不得补全证据中没有的信息。",
        "prompt": (
            "证据 ev-c：当前远场中文识别错误率约 13%，目标是不高于 15%。\n"
            "证据 ev-d：重叠语音样本仍需人工复核。\n"
            "返回 JSON，顶层只能有 facts；每条包含 kind、statement、evidence_ids、confidence。"
        ),
        "expected": {"ev-c": ["metric"], "ev-d": ["risk", "action"]},
    },
    {
        "system": "Extract only meeting facts supported by evidence. Output JSON only.",
        "prompt": (
            "Evidence ev-e: Alice: We will freeze the API schema today.\n"
            "Evidence ev-f: Bob: The OCR benchmark is blocked by missing labels.\n"
            "Return {\"facts\":[...]}; each fact needs kind, statement, evidence_ids, confidence."
        ),
        "expected": {"ev-e": ["decision", "action"], "ev-f": ["risk"]},
    },
    {
        "system": "你是会议事实抽取器。无法确认的内容不要输出。",
        "prompt": (
            "证据 ev-g：PPT 显示端侧峰值内存 5.8 GiB。\n"
            "证据 ev-h：发言人表示暂不启用 Swap。\n"
            "返回严格 JSON，顶层字段 facts。"
        ),
        "expected": {"ev-g": ["metric"], "ev-h": ["decision"]},
    },
]

FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["topic", "decision", "action", "risk", "question", "metric", "term"],
                    },
                    "statement": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["kind", "statement", "evidence_ids", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return round(ordered[index], 3)


def _mem_available_mb() -> int | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return None


def _process_tree_pids(pid: int) -> set[int]:
    found = {pid}
    pending = [pid]
    while pending:
        current = pending.pop()
        children = Path(f"/proc/{current}/task/{current}/children")
        try:
            child_pids = [int(value) for value in children.read_text().split()]
        except (FileNotFoundError, PermissionError, ValueError):
            child_pids = []
        for child in child_pids:
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def _process_tree_rss_mb(pid: int) -> int:
    rss_kb = 0
    for process_id in _process_tree_pids(pid):
        try:
            lines = Path(f"/proc/{process_id}/status").read_text().splitlines()
        except (FileNotFoundError, PermissionError):
            continue
        for line in lines:
            if line.startswith("VmRSS:"):
                rss_kb += int(line.split()[1])
                break
    return rss_kb // 1024


class ResourceMonitor:
    def __init__(self, pid: int):
        self.pid = pid
        self.peak_rss_mb = 0
        self.min_mem_available_mb: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.wait(0.1):
            self.peak_rss_mb = max(self.peak_rss_mb, _process_tree_rss_mb(self.pid))
            available = _mem_available_mb()
            if available is not None:
                if self.min_mem_available_mb is None:
                    self.min_mem_available_mb = available
                else:
                    self.min_mem_available_mb = min(self.min_mem_available_mb, available)


def _request_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 180) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _extract_json(content: str) -> Any:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:-1]).strip()
        if content.startswith("json"):
            content = content[4:].lstrip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise


def _fact_shape_error(value: Any) -> str | None:
    if not isinstance(value, dict) or not isinstance(value.get("facts"), list):
        return "顶层必须是包含 facts 数组的对象"
    allowed_kinds = {"topic", "decision", "action", "risk", "question", "metric", "term"}
    for index, fact in enumerate(value["facts"]):
        if not isinstance(fact, dict):
            return f"facts[{index}] 不是对象"
        if fact.get("kind") not in allowed_kinds:
            return f"facts[{index}].kind 非法：{fact.get('kind')!r}"
        if not isinstance(fact.get("statement"), str):
            return f"facts[{index}].statement 不是字符串"
        if not isinstance(fact.get("evidence_ids"), list):
            return f"facts[{index}].evidence_ids 不是数组"
        if not isinstance(fact.get("confidence"), (int, float)):
            return f"facts[{index}].confidence 不是数字"
    return None


def _load_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_CASES
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("prompt"), str):
            raise ValueError(f"{path}:{line_number} 必须是包含 prompt 的 JSON 对象")
        cases.append(
            {
                "system": str(value.get("system") or DEFAULT_CASES[0]["system"]),
                "prompt": value["prompt"],
                "expected": value.get("expected"),
            }
        )
    if not cases:
        raise ValueError(f"{path} 没有可用样本")
    return cases


def _semantic_metrics(parsed: Any, case: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected")
    if not isinstance(expected, dict) or not expected:
        return {}
    facts = parsed.get("facts", []) if isinstance(parsed, dict) else []
    expected_ids = set(expected)
    returned_ids: set[str] = set()
    classified_ids: set[str] = set()
    statements: list[str] = []
    for fact in facts if isinstance(facts, list) else []:
        if not isinstance(fact, dict):
            continue
        kind = str(fact.get("kind") or "")
        statements.append(str(fact.get("statement") or ""))
        for evidence_id in fact.get("evidence_ids", []):
            if not isinstance(evidence_id, str):
                continue
            returned_ids.add(evidence_id)
            allowed = expected.get(evidence_id)
            if isinstance(allowed, list) and kind in allowed:
                classified_ids.add(evidence_id)
    correct_ids = returned_ids & expected_ids
    evidence_recall = len(correct_ids) / len(expected_ids)
    evidence_precision = len(correct_ids) / len(returned_ids) if returned_ids else 0.0
    kind_recall = len(classified_ids) / len(expected_ids)
    source_numbers = set(re.findall(r"\d+(?:\.\d+)?", str(case.get("prompt") or "")))
    statement_numbers = set(re.findall(r"\d+(?:\.\d+)?", "\n".join(statements)))
    numbers_consistent = statement_numbers.issubset(source_numbers)
    return {
        "evidence_recall": round(evidence_recall, 4),
        "evidence_precision": round(evidence_precision, 4),
        "kind_recall": round(kind_recall, 4),
        "hallucinated_evidence_ids": sorted(returned_ids - expected_ids),
        "numbers_consistent": numbers_consistent,
        "semantic_pass": (
            evidence_recall == 1
            and evidence_precision == 1
            and kind_recall == 1
            and numbers_consistent
        ),
    }


def run_throughput(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        args.bench_bin, "-m", str(args.model), "-t", str(args.threads),
        "-p", str(args.prompt_tokens), "-n", str(args.generate_tokens),
        "-mmp", "0", "-fa", "1", "-ub", "128",
    ]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
    elapsed = time.monotonic() - started
    output = result.stdout + "\n" + result.stderr
    rates = {
        name: float(value)
        for name, value in re.findall(r"\|\s+(pp\d+|tg\d+)\s+\|\s+([0-9.]+)\s+±", output)
    }
    return {
        "elapsed_seconds": round(elapsed, 3),
        "exit_code": result.returncode,
        "rates_tokens_per_second": rates,
        "raw_output": output,
    }


def _wait_for_server(base_url: str, process: subprocess.Popen[bytes], timeout: int) -> float:
    started = time.monotonic()
    deadline = started + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server 提前退出，代码 {process.returncode}")
        try:
            _request_json(f"{base_url}/models", timeout=2)
            return round(time.monotonic() - started, 3)
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(0.25)
    raise TimeoutError(f"llama-server 在 {timeout} 秒内未就绪")


def run_server_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cases = _load_cases(args.prompts_jsonl)
    base_url = f"http://127.0.0.1:{args.port}/v1"
    command = [
        args.server_bin, "-m", str(args.model), "--host", "127.0.0.1",
        "--port", str(args.port), "--ctx-size", str(args.ctx_size), "--parallel", "1",
        "--threads", str(args.threads), "--reasoning-budget", "0", "--jinja",
        "--flash-attn", "on", "--metrics",
    ]
    with tempfile.NamedTemporaryFile(prefix="k3-llama-server-", suffix=".log", delete=False) as log_file:
        log_path = Path(log_file.name)
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    monitor = ResourceMonitor(process.pid)
    monitor.start()
    samples: list[dict[str, Any]] = []
    load_seconds: float | None = None
    error: str | None = None
    try:
        load_seconds = _wait_for_server(base_url, process, args.startup_timeout)
        for index in range(args.server_samples):
            case = cases[index % len(cases)]
            payload = {
                "model": args.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": case["system"] + "\n只输出有效 JSON，不输出思考过程、Markdown 或代码块。",
                    },
                    {"role": "user", "content": case["prompt"]},
                ],
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning_budget": 0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "meeting_facts",
                        "strict": True,
                        "schema": FACT_SCHEMA,
                    },
                },
            }
            started = time.monotonic()
            sample: dict[str, Any] = {"index": index + 1}
            try:
                response = _request_json(f"{base_url}/chat/completions", payload, args.timeout)
                sample["latency_seconds"] = round(time.monotonic() - started, 3)
                content = str(response["choices"][0]["message"]["content"])
                parsed = _extract_json(content)
                shape_error = _fact_shape_error(parsed)
                sample["valid_json"] = True
                sample["valid_fact_shape"] = shape_error is None
                if shape_error:
                    sample["shape_error"] = shape_error
                if isinstance(parsed, dict):
                    sample["top_level_keys"] = sorted(str(key) for key in parsed)
                    facts = parsed.get("facts")
                    if isinstance(facts, list) and facts and isinstance(facts[0], dict):
                        sample["first_fact_keys"] = sorted(str(key) for key in facts[0])
                sample["fact_count"] = len(parsed.get("facts", [])) if isinstance(parsed, dict) else 0
                sample.update(_semantic_metrics(parsed, case))
                timings = response.get("timings") or {}
                sample["prompt_tokens_per_second"] = timings.get("prompt_per_second")
                sample["generated_tokens_per_second"] = timings.get("predicted_per_second")
            except Exception as exc:
                sample["latency_seconds"] = round(time.monotonic() - started, 3)
                sample["valid_json"] = False
                sample["valid_fact_shape"] = False
                sample["error"] = f"{type(exc).__name__}: {exc}"
            samples.append(sample)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        monitor.stop()

    latencies = [float(item["latency_seconds"]) for item in samples]
    valid_json = sum(bool(item.get("valid_json")) for item in samples)
    valid_shape = sum(bool(item.get("valid_fact_shape")) for item in samples)
    prompt_rates = [float(item["prompt_tokens_per_second"]) for item in samples if item.get("prompt_tokens_per_second")]
    generation_rates = [float(item["generated_tokens_per_second"]) for item in samples if item.get("generated_tokens_per_second")]
    semantic_samples = [item for item in samples if "semantic_pass" in item]
    log_tail = ""
    try:
        log_tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
    finally:
        log_path.unlink(missing_ok=True)
    return {
        "exit_code": process.returncode,
        "error": error,
        "load_seconds": load_seconds,
        "peak_process_tree_rss_mb": monitor.peak_rss_mb,
        "min_mem_available_mb": monitor.min_mem_available_mb,
        "requested_samples": args.server_samples,
        "completed_samples": len(samples),
        "valid_json_rate": round(valid_json / len(samples), 4) if samples else 0,
        "valid_fact_shape_rate": round(valid_shape / len(samples), 4) if samples else 0,
        "semantic_pass_rate": (
            round(sum(bool(item["semantic_pass"]) for item in semantic_samples) / len(semantic_samples), 4)
            if semantic_samples else None
        ),
        "evidence_recall_mean": (
            round(statistics.mean(float(item["evidence_recall"]) for item in semantic_samples), 4)
            if semantic_samples else None
        ),
        "evidence_precision_mean": (
            round(statistics.mean(float(item["evidence_precision"]) for item in semantic_samples), 4)
            if semantic_samples else None
        ),
        "kind_recall_mean": (
            round(statistics.mean(float(item["kind_recall"]) for item in semantic_samples), 4)
            if semantic_samples else None
        ),
        "latency_seconds": {
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
        },
        "tokens_per_second": {
            "prompt_mean": round(statistics.mean(prompt_rates), 3) if prompt_rates else None,
            "generation_mean": round(statistics.mean(generation_rates), 3) if generation_rates else None,
        },
        "samples": samples,
        "server_log_tail": log_tail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a repeatable K3 llama.cpp benchmark")
    parser.add_argument("model", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--generate-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bench-bin", default="llama-bench")
    parser.add_argument("--server-bin", default="llama-server")
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--server-samples", type=int, default=0)
    parser.add_argument("--prompts-jsonl", type=Path)
    parser.add_argument("--model-name", default="local-model")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--startup-timeout", type=int, default=90)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"模型不存在：{args.model}")
    if args.server_samples < 0:
        parser.error("--server-samples 不能小于 0")

    payload: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": str(args.model.resolve()),
        "size_bytes": args.model.stat().st_size,
        "threads": args.threads,
        "prompt_tokens": args.prompt_tokens,
        "generate_tokens": args.generate_tokens,
    }
    exit_code = 0
    if not args.skip_throughput:
        payload["throughput"] = run_throughput(args)
        exit_code = max(exit_code, int(payload["throughput"]["exit_code"] != 0))
    if args.server_samples:
        payload["structured_api"] = run_server_benchmark(args)
        server_result = payload["structured_api"]
        if server_result["error"] or server_result["completed_samples"] != args.server_samples:
            exit_code = 1

    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
