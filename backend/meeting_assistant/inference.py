from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import Settings


def mem_available_mb() -> int | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return None


def process_rss_mb(pid: int) -> int | None:
    path = Path(f"/proc/{pid}/status")
    if not path.exists():
        return None
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) // 1024
    return None


def process_group_rss_mb(pid: int) -> int | None:
    """Measure the owned session's processes, including adapter subprocesses."""
    proc = Path("/proc")
    if not proc.exists():
        return process_rss_mb(pid)
    rss = 0
    found = False
    descendants = _descendants(pid)
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # comm can contain spaces and parentheses; pgrp is field 5.
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pid or int(entry.name) in descendants:
                value = process_rss_mb(int(entry.name))
                rss += value or 0
                found = True
        except (OSError, ValueError, IndexError):
            continue
    return rss if found else None


def _descendants(pid: int) -> dict[int, str]:
    records: dict[int, tuple[int, str]] = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            records[int(path.parent.name)] = (int(fields[1]), fields[19])
        except (OSError, ValueError, IndexError):
            continue
    owned = {pid}
    while True:
        added = {child for child, (parent, _) in records.items() if parent in owned} - owned
        if not added:
            break
        owned.update(added)
    return {child: records[child][1] for child in owned if child != pid}


def terminate_owned_process(process: subprocess.Popen[Any], descendants: dict[int, str] | None = None) -> None:
    """Only accept a process launched by us with start_new_session=True."""
    owned = dict(descendants or {})
    owned.update(_descendants(process.pid))
    def signal_descendants(sig: int) -> None:
        for pid, birth in owned.items():
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                if fields[19] == birth:
                    os.kill(pid, sig)
            except (ProcessLookupError, FileNotFoundError):
                pass

    signal_descendants(signal.SIGTERM)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    signal_descendants(signal.SIGKILL)
    # The leader may have exited while a descendant remained alive.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=3)


class ResourceGuard:
    def __init__(self, settings: Settings):
        self.settings = settings

    def assert_memory(self, pid: int | None = None) -> None:
        available = mem_available_mb()
        if available is not None and available < self.settings.min_mem_available_mb:
            raise RuntimeError(
                f"可用内存仅 {available}MiB，低于安全阈值 {self.settings.min_mem_available_mb}MiB"
            )
        if pid:
            rss = process_group_rss_mb(pid)
            if rss is not None and rss > self.settings.max_model_rss_mb:
                raise RuntimeError(
                    f"模型进程 RSS {rss}MiB，超过限制 {self.settings.max_model_rss_mb}MiB"
                )


class MemoryMonitor:
    """Watch throughout blocking inference calls, not only before/after them."""

    def __init__(self, guard: ResourceGuard, process: subprocess.Popen[Any] | None,
                 stop: Any, interval: float = 0.2):
        self.guard, self.process, self.stop = guard, process, stop
        self.interval = interval
        self.error: RuntimeError | None = None
        self.descendants: dict[int, str] = {}
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._watch, name="model-memory-guard", daemon=True)

    def _watch(self) -> None:
        while not self._done.is_set():
            try:
                if self.process:
                    self.descendants.update(_descendants(self.process.pid))
                self.guard.assert_memory(self.process.pid if self.process else None)
            except RuntimeError as exc:
                self.error = exc
                self.stop()
                return
            self._done.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._done.set()
        self._thread.join(timeout=8)


def run_guarded_command(args: list[str], settings: Settings, timeout: float = 3600) -> subprocess.CompletedProcess[str]:
    """Run an adapter with a bounded lifetime and ownership of its process group."""
    guard = ResourceGuard(settings)
    guard.assert_memory()
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    stop_lock = threading.Lock()
    stopped = False

    def stop() -> None:
        nonlocal stopped
        with stop_lock:
            if not stopped:
                terminate_owned_process(process, monitor.descendants)
                stopped = True

    monitor = MemoryMonitor(guard, process, stop)
    monitor.start()
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        if monitor.error:
            raise monitor.error
        return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
    finally:
        monitor.close()
        stop()


class LlamaProcessManager:
    """Owns the single local llama-server process used by heavy model stages."""

    _heavy_model_lock = threading.Lock()

    def __init__(self, settings: Settings, purpose: str = "text"):
        self.settings = settings
        self.purpose = purpose
        self.guard = ResourceGuard(settings)
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any | None = None

    def _health(self) -> bool:
        try:
            request = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/models",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                return response.status == 200
        except Exception:
            return False

    def ensure_running(self) -> bool:
        with self._lock:
            if self._health():
                return True
            model_path = self.model_path
            if not model_path or not model_path.exists():
                return False
            if self.purpose == "vision" and not self.settings.local_vlm_model_dir:
                return False
            self.stop()
            self.guard.assert_memory()
            log_dir = self.settings.data_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_handle = (log_dir / f"llama-server-{self.purpose}.log").open("ab")
            port = self.base_url.rstrip("/").split(":")[-1].split("/")[0]
            args = [
                self.settings.llama_server_bin,
                "-m",
                str(model_path),
                "--host",
                "127.0.0.1",
                "--port",
                port,
                "--ctx-size",
                "4096",
                "--parallel",
                "1",
                "--cache-ram",
                "0",
                "--batch-size",
                "512",
                "--ubatch-size",
                "128",
                "--reasoning-budget",
                "0",
                "--threads",
                "8",
                "--metrics",
                "--jinja",
                "--flash-attn",
                "on",
                "--no-warmup",
            ]
            if self.purpose == "vision":
                args.extend(
                    [
                        "--vision-backend",
                        "smt",
                        "--smt-config-dir",
                        str(self.settings.local_vlm_model_dir),
                    ]
                )
            self._process = subprocess.Popen(
                args,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            # Cold model loading on the restored K3 stack can exceed 45 s.
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    self.stop()
                    return False
                self.guard.assert_memory(self._process.pid)
                if self._health():
                    return True
                time.sleep(0.5)
            self.stop()
            return False

    def stop(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            if process:
                terminate_owned_process(process)
            if self._log_handle:
                self._log_handle.close()
                self._log_handle = None

    def assert_resources(self) -> None:
        """Re-check hard memory limits during long multi-request sessions."""
        process = self._process
        self.guard.assert_memory(process.pid if process and process.poll() is None else None)

    @contextmanager
    def session(self) -> Iterator[bool]:
        with self._heavy_model_lock:
            monitor = None
            try:
                ready = self.ensure_running()
                if ready:
                    monitor = MemoryMonitor(self.guard, self._process, self.stop)
                    monitor.start()
                yield ready
                if monitor and monitor.error:
                    raise monitor.error
            finally:
                if monitor:
                    monitor.close()
                self.stop()

    @property
    def base_url(self) -> str:
        return (
            self.settings.local_vlm_url
            if self.purpose == "vision"
            else self.settings.local_llm_url
        )

    @property
    def model_path(self) -> Path | None:
        return (
            self.settings.local_vlm_model_path
            if self.purpose == "vision"
            else self.settings.local_llm_model_path
        )


def _post_json(
    url: str,
    payload: dict[str, Any],
    api_key: str | None = None,
    timeout: int = 600,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers=headers,
        method="POST",
    )
    # A 4B model on K3 decodes long, schema-constrained report chapters at
    # roughly 7 token/s.  A high-quality 1,400-token chapter therefore needs
    # slightly more than three minutes; treating that as a network failure
    # causes an unnecessary second generation and can discard a valid result.
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _parse_json_content(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:-1]).strip()
        if content.startswith("json"):
            content = content[4:].lstrip()
    try:
        value = json.loads(content)
        return value if isinstance(value, dict) else {"value": value}
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise


class OpenAIBackend:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        text_model: str,
        vision_model: str | None = None,
        asr_model: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.text_model = text_model
        self.vision_model = vision_model or text_model
        self.asr_model = asr_model

    def complete_json(
        self,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 320,
    ) -> dict[str, Any]:
        response_format: dict[str, Any] = {"type": "json_object"}
        if schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "structured_result", "strict": True, "schema": schema},
            }
        payload = {
            "model": self.text_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"{system}\nOnly output valid JSON. Do not include analysis, "
                        "reasoning, markdown, or code fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_budget": 0,
            "response_format": response_format,
        }
        try:
            result = _post_json(
                f"{self.base_url}/chat/completions", payload, self.api_key
            )
        except urllib.error.HTTPError:
            payload["response_format"] = {"type": "json_object"}
            result = _post_json(
                f"{self.base_url}/chat/completions", payload, self.api_key
            )
        content = result["choices"][0]["message"]["content"]
        return _parse_json_content(content)

    def analyze_image(
        self, image_path: Path, ocr_text: str, instruction: str
    ) -> dict[str, Any]:
        encoded = base64.b64encode(image_path.read_bytes()).decode()
        suffix = image_path.suffix.lower().removeprefix(".") or "jpeg"
        payload = {
            "model": self.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{instruction}\nOCR结果：{ocr_text}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{suffix};base64,{encoded}"},
                        },
                    ],
                }
            ],
            "temperature": 0.1,
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
        try:
            result = _post_json(
                f"{self.base_url}/chat/completions", payload, self.api_key
            )
        except urllib.error.HTTPError:
            payload["response_format"] = {"type": "json_object"}
            result = _post_json(
                f"{self.base_url}/chat/completions", payload, self.api_key
            )
        try:
            return _parse_json_content(result["choices"][0]["message"]["content"])
        except (json.JSONDecodeError, ValueError):
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
            result = _post_json(
                f"{self.base_url}/chat/completions", retry, self.api_key
            )
            return _parse_json_content(result["choices"][0]["message"]["content"])

    def transcribe(self, audio_path: Path, glossary: str = "") -> dict[str, Any]:
        if not self.asr_model:
            raise RuntimeError("未配置云端 ASR 模型")
        boundary = f"----meeting-assistant-{int(time.time() * 1000)}"
        fields = {
            "model": self.asr_model,
            "prompt": glossary,
            "response_format": "verbose_json",
        }
        chunks: list[bytes] = []
        for name, value in fields.items():
            chunks.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
            )
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{audio_path.name}\"\r\nContent-Type: audio/wav\r\n\r\n".encode()
        )
        chunks.append(audio_path.read_bytes())
        chunks.append(f"\r\n--{boundary}--\r\n".encode())
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/audio/transcriptions",
            data=b"".join(chunks),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read())


class InferenceRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_manager = LlamaProcessManager(settings, "text")
        self.vlm_manager = LlamaProcessManager(settings, "vision")

    def local_backend(self) -> OpenAIBackend:
        return OpenAIBackend(
            self.settings.local_llm_url,
            None,
            self.settings.local_llm_model,
        )

    def local_vision_backend(self) -> OpenAIBackend:
        return OpenAIBackend(
            self.settings.local_vlm_url,
            None,
            self.settings.local_vlm_model,
            self.settings.local_vlm_model,
        )

    def cloud_backend(self) -> OpenAIBackend | None:
        if not (
            self.settings.cloud_base_url
            and self.settings.cloud_api_key
            and self.settings.cloud_text_model
        ):
            return None
        return OpenAIBackend(
            self.settings.cloud_base_url,
            self.settings.cloud_api_key,
            self.settings.cloud_text_model,
            self.settings.cloud_vision_model,
            self.settings.cloud_asr_model,
        )
