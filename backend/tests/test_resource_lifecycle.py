from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from meeting_assistant import inference
from .test_pipeline import settings_for


def sleeper():
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)


def test_startup_exception_cleans_owned_process_and_log(tmp_path, monkeypatch):
    settings = settings_for(tmp_path)
    model = tmp_path / "model.gguf"
    model.touch()
    object.__setattr__(settings, "local_llm_model_path", model)
    manager = inference.LlamaProcessManager(settings)
    process = sleeper()
    monkeypatch.setattr(manager, "_health", lambda: False)
    monkeypatch.setattr(inference.subprocess, "Popen", lambda *a, **k: process)

    def assert_memory(pid=None):
        if pid:
            raise RuntimeError("startup memory limit")

    monkeypatch.setattr(manager.guard, "assert_memory", assert_memory)
    with pytest.raises(RuntimeError, match="startup memory limit"):
        with manager.session():
            pytest.fail("startup should not succeed")
    assert process.poll() is not None
    assert manager._process is None
    assert manager._log_handle is None


def test_memory_guard_interrupts_blocking_inference(tmp_path, monkeypatch):
    manager = inference.LlamaProcessManager(settings_for(tmp_path))
    process = sleeper()
    exceed = threading.Event()

    def ensure_running():
        manager._process = process
        return True

    def assert_memory(pid=None):
        if exceed.is_set():
            raise RuntimeError("runtime memory limit")

    monkeypatch.setattr(manager, "ensure_running", ensure_running)
    monkeypatch.setattr(manager.guard, "assert_memory", assert_memory)
    with pytest.raises(RuntimeError, match="runtime memory limit"):
        with manager.session():
            exceed.set()
            # No explicit assert_resources call: the watchdog must terminate it.
            process.wait(timeout=4)
    assert process.poll() is not None
    assert manager._process is None


def test_external_server_is_never_terminated(tmp_path, monkeypatch):
    manager = inference.LlamaProcessManager(settings_for(tmp_path))
    monkeypatch.setattr(manager, "_health", lambda: True)
    monkeypatch.setattr(inference.os, "killpg", lambda *args: pytest.fail("external process must not be killed"))
    with manager.session() as ready:
        assert ready
    assert manager._process is None


def test_guarded_adapter_timeout_reaps_process(tmp_path, monkeypatch):
    launched = []
    original = subprocess.Popen

    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(inference.subprocess, "Popen", launch)
    with pytest.raises(subprocess.TimeoutExpired):
        inference.run_guarded_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            settings_for(tmp_path), timeout=0.15,
        )
    assert launched and launched[0].poll() is not None


def test_guarded_adapter_memory_limit(tmp_path, monkeypatch):
    def assert_memory(self, pid=None):
        if pid:
            raise RuntimeError("adapter memory limit")
    monkeypatch.setattr(inference.ResourceGuard, "assert_memory", assert_memory)
    with pytest.raises(RuntimeError, match="adapter memory limit"):
        inference.run_guarded_command(
            [sys.executable, "-c", "import time; time.sleep(30)"], settings_for(tmp_path), timeout=5,
        )


def test_resource_guard_checks_group_rss(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "mem_available_mb", lambda: 1024)
    monkeypatch.setattr(inference, "process_group_rss_mb", lambda pid: 7000)
    with pytest.raises(RuntimeError, match="RSS"):
        inference.ResourceGuard(settings_for(tmp_path)).assert_memory(123)


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="Linux process tree integration")
def test_terminates_adapter_child_that_created_its_own_session():
    source = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); "
        "print(p.pid,flush=True); time.sleep(30)"
    )
    parent = subprocess.Popen([sys.executable, "-c", source], stdout=subprocess.PIPE,
                              text=True, start_new_session=True)
    child_pid = int(parent.stdout.readline())
    try:
        assert child_pid in inference._descendants(parent.pid)
        inference.terminate_owned_process(parent)
        status = Path(f"/proc/{child_pid}/stat")
        if status.exists():
            assert status.read_text().rsplit(")", 1)[1].split()[0] == "Z"
        assert parent.poll() is not None
    finally:
        for pid in (child_pid, parent.pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        parent.wait(timeout=3)
        parent.stdout.close()


def test_production_preflight_rejects_missing_token(tmp_path):
    path = Path(__file__).resolve().parents[2] / "deploy" / "check-service.py"
    spec = importlib.util.spec_from_file_location("check_service", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {"MEETING_DATA_DIR": str(tmp_path), "MEETING_STATIC_DIR": str(tmp_path)}
    with pytest.raises(RuntimeError, match="MEETING_API_TOKEN"):
        module.validate_config(config)
    config["MEETING_API_TOKEN"] = " "
    with pytest.raises(RuntimeError, match="MEETING_API_TOKEN"):
        module.validate_config(config)
    config["MEETING_API_TOKEN"] = "test-token"
    module.validate_config(config)


def test_service_requires_config_and_runs_preflight():
    root = Path(__file__).resolve().parents[2]
    unit = (root / "deploy" / "meeting-assistant.service").read_text()
    installer = (root / "deploy" / "install-k3.sh").read_text()
    assert "EnvironmentFile=/etc/meeting-assistant.env" in unit
    assert "--check-config" in unit
    assert "systemctl restart meeting-assistant.service" in installer
    assert "--health" in installer
