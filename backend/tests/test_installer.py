from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import subprocess
import sqlite3
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = load('installer_preflight', ROOT / 'deploy/preflight.py')


def package_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    return load('installer_package', ROOT / 'scripts/package_k3.py')


def bundle(tmp_path, monkeypatch, name='k3-meeting-assistant/install.sh', link=False):
    module = package_module(monkeypatch)
    memory = io.BytesIO()
    with tarfile.open(fileobj=memory, mode='w:gz') as archive:
        info = tarfile.TarInfo(name)
        data = b'#!/bin/bash\nprintf "TEST_INSTALLER_%s\\n" "$1"\n'
        if link:
            info.type = tarfile.SYMTYPE
            info.linkname = '/etc/passwd'
        else:
            info.size = len(data)
        archive.addfile(info, None if link else io.BytesIO(data))
    payload = memory.getvalue()
    result = tmp_path / 'installer.run'
    template = (ROOT / 'deploy/bundle-header.sh').read_bytes()
    result.write_bytes(module.header(template, payload) + payload)
    return result


def test_bundle_extracts_and_forwards_options(tmp_path, monkeypatch):
    result = subprocess.run(['bash', str(bundle(tmp_path, monkeypatch)), '--check'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'TEST_INSTALLER_--check' in result.stdout


def test_bundle_rejects_corrupt_payload(tmp_path, monkeypatch):
    path = bundle(tmp_path, monkeypatch)
    path.write_bytes(path.read_bytes()[:-1] + b'X')
    result = subprocess.run(['bash', str(path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert '校验失败' in result.stderr
    assert 'TEST_INSTALLER' not in result.stdout


@pytest.mark.parametrize('name,link', [('../escaped', False), ('/tmp/escaped', False), ('link', True)])
def test_bundle_rejects_unsafe_members(tmp_path, monkeypatch, name, link):
    result = subprocess.run(['bash', str(bundle(tmp_path, monkeypatch, name, link))], capture_output=True, text=True)
    assert result.returncode != 0
    assert '不安全' in result.stderr


def test_config_is_data_and_token_not_diagnosed(tmp_path, monkeypatch):
    path = tmp_path / 'settings'
    path.write_text('MEETING_API_TOKEN=private-token\nLOCAL_LLM_MODEL_PATH="/missing file"\nOTHER=$(touch /tmp/not-executed)\n')
    values = preflight.read_environment(path)
    assert values['LOCAL_LLM_MODEL_PATH'] == '/missing file'
    assert values['OTHER'] == '$(touch /tmp/not-executed)'
    monkeypatch.setattr(preflight.shutil, 'which', lambda name: None)
    monkeypatch.setattr(preflight.subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess(a, 1))
    report = '\n'.join(preflight.model_diagnostics(values))
    assert 'private-token' not in report
    assert '尚未配置或文件缺失' in report


def test_preflight_rejects_wrong_architecture(monkeypatch):
    monkeypatch.setattr(preflight.platform, 'machine', lambda: 'x86_64')
    assert any('riscv64' in error for error in preflight.platform_errors())


def test_preflight_missing_static(tmp_path):
    assert any('backend/static/index.html' in error for error in preflight.application_errors(tmp_path))


def test_cli_help_and_unknown_option():
    result = subprocess.run(['bash', str(ROOT / 'install.sh'), '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert '--install-deps' in result.stdout
    result = subprocess.run(['bash', str(ROOT / 'install.sh'), '--unknown'], capture_output=True, text=True)
    assert result.returncode == 64


def test_busy_meeting_prevents_install(tmp_path):
    with sqlite3.connect(tmp_path / 'meeting-assistant.sqlite3') as db:
        db.execute('CREATE TABLE jobs (status TEXT)')
        db.execute("INSERT INTO jobs VALUES ('RUNNING')")
    with pytest.raises(RuntimeError, match='任务'):
        preflight.ensure_idle({'MEETING_DATA_DIR': str(tmp_path)})
    with sqlite3.connect(tmp_path / 'meeting-assistant.sqlite3') as db:
        db.execute("UPDATE jobs SET status='DONE'")
    preflight.ensure_idle({'MEETING_DATA_DIR': str(tmp_path)})


def test_update_paths_retain_installer_entrypoint():
    for name in ('deploy/quickstart.sh', 'scripts/deploy-k3.sh'):
        assert 'scripts install.sh pyproject.toml' in (ROOT / name).read_text()
