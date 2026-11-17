"""Read-only application preflight; model diagnostics never imply inference acceptance."""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import platform
import shlex
import shutil
import sqlite3
import subprocess
import sys


def platform_errors() -> list[str]:
    errors = []
    if platform.system() != 'Linux' or platform.machine() != 'riscv64':
        errors.append('安装包仅用于 Linux riscv64 K3；开发机请使用 README 的开发流程。')
    if sys.version_info < (3, 11):
        errors.append('需要系统 Python 3.11 或更新版本。')
    if not Path('/run/systemd/system').is_dir():
        errors.append('需要运行中的 systemd。')
    for binary in ('systemctl', 'flock'):
        if not shutil.which(binary):
            errors.append(f'缺少系统命令: {binary}')
    return errors


def application_errors(root: Path) -> list[str]:
    errors = []
    for relative in ('backend/static/index.html', 'backend/meeting_assistant/main.py',
                     'deploy/activate-release.sh', 'deploy/install-k3.sh', 'pyproject.toml', '.env.example'):
        if not (root / relative).is_file():
            errors.append(f'安装包缺少 {relative}；源码需要先在开发机构建前端或使用 .run 安装包。')
    for name in ('fastapi', 'uvicorn', 'multipart', 'docx', 'PIL', 'pydantic'):
        try:
            module = importlib.import_module(name)
            if name == 'pydantic' and int(module.__version__.split('.')[0]) < 2:
                errors.append('pydantic 需要 2.x，请核对系统 Python 依赖。')
        except (ImportError, OSError) as exc:
            errors.append(f'缺少或无法加载 Python 模块 {name}: {type(exc).__name__}')
    for name in ('ffmpeg', 'libreoffice', 'pdftoppm'):
        if not shutil.which(name):
            errors.append(f'缺少应用命令: {name}（可使用 --install-deps 补装）')
    return errors


def read_environment(path: Path) -> dict[str, str]:
    values = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            key, sep, value = line.strip().partition('=')
            if sep and not key.startswith('#'):
                # Parse as data; never source/execute the configuration.
                if value[:1] in ('"', "'"):
                    parts = shlex.split(value)
                    value = parts[0] if len(parts) == 1 else value
                values[key] = value
    return values


def model_diagnostics(values: dict[str, str]) -> list[str]:
    lines = []
    for key in ('LOCAL_LLM_MODEL_PATH', 'LOCAL_VLM_MODEL_PATH'):
        present = bool(values.get(key)) and Path(values[key]).is_file()
        lines.append(f'{key}: {"文件存在，推理未验证" if present else "尚未配置或文件缺失"}')
    server = values.get('LLAMA_SERVER_BIN', 'llama-server')
    lines.append('llama-server: ' + ('已找到，推理未验证' if shutil.which(server) else '未找到'))
    env = dict(os.environ)
    if values.get('LD_LIBRARY_PATH'):
        env['LD_LIBRARY_PATH'] = values['LD_LIBRARY_PATH']
    for name in ('spacemit_vad', 'spacemit_vision', 'onnxruntime'):
        try:
            result = subprocess.run([sys.executable, '-c', f'import {name}'], env=env,
                                    capture_output=True, timeout=20)
            ok = result.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
        lines.append(f'{name}: {"导入通过，推理未验证" if ok else "缺失或无法加载，请按部署文档准备厂商运行时"}')
    lines.append('音频、OCR、说话人及模型质量仍须分别实测；下载模型不能代替原生运行时安装。')
    return lines


def ensure_idle(values: dict[str, str]) -> None:
    data = Path(values.get('MEETING_DATA_DIR', '/var/lib/meeting-assistant'))
    path = data / 'meeting-assistant.sqlite3'
    if not path.exists():
        return
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=5) as db:
        active = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('QUEUED','RUNNING')").fetchone()[0]
    if active:
        raise RuntimeError('存在排队或运行中的会议任务，请等待完成后再安装。')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--platform-only', action='store_true')
    group.add_argument('--models-only', action='store_true')
    group.add_argument('--idle-only', action='store_true')
    args = parser.parse_args()
    if args.idle_only:
        try:
            ensure_idle(read_environment(Path('/etc/meeting-assistant.env')))
        except (OSError, ValueError, sqlite3.Error, RuntimeError):
            print('无法确认任务空闲（存在活动任务或状态不可读），拒绝重启；请检查工作台及数据库。')
            return 1
        return 0
    if args.models_only:
        try:
            for line in model_diagnostics(read_environment(Path('/etc/meeting-assistant.env'))):
                print(line)
        except (OSError, ValueError):
            print('无法读取模型配置；请由管理员执行诊断。')
        return 0
    errors = platform_errors()
    if not args.platform_only:
        errors.extend(application_errors(args.root))
    for error in errors:
        print(f'[需处理] {error}')
    if errors:
        return 1
    print('平台检查通过' if args.platform_only else '应用环境检查通过（不代表模型就绪）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
