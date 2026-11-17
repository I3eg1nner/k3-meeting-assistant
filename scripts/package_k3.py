"""Build a self-contained K3 application installer; excludes models and user data."""
from __future__ import annotations
import hashlib
import io
from pathlib import Path
import tarfile

import prepare_release

ROOT = Path(__file__).resolve().parents[1]


def header(template: bytes, payload: bytes) -> bytes:
    result = template.replace(b'@SHA256@', hashlib.sha256(payload).hexdigest().encode())
    offset = 0
    while True:
        candidate = result.replace(b'@OFFSET@', str(offset).encode())
        if len(candidate) == offset:
            return candidate
        offset = len(candidate)


def main() -> None:
    static = ROOT / 'backend/static'
    if not (static / 'index.html').is_file():
        raise SystemExit('先在开发机运行 npm ci && npm run build:k3；板端无需 Node.js。')
    prepare_release.main()
    payload = io.BytesIO()
    with tarfile.open(ROOT / 'release/k3-meeting-assistant-source.tar.gz') as source:
        with tarfile.open(fileobj=payload, mode='w:gz') as target:
            for member in source.getmembers():
                target.addfile(member, source.extractfile(member))
            for path in sorted(static.rglob('*')):
                if path.is_symlink():
                    raise SystemExit('静态资源不能包含符号链接。')
                if path.is_file():
                    if path.suffix == '.map':
                        continue
                    target.add(path, arcname=f'k3-meeting-assistant/{path.relative_to(ROOT)}', recursive=False)
    data = payload.getvalue()
    template = (ROOT / 'deploy/bundle-header.sh').read_bytes()
    output = ROOT / 'release/k3-meeting-assistant-k3.run'
    output.write_bytes(header(template, data) + data)
    output.chmod(0o755)
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.run.sha256').write_text(f'{sha}  {output.name}\n')
    print(f'安装包: {output}\nSHA-256: {sha}')


if __name__ == '__main__':
    main()
