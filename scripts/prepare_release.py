"""Build a source-only release; never include meeting data or model weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("backend/meeting_assistant", "backend/tests", "web", "tests", "scripts", "deploy", ".github")
FILES = ("README.md", "SECURITY.md", "LICENSE", "NOTICE", ".gitignore", ".env.example", "package.json", "package-lock.json", "pyproject.toml", "tsconfig.json", "vite.k3.config.ts", "eslint.config.mjs", "docs/PROJECT_OVERVIEW_ZH.md", "docs/RELEASE_CHECKLIST.md", "docs/OPERATIONS.md", "docs/VALIDATION_HISTORY.md")
ALLOWED = {".py", ".sh", ".md", ".json", ".ts", ".tsx", ".css", ".html", ".mjs", ".toml", ".yml", ".yaml", ".service", ".logrotate"}
PRIVATE_SCRIPTS = {"scripts/annotate_reference_manuscript.py", "scripts/build_readable_report.py"}
SECRET = re.compile(r"(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)")


def main():
    candidates = [ROOT / name for name in FILES]
    for name in DIRECTORIES:
        candidates.extend(p for p in (ROOT / name).rglob("*") if p.is_file() and p.suffix in ALLOWED and "__pycache__" not in p.parts)
    paths = sorted(p for p in set(candidates) if str(p.relative_to(ROOT)) not in PRIVATE_SCRIPTS)
    manifest = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"Missing or unsafe release file: {path.relative_to(ROOT)}")
        data = path.read_bytes()
        if SECRET.search(data.decode("utf-8")):
            raise SystemExit(f"Potential secret detected in {path.relative_to(ROOT)} (value withheld)")
        manifest[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    output = ROOT / "release"
    output.mkdir(exist_ok=True)
    archive = output / "k3-meeting-assistant-source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in paths:
            tar.add(path, arcname=f"k3-meeting-assistant/{path.relative_to(ROOT)}", recursive=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(paths)} source files: {archive}")
    print("Candidate only: Apache-2.0 source; requires board acceptance and manual privacy review before publication.")


if __name__ == "__main__":
    main()
