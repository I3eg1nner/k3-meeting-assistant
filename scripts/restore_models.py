"""Restore pinned public model files. Run on K3; weights never enter Git."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import urllib.request

MODELS = [
    ("Qwen/Qwen3-4B-GGUF", "bc640142c66e1fdd12af0bd68f40445458f3869b", "text", {"Qwen3-4B-Q4_K_M.gguf", "README.md", "LICENSE"}),
]


def digest(path: Path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    if algorithm == "sha1":
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    records = []
    for repo, revision, folder, selected in MODELS:
        url = f"{args.endpoint}/api/models/{repo}/revision/{revision}?blobs=true"
        with urllib.request.urlopen(url, timeout=60) as response:
            metadata = json.load(response)
        if metadata["sha"] != revision:
            raise RuntimeError("Unexpected model revision")
        for item in metadata["siblings"]:
            name = item["rfilename"]
            if selected is not None and name not in selected:
                continue
            if name.startswith("configs/K1/") or name == ".gitattributes":
                continue
            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                raise RuntimeError("Unsafe repository filename")
            path = args.root / folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            expected = item.get("lfs", {}).get("sha256") or item["blobId"]
            algorithm = "sha256" if item.get("lfs") else "sha1"
            if not path.exists() or digest(path, algorithm) != expected:
                partial = path.with_name(path.name + ".partial")
                print(f"Downloading {repo}@{revision[:8]} / {name}", flush=True)
                route = "resolve" if item.get("lfs") else "raw"
                subprocess.run(["curl", "-fL", "--retry", "8", "--retry-all-errors", "--connect-timeout", "20", "--max-time", "7200", "-C", "-", "--silent", "--show-error", "-o", str(partial), f"{args.endpoint}/{repo}/{route}/{revision}/{name}"], check=True)
                if digest(partial, algorithm) != expected:
                    raise RuntimeError(f"Hash mismatch: {name}; refusing activation")
                partial.replace(path)
            records.append({"repository": repo, "revision": revision, "file": str(path.relative_to(args.root)), "sha256": digest(path)})
            (args.root / "download-manifest.json").write_text(json.dumps(records, indent=2) + "\n")
            print(f"Verified {name}", flush=True)
    print("All selected files verified. Model runtime acceptance is a separate step.", flush=True)


if __name__ == "__main__":
    main()
