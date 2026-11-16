"""Restore previously used K3 model packages from the vendor HTTPS archive."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import shutil

BASE = "https://archive.spacemit.com/spacemit-ai/model_zoo/"
FILES = [
    ("asr/qwen3-asr-0.6B-dynq-q40.tar.gz", "900602a9fbd156d905a442c91fabe646", "asr"),
    ("vlm/Qwen3.5-2B.tar.gz", "960d809dfedef5adfb6a55d7cc7a67eb", "vlm"),
    ("vision/ppocr/PP-OCRv5_mobile_det.onnx", "b82c3c45a26b43e002a37b313b171b45", "ocr"),
    ("vision/ppocr/PP-OCRv5_mobile_rec.onnx", "f1d8f4218ead7c8ed2e54dc8aeecf575", "ocr"),
    ("vp/campplus/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx", None, "speaker"),
    ("vad/silero/silero_vad.onnx", None, "vad"),
]

def checksum(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    def fetch(item):
        relative, md5, folder = item
        target = args.root / ("archives" if relative.endswith(".tar.gz") else folder) / Path(relative).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or (md5 and checksum(target, "md5") != md5):
            partial = target.with_name(target.name + ".partial")
            print(f"Downloading {relative}", flush=True)
            command = (["aria2c", "--continue=true", "--auto-file-renaming=false", "--allow-overwrite=true", "--file-allocation=none", "--max-connection-per-server=8", "--split=8", "--max-tries=8", "--retry-wait=2", "--console-log-level=warn", "--summary-interval=0", "--dir", str(partial.parent), "--out", partial.name, BASE + relative] if shutil.which("aria2c") else ["curl", "-fL", "--retry", "8", "--retry-all-errors", "--connect-timeout", "20", "--max-time", "7200", "-C", "-", "-sS", "-o", str(partial), BASE + relative])
            subprocess.run(command, check=True)
            if md5 and checksum(partial, "md5") != md5:
                raise RuntimeError(f"Vendor checksum mismatch: {relative}")
            partial.replace(target)
        if relative.endswith(".tar.gz"):
            destination = args.root / folder
            destination.mkdir(parents=True, exist_ok=True)
            with tarfile.open(target) as archive:
                archive.extractall(destination, filter="data")
        result = {"source": BASE + relative, "file": str(target.relative_to(args.root)), "vendor_md5": md5, "sha256": checksum(target, "sha256"), "verification": "vendor-md5-and-local-sha256" if md5 else "https-download-local-sha256-only"}
        print(f"Restored {relative}", flush=True)
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(fetch, FILES))
    (args.root / "vendor-download-manifest.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Downloads complete; runtime acceptance required before activation.", flush=True)

if __name__ == "__main__":
    main()
