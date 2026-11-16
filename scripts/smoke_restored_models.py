"""Serial local-only restoration smoke test; not an accuracy acceptance suite."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--audio', required=True, type=Path)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--opencv-lib')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    os.environ['MEETING_VAD_MODEL_DIR'] = str(args.root/'vad')
    if args.opencv_lib:
        os.environ['LD_LIBRARY_PATH'] = args.opencv_lib
    audio = args.output/'sample.wav'
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(args.audio),'-t','20','-ar','16000','-ac','1',str(audio)],check=True)
    commands = [
        ('asr', [sys.executable,str(scripts/'k3_asr_adapter.py'),'--input',str(audio),'--output',str(args.output/'asr.json'),'--model-dir',str(args.root/'asr/qwen3-asr-0.6B-dynq-q40')]),
        ('diarization',[sys.executable,str(scripts/'k3_diarization_adapter.py'),'--input',str(audio),'--output',str(args.output/'diarization.json'),'--model',str(args.root/'speaker/campplus-native-compatible.onnx')]),
        ('ocr',[sys.executable,str(scripts/'k3_ocr_adapter.py'),'--input',str(args.image),'--output',str(args.output/'ocr.json'),'--config',str(args.root/'ocr/ppocr.yaml')]),
    ]
    results = []
    for name, command in commands:
        started = time.monotonic()
        with (args.output/f'{name}.log').open('w') as log:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        entry = {'stage':name,'exit_code':result.returncode,'seconds':round(time.monotonic()-started,2)}
        if result.returncode == 0:
            value = json.loads((args.output/f'{name}.json').read_text())
            if name == 'asr':
                entry['segments'] = len(value.get('segments', []))
                entry['nonempty'] = any(s.get('text','').strip() for s in value.get('segments', []))
            elif name == 'ocr':
                entry['regions'] = len(value.get('regions', []))
                entry['nonempty'] = bool(value.get('text','').strip())
            else:
                entry['nonempty'] = bool(value.get('segments') or value.get('turns'))
        results.append(entry)
        (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(entry),flush=True)
    return 0 if all(row['exit_code']==0 and row.get('nonempty') for row in results) else 1

if __name__ == '__main__':
    raise SystemExit(main())
