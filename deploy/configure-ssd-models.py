"""Configure restored SSD models without moving existing meeting data."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/meeting-assistant/models'))
    parser.add_argument('--mount', type=Path, default=Path('/data'))
    parser.add_argument('--opencv-lib', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not args.mount.is_mount() or not root.is_relative_to(args.mount.resolve()):
        raise SystemExit('SSD mount must be present and contain model directory')
    text = root/'text/Qwen3-4B-Q4_K_M.gguf'
    asr = root/'asr/qwen3-asr-0.6B-dynq-q40'
    vlm = root/'vlm/Qwen3.5-2B'
    speaker = root/'speaker/campplus-native-compatible.onnx'
    for path in (text, asr/'config.json', vlm/'config.json', vlm/'qwen3_5_2b-text-q41.gguf', speaker, root/'vad/silero_vad.onnx', root/'ocr/PP-OCRv5_mobile_det.onnx', root/'ocr/PP-OCRv5_mobile_rec.onnx'):
        if not path.is_file():
            raise SystemExit(f'Missing model: {path}')
    dictionary = Path('/opt/meeting-assistant/vendor/vision-src/assets/labels/ppocr_keys.txt')
    if not dictionary.is_file():
        raise SystemExit('Missing OCR runtime dictionary')
    shutil.copy2(dictionary, root/'ocr/ppocr_keys.txt')
    (root/'ocr/ppocr.yaml').write_text(f'''model_path: {root}/ocr/PP-OCRv5_mobile_det.onnx
class: deploy.ppocr.PPOCRDetector
default_params:
  rec_model_path: {root}/ocr/PP-OCRv5_mobile_rec.onnx
  dict_path: {root}/ocr/ppocr_keys.txt
  det_limit_side_len: 960
  det_db_thresh: 0.3
  det_db_box_thresh: 0.6
  det_db_unclip_ratio: 2.0
  rec_img_h: 48
  rec_img_w_max: 320
  num_threads: 4
  providers:
    - CPUExecutionProvider
''')
    scripts = '/opt/meeting-assistant/scripts'
    values = {
        'LOCAL_LLM_MODEL':'Qwen3-4B', 'LOCAL_LLM_MODEL_PATH':str(text),
        'LOCAL_VLM_MODEL':'Qwen3.5-2B', 'LOCAL_VLM_MODEL_PATH':str(vlm/'qwen3_5_2b-text-q41.gguf'),
        'LOCAL_VLM_MODEL_DIR':str(vlm), 'MEETING_VAD_MODEL_DIR':str(root/'vad'),
        'ASR_COMMAND':f'python3 {scripts}/k3_asr_adapter.py --input {{input}} --output {{output}} --model-dir {asr}',
        'DIARIZATION_COMMAND':f'python3 {scripts}/k3_diarization_adapter.py --input {{input}} --output {{output}} --model {speaker}',
        'OCR_COMMAND':f'python3 {scripts}/k3_ocr_adapter.py --input {{input}} --output {{output}} --config {root}/ocr/ppocr.yaml',
    }
    if args.opencv_lib:
        if not (args.opencv_lib/'libopencv_dnn.so.414').exists():
            raise SystemExit('Missing OpenCV compatibility library')
        values['LD_LIBRARY_PATH'] = str(args.opencv_lib)
    env = Path('/etc/meeting-assistant.env')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = env.with_name(env.name+'.before-ssd-'+stamp)
    shutil.copy2(env, backup)
    os.chmod(backup, 0o600)
    lines = [line for line in env.read_text().splitlines() if line.partition('=')[0] not in values]
    lines.extend(f'{key}={value}' for key,value in values.items())
    # Keep the existing token and meeting storage directory unchanged.
    temporary = env.with_name(env.name+'.ssd-pending')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write('\n'.join(lines)+'\n')
    temporary.replace(env)
    dropin = Path('/etc/systemd/system/meeting-assistant.service.d')
    dropin.mkdir(exist_ok=True)
    (dropin/'ssd-models.conf').write_text(f'[Unit]\nRequiresMountsFor={root}\nConditionPathIsMountPoint={args.mount}\n')
    print(f'Configured SSD models; previous configuration: {backup}. Restart service after runtime checks.')

if __name__ == '__main__':
    main()
