#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "请以 root 运行此脚本"
  exit 1
fi

if [[ "${MEETING_SKIP_SYSTEM_PACKAGES:-0}" != 1 ]]; then
  apt-get update
  apt-get install -y \
  python3-fastapi python3-uvicorn python3-multipart python3-docx python3-pil \
  python3-pybind11 python3-scipy libyaml-cpp-dev nlohmann-json3-dev \
    ffmpeg opencv-spacemit libreoffice-writer poppler-utils fonts-noto-cjk
else
  # Existing installations may update application code without upgrading the OS.
  python3 -c 'import fastapi, uvicorn, multipart, docx, PIL, pydantic'
  for binary in ffmpeg libreoffice pdftoppm; do command -v "$binary" >/dev/null; done
fi

install -d -m 0755 /opt/meeting-assistant
install -d -m 0750 /var/lib/meeting-assistant/{meetings,exports,models,logs}

if [[ ! -f /etc/meeting-assistant.env ]]; then
  token="$(python3 -c 'import secrets; print(secrets.token_urlsafe(18))')"
  install -m 0600 /dev/null /etc/meeting-assistant.env
  {
    echo 'MEETING_DATA_DIR=/var/lib/meeting-assistant'
    echo 'MEETING_STATIC_DIR=/opt/meeting-assistant/backend/static'
    echo "MEETING_API_TOKEN=${token}"
    echo 'ALLOW_DEGRADED=true'
    echo 'LOCAL_LLM_URL=http://127.0.0.1:8081/v1'
    if [[ -f /var/lib/meeting-assistant/models/Qwen3-4B-Q4_K_M.gguf ]]; then
      echo 'LOCAL_LLM_MODEL=Qwen3-4B'
      echo 'LOCAL_LLM_MODEL_PATH=/var/lib/meeting-assistant/models/Qwen3-4B-Q4_K_M.gguf'
    elif [[ -f /root/Qwen3-0.6B-Q4_0.gguf ]]; then
      echo 'LOCAL_LLM_MODEL=Qwen3-0.6B'
      echo 'LOCAL_LLM_MODEL_PATH=/root/Qwen3-0.6B-Q4_0.gguf'
    else
      echo 'LOCAL_LLM_MODEL=local-model'
      echo 'LOCAL_LLM_MODEL_PATH=/var/lib/meeting-assistant/models/text-model.gguf'
    fi
    echo 'LLAMA_SERVER_BIN=llama-server'
    echo 'LOCAL_VLM_URL=http://127.0.0.1:18086/v1'
    echo 'LOCAL_VLM_MODEL=Qwen3.5-2B'
    echo 'LOCAL_VLM_MODEL_PATH=/var/lib/meeting-assistant/models/vlm/Qwen3.5-2B/qwen3_5_2b-text-q41.gguf'
    echo 'LOCAL_VLM_MODEL_DIR=/var/lib/meeting-assistant/models/vlm/Qwen3.5-2B'
    echo 'ASR_COMMAND=python3 /opt/meeting-assistant/scripts/k3_asr_adapter.py --input {input} --output {output}'
    echo 'DIARIZATION_COMMAND=python3 /opt/meeting-assistant/scripts/k3_diarization_adapter.py --input {input} --output {output}'
    echo 'OCR_COMMAND=python3 /opt/meeting-assistant/scripts/k3_ocr_adapter.py --input {input} --output {output}'
    echo 'MIN_MEM_AVAILABLE_MB=512'
    echo 'MAX_MODEL_RSS_MB=6349'
    echo 'MEETING_EXPORT_REQUIRE_RENDER=true'
  } >> /etc/meeting-assistant.env
  echo "已生成 Web 访问令牌：${token}"
fi

install -m 0644 /opt/meeting-assistant/deploy/meeting-assistant.service /etc/systemd/system/meeting-assistant.service
install -m 0644 /opt/meeting-assistant/deploy/meeting-assistant.logrotate /etc/logrotate.d/meeting-assistant
systemctl daemon-reload
systemctl enable meeting-assistant.service
systemctl restart meeting-assistant.service
/usr/bin/python3 /opt/meeting-assistant/deploy/check-service.py --health
systemctl --no-pager --full status meeting-assistant.service
