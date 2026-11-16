#!/usr/bin/env bash
set -euo pipefail

env_file="${1:-/etc/meeting-assistant.env}"
if [[ ! -f "$env_file" ]]; then
  echo "配置文件不存在：$env_file" >&2
  exit 1
fi

set_value() {
  local key="$1" value="$2" escaped
  escaped="${value//&/\\&}"
  escaped="${escaped//|/\\|}"
  if grep -q "^${key}=" "$env_file"; then
    sed -i "s|^${key}=.*$|${key}=${escaped}|" "$env_file"
  else
    printf '%s=%s\n' "$key" "$value" >> "$env_file"
  fi
}

# The two 1.7B K3 ASR quantizations produced empty/repetitive output on the
# real mixed-language meeting. Keep the verified 0.6B q40 package as default.
set_value LOCAL_VLM_URL "http://127.0.0.1:18086/v1"
set_value LOCAL_VLM_MODEL "Qwen3.5-2B"
set_value LOCAL_VLM_MODEL_PATH "/var/lib/meeting-assistant/models/vlm/Qwen3.5-2B/qwen3_5_2b-text-q41.gguf"
set_value LOCAL_VLM_MODEL_DIR "/var/lib/meeting-assistant/models/vlm/Qwen3.5-2B"
set_value ASR_COMMAND "python3 /opt/meeting-assistant/scripts/k3_asr_adapter.py --input {input} --output {output} --model-dir /var/lib/meeting-assistant/models/asr/qwen3-asr-0.6B-dynq-q40"
set_value DIARIZATION_COMMAND "python3 /opt/meeting-assistant/scripts/k3_diarization_adapter.py --input {input} --output {output}"
set_value OCR_COMMAND "python3 /opt/meeting-assistant/scripts/k3_ocr_adapter.py --input {input} --output {output}"
set_value MAX_MODEL_RSS_MB "6349"
set_value MIN_MEM_AVAILABLE_MB "512"
set_value MAX_UPLOAD_BYTES "2147483648"

chmod 0600 "$env_file"
echo "K3 模型与资源门禁已写入 $env_file"
