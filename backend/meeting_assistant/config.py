from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    static_dir: Path
    api_token: str | None
    allow_degraded: bool
    local_llm_url: str
    local_llm_model: str
    local_llm_model_path: Path | None
    local_vlm_url: str
    local_vlm_model: str
    local_vlm_model_path: Path | None
    local_vlm_model_dir: Path | None
    llama_server_bin: str
    cloud_base_url: str | None
    cloud_api_key: str | None
    cloud_text_model: str | None
    cloud_vision_model: str | None
    cloud_asr_model: str | None
    asr_command: str | None
    diarization_command: str | None
    ocr_command: str | None
    vlm_command: str | None
    max_upload_bytes: int
    min_mem_available_mb: int
    max_model_rss_mb: int
    export_require_render: bool

    @property
    def db_path(self) -> Path:
        return self.data_dir / "meeting-assistant.sqlite3"

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(__file__).resolve().parents[2]
        data_dir = Path(os.getenv("MEETING_DATA_DIR", root / ".data")).resolve()
        static_dir = Path(
            os.getenv("MEETING_STATIC_DIR", root / "backend" / "static")
        ).resolve()
        model_path = os.getenv("LOCAL_LLM_MODEL_PATH")
        vlm_model_path = os.getenv("LOCAL_VLM_MODEL_PATH")
        vlm_model_dir = os.getenv("LOCAL_VLM_MODEL_DIR")
        return cls(
            data_dir=data_dir,
            static_dir=static_dir,
            api_token=os.getenv("MEETING_API_TOKEN") or None,
            allow_degraded=_bool("ALLOW_DEGRADED", True),
            local_llm_url=os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:8081/v1"),
            local_llm_model=os.getenv("LOCAL_LLM_MODEL", "local-model"),
            local_llm_model_path=Path(model_path).resolve() if model_path else None,
            local_vlm_url=os.getenv("LOCAL_VLM_URL", "http://127.0.0.1:18086/v1"),
            local_vlm_model=os.getenv("LOCAL_VLM_MODEL", "local-vlm"),
            local_vlm_model_path=Path(vlm_model_path).resolve() if vlm_model_path else None,
            local_vlm_model_dir=Path(vlm_model_dir).resolve() if vlm_model_dir else None,
            llama_server_bin=os.getenv("LLAMA_SERVER_BIN", "llama-server"),
            cloud_base_url=os.getenv("CLOUD_BASE_URL") or None,
            cloud_api_key=os.getenv("CLOUD_API_KEY") or None,
            cloud_text_model=os.getenv("CLOUD_TEXT_MODEL") or None,
            cloud_vision_model=os.getenv("CLOUD_VISION_MODEL") or None,
            cloud_asr_model=os.getenv("CLOUD_ASR_MODEL") or None,
            asr_command=os.getenv("ASR_COMMAND") or None,
            diarization_command=os.getenv("DIARIZATION_COMMAND") or None,
            ocr_command=os.getenv("OCR_COMMAND") or None,
            vlm_command=os.getenv("VLM_COMMAND") or None,
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024**3))),
            min_mem_available_mb=int(os.getenv("MIN_MEM_AVAILABLE_MB", "512")),
            max_model_rss_mb=int(os.getenv("MAX_MODEL_RSS_MB", "6349")),
            export_require_render=_bool("MEETING_EXPORT_REQUIRE_RENDER", True),
        )

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.static_dir,
            self.data_dir / "meetings",
            self.data_dir / "exports",
            self.data_dir / "models",
        ):
            path.mkdir(parents=True, exist_ok=True)
