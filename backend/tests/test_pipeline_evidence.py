from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from meeting_assistant.config import Settings
from meeting_assistant.db import Database, now_iso
from meeting_assistant.pipeline import PipelineRunner, _lexical_tokens, _media_fact_link_score


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        static_dir=tmp_path / "static",
        api_token=None,
        allow_degraded=True,
        local_llm_url="http://127.0.0.1:65530/v1",
        local_llm_model="test",
        local_llm_model_path=None,
        local_vlm_url="http://127.0.0.1:65531/v1",
        local_vlm_model="test-vlm",
        local_vlm_model_path=None,
        local_vlm_model_dir=None,
        llama_server_bin="llama-server",
        cloud_base_url=None,
        cloud_api_key=None,
        cloud_text_model=None,
        cloud_vision_model=None,
        cloud_asr_model=None,
        asr_command=None,
        diarization_command=None,
        ocr_command=None,
        vlm_command=None,
        max_upload_bytes=1024**3,
        min_mem_available_mb=128,
        max_model_rss_mb=6349,
        export_require_render=False,
    )


def _seed_meeting(db: Database) -> str:
    timestamp = now_iso()
    return db.insert(
        "meetings",
        {
            "title": "HAV 技术分享",
            "mode": "LOCAL_ONLY",
            "glossary": "ZeBu\nHAPS",
            "status": "PROCESSING",
            "created_at": timestamp,
            "updated_at": timestamp,
        },
    )


def _asset_and_evidence(db: Database, meeting_id: str, tmp_path: Path, name: str, text: str):
    path = tmp_path / name
    path.write_bytes(b"image")
    asset_id = db.insert(
        "assets",
        {
            "meeting_id": meeting_id,
            "kind": "image",
            "filename": name,
            "content_type": "image/jpeg",
            "path": str(path),
            "sha256": name,
            "size_bytes": path.stat().st_size,
            "metadata_json": "{}",
            "created_at": now_iso(),
        },
    )
    evidence_id = db.insert(
        "evidence",
        {
            "meeting_id": meeting_id,
            "asset_id": asset_id,
            "segment_id": None,
            "kind": "image",
            "locator_json": "{}",
            "text": text,
            "confidence": 0.72,
            "source_backend": "local-vlm",
            "created_at": now_iso(),
        },
    )
    return asset_id, evidence_id


def _fact(
    db: Database,
    meeting_id: str,
    statement: str,
    evidence_ids: list[str],
    *,
    confidence: float = 0.8,
    user_locked: int = 0,
) -> str:
    timestamp = now_iso()
    return db.insert(
        "facts",
        {
            "meeting_id": meeting_id,
            "kind": "topic",
            "statement": statement,
            "participants_json": "[]",
            "owner": None,
            "due_at": None,
            "status": "confirmed" if user_locked else None,
            "evidence_ids_json": json.dumps(evidence_ids),
            "confidence": confidence,
            "contradiction_ids_json": "[]",
            "source_backend": "user" if user_locked else "local-llm",
            "user_locked": user_locked,
            "created_at": timestamp,
            "updated_at": timestamp,
        },
    )


def test_long_slide_can_match_short_fact_without_jaccard_penalty():
    image = (
        "标题：ZeBu Fast Fault Emulation Technology\n"
        "目录 公司介绍 技术开放日 系统验证 平台架构 性能分析 调试流程 "
        "[视觉理解候选]\nZeBu Fast Fault Emulation Technology 支持故障注入与快速回归"
    )
    fact = "ZeBu Fast Fault Emulation Technology 支持故障注入。"
    image_tokens = _lexical_tokens(image)
    fact_tokens = _lexical_tokens(fact)
    old_jaccard = len(image_tokens & fact_tokens) / len(image_tokens | fact_tokens)

    assert old_jaccard < 0.7
    assert _media_fact_link_score(image, fact) >= 0.75


def test_number_mismatch_cannot_become_strong_similarity():
    image = "HAPS 原型平台运行频率为 2.8MHz。"
    fact = "HAPS 原型平台运行频率为 5MHz。"
    assert _media_fact_link_score(image, fact) < 0.22


def test_link_creates_candidate_but_does_not_promote_image_only_guess(tmp_path: Path):
    settings = _settings(tmp_path)
    settings.ensure_directories()
    db = Database(settings.db_path)
    db.initialize()
    meeting_id = _seed_meeting(db)
    asset_id, image_id = _asset_and_evidence(
        db,
        meeting_id,
        tmp_path,
        "fault.jpg",
        "标题：ZeBu Fast Fault Emulation\n[视觉理解候选]\nZeBu Fast Fault Emulation 支持故障注入。",
    )
    _fact(
        db,
        meeting_id,
        "ZeBu Fast Fault Emulation 支持故障注入。",
        [image_id],
        confidence=0.5,
    )

    result = PipelineRunner(settings, db, None)._link(meeting_id, "test")  # type: ignore[arg-type]
    link = db.one("SELECT * FROM media_links WHERE asset_id=?", (asset_id,))

    assert link and link["score"] >= 0.72
    assert link["confirmed"] == 0
    assert "候选" in result.message


def test_link_confirms_only_corroborated_direct_source_and_preserves_rejection(tmp_path: Path):
    settings = _settings(tmp_path)
    settings.ensure_directories()
    db = Database(settings.db_path)
    db.initialize()
    meeting_id = _seed_meeting(db)
    related_asset, related_image = _asset_and_evidence(
        db,
        meeting_id,
        tmp_path,
        "related.jpg",
        "标题：RISC-V 验证闭环\n[视觉理解候选]\nSTING 与 ImperasDV 组成 RISC-V 验证闭环。",
    )
    rejected_asset, _ = _asset_and_evidence(
        db,
        meeting_id,
        tmp_path,
        "unrelated.jpg",
        "标题：讲者介绍\n联系方式与活动日程。",
    )
    audio_id = db.insert(
        "evidence",
        {
            "meeting_id": meeting_id,
            "asset_id": None,
            "segment_id": None,
            "kind": "audio",
            "locator_json": json.dumps({"start_ms": 1000, "end_ms": 5000}),
            "text": "STING 与 ImperasDV 组成 RISC-V 验证闭环。",
            "confidence": 0.9,
            "source_backend": "local-asr",
            "created_at": now_iso(),
        },
    )
    fact_id = _fact(
        db,
        meeting_id,
        "STING 与 ImperasDV 组成 RISC-V 验证闭环。",
        [related_image, audio_id],
        confidence=0.85,
    )
    rejected_id = db.insert(
        "media_links",
        {
            "meeting_id": meeting_id,
            "asset_id": rejected_asset,
            "fact_id": fact_id,
            "score": 0.31,
            "confirmed": 0,
            "rejected": 1,
            "created_at": now_iso(),
            "updated_at": now_iso(),
        },
    )

    PipelineRunner(settings, db, None)._link(meeting_id, "test")  # type: ignore[arg-type]

    related = db.one("SELECT * FROM media_links WHERE asset_id=?", (related_asset,))
    rejected = db.one("SELECT * FROM media_links WHERE id=?", (rejected_id,))
    assert related and related["confirmed"] == 1 and related["score"] >= 0.86
    assert rejected and rejected["rejected"] == 1


def test_unrelated_image_is_not_linked_even_when_generic_words_overlap(tmp_path: Path):
    settings = _settings(tmp_path)
    settings.ensure_directories()
    db = Database(settings.db_path)
    db.initialize()
    meeting_id = _seed_meeting(db)
    asset_id, _ = _asset_and_evidence(
        db,
        meeting_id,
        tmp_path,
        "intro.jpg",
        "技术开放日系统平台产品介绍与演讲者联系方式。",
    )
    _fact(db, meeting_id, "HAPS 支持多芯片原型调试。", [], confidence=0.8)

    PipelineRunner(settings, db, None)._link(meeting_id, "test")  # type: ignore[arg-type]
    assert db.one("SELECT * FROM media_links WHERE asset_id=?", (asset_id,)) is None


class _ReadyManager:
    @contextmanager
    def session(self):
        yield True

    @staticmethod
    def assert_resources() -> None:
        return None


class _EditorialBackend:
    def __init__(self, evidence_id: str, *, unsupported_number: bool = False):
        self.evidence_id = evidence_id
        self.unsupported_number = unsupported_number
        self.calls: list[tuple[str, str, dict]] = []

    def complete_json(self, system: str, prompt: str, schema: dict, **_kwargs):
        self.calls.append((system, prompt, schema))
        if "chapters" in schema.get("properties", {}):
            return {"chapters": [{"title": "故障仿真方法", "evidence_ids": [self.evidence_id]}]}
        number = "99 倍" if self.unsupported_number else "12 小时"
        return {
            "title": "故障仿真的工程作用",
            "summary": (
                "故障仿真的核心作用是把故障注入与回归验证纳入同一流程。"
                "现场材料给出的处理目标为" + number + "，该信息仍需结合适用设计规模理解。"
            ),
            "points": ["材料将故障注入描述为回归流程中的关键环节。"],
            "table_headers": [],
            "table_rows": [],
            "evidence_note": "口径：现场材料所述处理目标；核实状态：适用条件未说明，待核实。",
        }


def _editorial_runner(tmp_path: Path, *, unsupported_number: bool = False):
    settings = _settings(tmp_path)
    settings.ensure_directories()
    db = Database(settings.db_path)
    db.initialize()
    meeting_id = _seed_meeting(db)
    _, evidence_id = _asset_and_evidence(
        db,
        meeting_id,
        tmp_path,
        "slide.jpg",
        "[视觉理解候选]\nZeBu 故障仿真把故障注入纳入回归流程，处理目标为 12 小时。",
    )
    backend = _EditorialBackend(evidence_id, unsupported_number=unsupported_number)
    db.insert("facts", {
        "meeting_id": meeting_id,
        "kind": "topic",
        "statement": "ZeBu 故障仿真把故障注入纳入回归流程，处理目标为 12 小时。",
        "evidence_ids_json": json.dumps([evidence_id]),
        "confidence": 0.5,
        "review_state": "accepted",
        "created_at": now_iso(),
        "updated_at": now_iso(),
    })
    inference = SimpleNamespace(
        model_manager=_ReadyManager(),
        local_backend=lambda: backend,
    )
    return PipelineRunner(settings, db, inference), meeting_id, evidence_id, backend


def test_editorial_prompt_requires_article_flow_and_evidence_qualification(tmp_path: Path):
    runner, meeting_id, evidence_id, backend = _editorial_runner(tmp_path)
    markdown, _html, used_ids = runner._editorial_article(
        meeting_id,
        {"title": "HAV 技术分享"},
        runner.db.all("SELECT * FROM facts WHERE meeting_id=?", (meeting_id,)),
        generate=True,
    )

    section_system, _section_prompt, section_schema = backend.calls[-1]
    assert "逐条抄录" in section_system and "先给出本节结论" in section_system
    assert "空标签" in section_system and "孤立箭头" in section_system and "截断句" in section_system
    assert "口径" in section_system and "核实状态" in section_system
    assert "evidence_note" in section_schema["required"]
    assert "12 小时" in markdown and "适用条件未说明，待核实" in markdown
    assert evidence_id in used_ids


def test_editorial_generation_rejects_number_absent_from_source(tmp_path: Path):
    runner, meeting_id, _evidence_id, _backend = _editorial_runner(
        tmp_path,
        unsupported_number=True,
    )
    markdown, _html, _used_ids = runner._editorial_article(
        meeting_id,
        {"title": "HAV 技术分享"},
        runner.db.all("SELECT * FROM facts WHERE meeting_id=?", (meeting_id,)),
        generate=True,
    )

    assert "99 倍" not in markdown
    assert "现场材料" in markdown
