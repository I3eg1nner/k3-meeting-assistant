from __future__ import annotations

import json
from pathlib import Path

from meeting_assistant.config import Settings
from meeting_assistant.db import Database, now_iso
from meeting_assistant.exporter import ReportExporter
from meeting_assistant.inference import InferenceRegistry
from meeting_assistant.pipeline import PipelineRunner, _apply_known_terms, _canonical_terms, _lexical_tokens, _parse_feishu_transcript


def settings_for(tmp_path: Path) -> Settings:
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


def test_feishu_transcript_preserves_speakers_and_timestamps():
    transcript = (
        "2026年7月10日 上午 10:01|1小时 31分钟 8秒\n\n"
        "文字记录:\n"
        "说话人 1 00:00\n能看到吗？\n\n"
        "说话人 2 00:02\n可以播放一下吗？\n\n"
        "说话人 1 01:30:58\n谢谢各位老师。\n"
    )
    turns = _parse_feishu_transcript(transcript)
    assert len(turns) == 3
    assert turns[0] == {
        "speaker": "Speaker 1",
        "start": 0,
        "end": 2,
        "confidence": 0.85,
        "text": "能看到吗？",
    }
    assert turns[1]["speaker"] == "Speaker 2"
    assert turns[2]["start"] == 5458
    assert turns[2]["end"] == 5468


def test_fact_windows_hard_limit_long_speaker_turns(tmp_path: Path):
    settings = settings_for(tmp_path)
    runner = PipelineRunner(settings, None, None)  # type: ignore[arg-type]
    evidence = [{"id": "e1", "text": "很长的内容。" * 900}]
    windows = runner._windows(evidence)
    assert len(windows) > 1
    assert all(sum(len(item["text"]) for item in window) <= 2000 for window in windows)
    assert all({item["id"] for item in window} == {"e1"} for window in windows)


def test_asr_segments_receive_greatest_overlap_speaker():
    asr = [
        {"start": 0.0, "end": 2.0, "text": "第一句"},
        {"start": 1.8, "end": 4.0, "text": "第二句"},
        {"start": 5.0, "end": 6.0, "text": "无匹配"},
    ]
    turns = [
        {"start": 0.0, "end": 2.2, "speaker": "Speaker 1"},
        {"start": 2.2, "end": 4.2, "speaker": "Speaker 2"},
    ]
    assigned = PipelineRunner._assign_diarized_speakers(asr, turns)
    assert assigned[0]["speaker"] == "Speaker 1"
    assert assigned[1]["speaker"] == "Speaker 2"
    assert "speaker" not in assigned[2]


def test_mixed_language_tokens_match_chinese_phrases_and_english_terms():
    slide = _lexical_tokens("性能验证必须充分，RISC-V system validation")
    fact = _lexical_tokens("会议讨论性能验证与 System Validation")
    assert {"性能", "能验", "验证", "system", "validation"}.issubset(slide & fact)


def test_title_glossary_corrects_only_close_asr_term_variants():
    corrected = _apply_known_terms(
        "HIV工作流与 HIV 平台、HIPS 平台、12 hrs TaT",
        ["HAV", "RISC-V", "HAPS", "ATS"],
        {"HAV", "RISC-V"},
    )
    assert corrected == "HAV工作流与 HAV 平台、HAPS 平台、12 hrs TaT"


def test_editorial_html_hides_internal_evidence_and_renders_structure():
    rendered = PipelineRunner._editorial_html(
        "# 技术文稿\n\n> 内容导读 [证据:aaaaaaaa]\n\n"
        "## 一、平台能力\n\n### 1.1 核心方法\n\n"
        "解释性正文。 [证据:aaaaaaaa]\n\n- 关键要点 [证据:bbbbbbbb]"
    )

    assert "[证据:" not in rendered
    assert "<h1>技术文稿</h1>" in rendered
    assert "<h2>一、平台能力</h2>" in rendered
    assert "<h3>1.1 核心方法</h3>" in rendered
    assert "<blockquote>" in rendered
    assert "<ul>" in rendered and "<li>" in rendered


def test_repeated_image_terms_extend_title_glossary_corrections():
    terms = _canonical_terms(
        "HAV 技术分享",
        "RISC-V",
        [
            {"kind": "image", "text": "OCR junk\n[视觉理解候选]\nHAPS platform and STING"},
            {"kind": "image", "text": "OCR junk\n[视觉理解候选]\nHAPS / STING"},
        ],
    )
    assert {"HAV", "RISC-V", "HAPS", "STING"}.issubset(terms)


def seed_meeting(db: Database, tmp_path: Path) -> tuple[str, str]:
    timestamp = now_iso()
    meeting_id = db.insert(
        "meetings",
        {
            "title": "K3 原型评审",
            "mode": "LOCAL_ONLY",
            "glossary": "RISC-V\nSpacemiT",
            "status": "DRAFT",
            "created_at": timestamp,
            "updated_at": timestamp,
        },
    )
    transcript = tmp_path / "meeting.txt"
    transcript.write_text(
        "本次会议决定采用 K3 作为本地推理平台。\n"
        "张工负责本周五前完成 ASR 模型适配。\n"
        "当前风险是 8GB 内存可能限制多模型并发。\n"
        "图片 OCR 的准确率是否达标？"
    )
    asset_id = db.insert(
        "assets",
        {
            "meeting_id": meeting_id,
            "kind": "transcript",
            "filename": "meeting.txt",
            "content_type": "text/plain",
            "path": str(transcript),
            "sha256": "test",
            "size_bytes": transcript.stat().st_size,
            "metadata_json": "{}",
            "created_at": timestamp,
        },
    )
    return meeting_id, asset_id


def test_pipeline_generates_evidence_facts_and_report(tmp_path: Path):
    settings = settings_for(tmp_path)
    settings.ensure_directories()
    db = Database(settings.db_path)
    db.initialize()
    meeting_id, _ = seed_meeting(db, tmp_path)
    timestamp = now_iso()
    job_id = db.insert(
        "jobs",
        {
            "meeting_id": meeting_id,
            "status": "QUEUED",
            "current_stage": "INGEST",
            "progress": 0,
            "error": None,
            "created_at": timestamp,
            "started_at": None,
            "finished_at": None,
            "updated_at": timestamp,
        },
    )
    inference = InferenceRegistry(settings)
    runner = PipelineRunner(settings, db, inference)
    runner.run(job_id)

    job = db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
    assert job and job["status"] == "DONE", job
    segments = db.all("SELECT * FROM transcript_segments WHERE meeting_id=?", (meeting_id,))
    facts = db.all("SELECT * FROM facts WHERE meeting_id=?", (meeting_id,))
    reports = db.all("SELECT * FROM reports WHERE meeting_id=?", (meeting_id,))
    assert len(segments) == 4
    assert {fact["kind"] for fact in facts} >= {"decision", "action", "risk", "question"}
    assert len(reports) == 9
    assert any(report["section"] == "article" for report in reports)
    for fact in facts:
        assert json.loads(fact["evidence_ids_json"])


def test_user_locked_fact_survives_reprocessing(tmp_path: Path):
    settings = settings_for(tmp_path)
    settings.ensure_directories()
    db = Database(settings.db_path)
    db.initialize()
    meeting_id, _ = seed_meeting(db, tmp_path)
    timestamp = now_iso()
    db.insert(
        "facts",
        {
            "meeting_id": meeting_id,
            "kind": "decision",
            "statement": "人工确认：采用 K3。",
            "participants_json": "[]",
            "owner": None,
            "due_at": None,
            "status": "confirmed",
            "evidence_ids_json": "[]",
            "confidence": 1,
            "contradiction_ids_json": "[]",
            "source_backend": "user",
            "user_locked": 1,
            "created_at": timestamp,
            "updated_at": timestamp,
        },
    )
    runner = PipelineRunner(settings, db, InferenceRegistry(settings))
    runner._asr(meeting_id, "test")
    runner._fact_extract(meeting_id, "test")
    locked = db.one("SELECT * FROM facts WHERE meeting_id=? AND user_locked=1", (meeting_id,))
    assert locked and locked["statement"] == "人工确认：采用 K3。"


def test_verify_republishes_report_without_unsupported_numbers(tmp_path: Path):
    settings = settings_for(tmp_path)
    settings.ensure_directories()
    db = Database(settings.db_path)
    db.initialize()
    meeting_id, _ = seed_meeting(db, tmp_path)
    runner = PipelineRunner(settings, db, InferenceRegistry(settings))
    runner._asr(meeting_id, "test")
    evidence = db.one("SELECT id FROM evidence WHERE meeting_id=? ORDER BY created_at LIMIT 1", (meeting_id,))
    assert evidence
    timestamp = now_iso()
    fact_id = db.insert(
        "facts",
        {
            "meeting_id": meeting_id,
            "kind": "decision",
            "statement": "会议决定采购 99 台 K3。",
            "participants_json": "[]",
            "owner": None,
            "due_at": None,
            "status": "confirmed",
            "evidence_ids_json": json.dumps([evidence["id"]]),
            "confidence": 0.9,
            "contradiction_ids_json": "[]",
            "source_backend": "test",
            "user_locked": 1,
            "created_at": timestamp,
            "updated_at": timestamp,
        },
    )
    runner._report(meeting_id, "test")
    result = runner._verify(meeting_id, "test")
    fact = db.one("SELECT * FROM facts WHERE id=?", (fact_id,))
    latest = db.one("SELECT MAX(version) AS version FROM reports WHERE meeting_id=?", (meeting_id,))
    decisions = db.one(
        "SELECT * FROM reports WHERE meeting_id=? AND version=? AND section='decisions'",
        (meeting_id, latest["version"]),
    )
    review = db.one(
        "SELECT * FROM reports WHERE meeting_id=? AND version=? AND section='review'",
        (meeting_id, latest["version"]),
    )
    assert result.degraded
    assert fact and fact["confidence"] == 0 and fact["status"] == "needs_review"
    assert latest and latest["version"] == 2
    assert decisions and "99 台" not in decisions["content_markdown"]
    assert review and "99 台" in review["content_markdown"]


def test_docx_export(tmp_path: Path):
    settings = settings_for(tmp_path)
    settings.ensure_directories()
    db = Database(settings.db_path)
    db.initialize()
    meeting_id, _ = seed_meeting(db, tmp_path)
    runner = PipelineRunner(settings, db, InferenceRegistry(settings))
    runner._asr(meeting_id, "test")
    runner._fact_extract(meeting_id, "test")
    from PIL import Image

    image_path = tmp_path / "slide.jpg"
    Image.new("RGB", (640, 360), "white").save(image_path)
    asset_id = db.insert(
        "assets",
        {
            "meeting_id": meeting_id,
            "kind": "image",
            "filename": "slide.jpg",
            "content_type": "image/jpeg",
            "path": str(image_path),
            "sha256": "image-test",
            "size_bytes": image_path.stat().st_size,
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
            "text": "K3 架构图",
            "confidence": 0.9,
            "source_backend": "test",
            "created_at": now_iso(),
        },
    )
    runner._report(meeting_id, "test")
    path = ReportExporter(settings, db).docx(meeting_id)
    assert path.exists()
    assert path.stat().st_size > 1000
    media = db.one("SELECT * FROM reports WHERE meeting_id=? AND section='media'", (meeting_id,))
    assert media and "slide.jpg" in media["content_markdown"]
    assert evidence_id[:8] in media["content_markdown"]
    assert "图片—结构化事实：尚未单独确认" in media["content_markdown"]
    import zipfile

    with zipfile.ZipFile(path) as archive:
        assert any(name.startswith("word/media/") for name in archive.namelist())
        assert "word/footnotes.xml" not in archive.namelist()
        document_xml = archive.read("word/document.xml").decode("utf-8")
        assert "执行摘要" in document_xml
        assert "决策与行动状态" in document_xml
        assert "K3 架构图" in document_xml
        assert "descr=\"图" in document_xml
        assert "[证据:" not in document_xml
        assert "[[MAFN" not in document_xml
    audit = db.one(
        "SELECT * FROM export_audits WHERE meeting_id=? AND format='docx' ORDER BY created_at DESC LIMIT 1",
        (meeting_id,),
    )
    assert audit and audit["status"] == "PASSED"


def test_report_explicitly_states_when_no_decision_or_action_exists(tmp_path: Path):
    settings = settings_for(tmp_path)
    settings.ensure_directories()
    db = Database(settings.db_path)
    db.initialize()
    timestamp = now_iso()
    meeting_id = db.insert(
        "meetings",
        {
            "title": "技术分享",
            "mode": "LOCAL_ONLY",
            "glossary": "",
            "status": "DONE",
            "created_at": timestamp,
            "updated_at": timestamp,
        },
    )
    evidence_id = db.insert(
        "evidence",
        {
            "meeting_id": meeting_id,
            "asset_id": None,
            "segment_id": None,
            "kind": "audio",
            "locator_json": json.dumps({"start_ms": 1_000, "end_ms": 8_000}),
            "text": "分享系统级验证方法。",
            "confidence": 0.9,
            "source_backend": "test",
            "created_at": timestamp,
        },
    )
    db.insert(
        "facts",
        {
            "meeting_id": meeting_id,
            "kind": "topic",
            "statement": "分享系统级验证方法。",
            "participants_json": "[]",
            "owner": None,
            "due_at": None,
            "status": None,
            "evidence_ids_json": json.dumps([evidence_id]),
            "confidence": 0.9,
            "contradiction_ids_json": "[]",
            "source_backend": "test",
            "user_locked": 0,
            "created_at": timestamp,
            "updated_at": timestamp,
        },
    )
    runner = PipelineRunner(settings, db, InferenceRegistry(settings))
    runner._report(meeting_id, "test")
    path = ReportExporter(settings, db).docx(meeting_id)
    from docx import Document

    document = Document(path)
    text = "\n".join(
        [paragraph.text for paragraph in document.paragraphs]
        + [cell.text for table in document.tables for row in table.rows for cell in row.cells]
    )
    assert "会议决策" in text and "行动项" in text
    assert text.count("未形成") >= 2
def test_editorial_text_removes_markdown_emphasis_markers():
    assert ReportExporter._editorial_text("*来源说明* 与 **关键结论**") == "来源说明 与 关键结论"
