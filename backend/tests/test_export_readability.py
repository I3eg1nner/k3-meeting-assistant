from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image
from docx import Document

from meeting_assistant.db import Database, now_iso
from meeting_assistant.exporter import ReportExporter
from meeting_assistant.report_quality import audit_docx

from .test_pipeline import settings_for


def _detailed_export(tmp_path: Path) -> Path:
    settings = settings_for(tmp_path)
    settings.ensure_directories()
    db = Database(settings.db_path)
    db.initialize()
    created_at = now_iso()
    meeting_id = "meeting-readability"
    db.insert(
        "meetings",
        {
            "id": meeting_id,
            "title": "HAV 技术开放日 2026",
            "mode": "LOCAL_ONLY",
            "report_detail": "DETAILED",
            "glossary": "",
            "status": "DONE",
            "created_at": created_at,
            "updated_at": created_at,
        },
    )

    image_path = tmp_path / "slide.png"
    Image.new("RGB", (1280, 720), "white").save(image_path)
    asset_id = "asset-image-01"
    db.insert(
        "assets",
        {
            "id": asset_id,
            "meeting_id": meeting_id,
            "kind": "image",
            "filename": "slide.png",
            "content_type": "image/png",
            "path": str(image_path),
            "sha256": "0" * 64,
            "size_bytes": image_path.stat().st_size,
            "metadata_json": "{}",
            "created_at": created_at,
        },
    )
    evidence_id = "e" * 32
    db.insert(
        "evidence",
        {
            "id": evidence_id,
            "meeting_id": meeting_id,
            "asset_id": asset_id,
            "segment_id": None,
            "kind": "image",
            "locator_json": "{}",
            "text": "标题：验证架构\n[视觉理解候选]\n1000TB~100000（错误且未经确认）",
            "confidence": 0.61,
            "source_backend": "local-vlm",
            "created_at": created_at,
        },
    )
    fact_id = "fact-candidate-01"
    db.insert(
        "facts",
        {
            "id": fact_id,
            "meeting_id": meeting_id,
            "kind": "topic",
            "statement": "自动生成的图片事实候选",
            "participants_json": "[]",
            "owner": None,
            "due_at": None,
            "status": None,
            "evidence_ids_json": "[]",
            "confidence": 0.4,
            "contradiction_ids_json": "[]",
            "source_backend": "local-vlm",
            "user_locked": 0,
            "created_at": created_at,
            "updated_at": created_at,
        },
    )
    db.insert(
        "media_links",
        {
            "meeting_id": meeting_id,
            "asset_id": asset_id,
            "fact_id": fact_id,
            "score": 0.2,
            "confirmed": 0,
            "rejected": 0,
            "created_at": created_at,
            "updated_at": created_at,
        },
    )

    chapters = []
    for number in range(1, 11):
        citation = f" [证据:{evidence_id}]" if number == 1 else ""
        chapters.append(
            f"## 第{number}章 技术主题 {number}\n"
            f"### {number}.1 验证闭环{citation}\n"
            f"本节是经过编辑确认的技术正文 {number}。{citation}"
        )
    article = "# 演讲内容整理文稿\n\n## 阅读与核对说明\n正文证据可回溯。\n\n" + "\n\n".join(chapters)
    for section, markdown in {
        "summary": "# 摘要\n- 本次为技术分享。",
        "article": article,
        "topics": "# 议题\n暂无已确认内容。",
        "decisions": "# 决定\n暂无已确认内容。",
        "actions": "# 行动\n暂无已确认内容。",
        "risks": "# 风险\n暂无已确认内容。",
        "questions": "# 问题\n暂无已确认内容。",
        "media": "# 图片\n暂无会议图片。",
        "review": "# 待核实\n暂无已确认内容。",
    }.items():
        db.insert(
            "reports",
            {
                "meeting_id": meeting_id,
                "version": 1,
                "section": section,
                "content_markdown": markdown,
                "content_html": "",
                "evidence_ids_json": "[]",
                "created_at": created_at,
                "updated_at": created_at,
            },
        )

    output = tmp_path / "readability.docx"
    ReportExporter(settings, db)._build_docx(meeting_id, output, "DETAILED")
    return output


def test_detailed_export_separates_article_evidence_and_hides_visual_candidates(tmp_path: Path):
    output = _detailed_export(tmp_path)
    with zipfile.ZipFile(output) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")

    assert "HAV 技术开放日 2026" in document_xml
    assert "演讲内容整理文稿" in document_xml
    assert "第一部分｜技术内容整理文稿" in document_xml
    assert "第二部分｜会议纪要与核实边界" in document_xml
    assert "第三部分｜来源与证据附录" in document_xml
    assert "执行摘要" not in document_xml
    assert document_xml.index("第一部分｜技术内容整理文稿") < document_xml.index("会议记录概况")
    assert "第10章 技术主题 10" in document_xml
    assert "正文引用：S01 · 图01 → 1.1 验证闭环" in document_xml
    assert "图片—结构化事实：存在自动候选，待人工确认" not in document_xml
    assert "尚未完成图片—结构化事实的逐条人工确认" in document_xml
    assert "图 01 · slide.png" in document_xml
    assert "原始材料" in document_xml
    assert "对应正文位置" in document_xml
    root = ET.fromstring(document_xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    assert all(row.find("./w:trPr/w:cantSplit", namespace) is not None for row in root.findall(".//w:tr", namespace))
    assert "1000TB~100000" not in document_xml
    assert "视觉候选信息" not in document_xml
    assert "尚未建立可靠的文字事实关联" not in document_xml
    assert "LOCAL_ONLY" not in document_xml
    assert "报告 V1" not in document_xml
    assert "质量门禁交付" not in document_xml
    assert "TECHNICAL BRIEF" not in document_xml
    assert audit_docx(output).passed


def test_article_image_selection_uses_cited_order_and_even_distribution():
    source_ids = [f"{index:032x}" for index in range(10)]
    images = [{"id": f"image-{index}"} for index in range(10)]
    evidence = {
        source_id: {"kind": "image", "asset_id": f"image-{index}"}
        for index, source_id in enumerate(source_ids)
    }

    selected = ReportExporter._select_article_images(source_ids, evidence, images, limit=3)

    assert [item["id"] for item in selected] == ["image-0", "image-4", "image-9"]


def test_formal_fact_appendix_repeats_metric_review_boundary(tmp_path: Path):
    document = Document()
    exporter = object.__new__(ReportExporter)
    exporter._add_evidence_appendix(
        document,
        {
            "topics": [
                {
                    "statement": "系统验证性能加速效果为六千倍。",
                    "evidence": [
                        {
                            "kind": "audio",
                            "locator_json": '{"start_ms":2559000,"end_ms":2565000}',
                        }
                    ],
                }
            ]
        },
        {},
    )
    output = tmp_path / "appendix.docx"
    document.save(output)
    with zipfile.ZipFile(output) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")

    assert "系统验证性能加速效果为六千倍。" in document_xml
    assert "比较基线、对象与适用范围需核实" in document_xml
