from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from meeting_assistant.report_quality import audit_detailed_manuscript, audit_pdf
from meeting_assistant.db import SCHEMA


def _image_evidence(identifier: str = "aaaaaaaa11111111", text: str = "峰值频率为 5 MHz，路线图计划在 2026 年交付。"):
    return {
        "id": identifier,
        "asset_id": "image-1",
        "kind": "image",
        "text": text,
        "confidence": 0.9,
        "source_backend": "local",
    }


def test_detailed_manuscript_gate_accepts_cited_readable_claim_but_requires_human_review():
    result = audit_detailed_manuscript(
        "## 技术路线\n### 性能与计划\n- 峰值频率为 5 MHz，路线图计划在 2026 年交付。[证据:aaaaaaaa]",
        evidence=[_image_evidence()],
        image_asset_ids={"image-1"},
        media_markdown="- 图 1｜正文来源已确认；结构化事实关联待确认。[证据:aaaaaaaa]",
        media_links=[],
        editorial_state="human_edited",
    )

    assert result.passed is True
    assert result.verification_level == "deterministic_source_consistency"
    assert result.limitations
    assert result.metrics["image_source_coverage"] == 1.0
    assert result.metrics["cited_images_without_confirmed_fact_link"] == ["image-1"]


def test_detailed_manuscript_gate_rejects_reader_facing_generation_artifacts():
    markdown = """## 技术内容
### 候选
- Copilot：[证据:aaaaaaaa]
- → HW 辅助测试自动化 → [证据:aaaaaaaa]
- 方案包括 [证据:aaaaaaaa]
- LOCAL_ONLY · DETAILED · 报告 V16
- 视觉候选信息（需核对原图）[证据:aaaaaaaa]
"""
    result = audit_detailed_manuscript(
        markdown,
        evidence=[_image_evidence()],
        image_asset_ids={"image-1"},
    )

    assert result.passed is False
    assert result.checks["no_empty_labels"] is False
    assert result.checks["no_isolated_arrows"] is False
    assert result.checks["no_truncated_sentences"] is False
    assert result.checks["no_internal_state_tokens"] is False
    assert result.checks["no_unconfirmed_candidate_content"] is False


def test_detailed_manuscript_gate_requires_resolvable_evidence_and_matching_numbers():
    result = audit_detailed_manuscript(
        "## 性能\n- 峰值频率达到 8 MHz。[证据:aaaaaaaa]\n- 预计在 2027 年交付。[证据:bbbbbbbb]",
        evidence=[_image_evidence()],
        image_asset_ids={"image-1"},
        editorial_state="human_edited",
    )

    assert result.checks["all_references_resolve"] is False
    assert result.checks["critical_numbers_have_reviewable_source"] is True
    assert result.metrics["critical_numbers_exact_match"] is False
    assert result.metrics["numeric_mismatches"][0]["numbers"] == ["8"]
    # Human editing may route image-backed numbers to visual review, but the
    # unresolved second reference still blocks delivery.
    assert result.passed is False


def test_automatic_draft_cannot_route_numeric_mismatch_to_visual_review():
    result = audit_detailed_manuscript(
        "## 性能\n- 峰值频率达到 8 MHz。[证据:aaaaaaaa]",
        evidence=[_image_evidence()],
        image_asset_ids={"image-1"},
        editorial_state="automatic_model_draft",
    )

    assert result.checks["critical_numbers_have_reviewable_source"] is False
    assert result.metrics["numeric_visual_review_allowed"] is False


def test_meeting_audit_json_separates_delivery_gate_from_semantic_verification(tmp_path: Path):
    database = tmp_path / "audit.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(SCHEMA)
    now = "2026-07-15T00:00:00+00:00"
    meeting_id = "meeting-1"
    connection.execute(
        "INSERT INTO meetings(id,title,mode,report_detail,glossary,status,created_at,updated_at) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (meeting_id, "测试", "LOCAL_ONLY", "DETAILED", "", "DONE", now, now),
    )
    article_lines = ["## 第一章", "### 性能"]
    for index in range(8):
        asset_id = f"image-{index}"
        evidence_id = f"{index + 1:08x}" + "a" * 24
        connection.execute(
            "INSERT INTO assets(id,meeting_id,kind,filename,path,sha256,size_bytes,metadata_json,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (asset_id, meeting_id, "image", f"{index}.jpg", f"/{index}.jpg", "x", 1, "{}", now),
        )
        connection.execute(
            "INSERT INTO evidence(id,meeting_id,asset_id,kind,locator_json,text,confidence,source_backend,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (evidence_id, meeting_id, asset_id, "image", "{}", f"频率 {index + 1} MHz", 0.9, "local", now),
        )
        article_lines.append(f"- 频率为 {index + 1} MHz。[证据:{evidence_id}]")
    for section in ("summary", "article", "topics", "decisions", "actions", "risks", "questions", "media", "review"):
        markdown = "\n".join(article_lines) if section == "article" else f"## {section}"
        connection.execute(
            "INSERT INTO reports(id,meeting_id,version,section,content_markdown,content_html,evidence_ids_json,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (f"report-{section}", meeting_id, 1, section, markdown, "", "[]", now, now),
        )
    connection.commit()
    connection.close()

    script = Path(__file__).resolve().parents[2] / "scripts" / "audit_meeting.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(database), meeting_id],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(completed.stdout)

    assert completed.returncode == 2  # Other meeting-level prerequisites are absent.
    assert "quality_layers" in payload
    assert payload["quality_layers"]["semantic_verification"]["full_text_semantically_verified"] is False
    assert payload["quality_layers"]["semantic_verification"]["human_domain_review_required"] is True
    assert payload["publication_readiness"] == "human_semantic_review_required"
    assert payload["delivery_gate_passed"] is payload["passed"]


def test_detailed_manuscript_gate_detects_missing_citation_and_media_status_conflict():
    result = audit_detailed_manuscript(
        "## 路线图\n- 路线图计划在 2026 年交付。\n- 峰值频率为 5 MHz。[证据:aaaaaaaa]",
        evidence=[_image_evidence()],
        image_asset_ids={"image-1", "image-2"},
        media_markdown="- 图 1｜尚未建立可靠的文字事实关联；仅作为会议材料保留。[证据:aaaaaaaa]",
    )

    assert result.checks["critical_claims_have_evidence"] is False
    assert result.checks["all_images_cited"] is False
    assert result.checks["no_media_relation_conflicts"] is False
    assert result.metrics["missing_image_asset_ids"] == ["image-2"]


@pytest.mark.skipif(
    not shutil.which("pdfinfo") or not shutil.which("pdftoppm"),
    reason="requires Poppler",
)
def test_pdf_gate_treats_header_footer_only_page_as_blank(tmp_path: Path):
    page_one = Image.new("RGB", (612, 792), "white")
    draw_one = ImageDraw.Draw(page_one)
    draw_one.rectangle((80, 180, 530, 260), fill="black")
    page_two = Image.new("RGB", (612, 792), "white")
    draw_two = ImageDraw.Draw(page_two)
    draw_two.rectangle((180, 35, 430, 42), fill="black")
    draw_two.rectangle((290, 750, 322, 757), fill="black")
    path = tmp_path / "blank-body.pdf"
    page_one.save(path, save_all=True, append_images=[page_two])

    result = audit_pdf(path, expected_max_pages=2, required_text="")

    assert result.checks["render_all_pages"] is True
    assert result.checks["no_blank_pages"] is False
    assert "2" in result.errors[-1]
