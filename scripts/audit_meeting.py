#!/usr/bin/env python3
"""Deterministic evidence and report audit for a processed meeting."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from meeting_assistant.report_quality import audit_detailed_manuscript


REQUIRED_SECTIONS = {
    "summary", "article", "topics", "decisions", "actions", "risks", "questions", "media", "review",
}


def decode(value: str | None, default: Any) -> Any:
    try:
        return json.loads(value or "")
    except (json.JSONDecodeError, TypeError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("meeting_id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    meeting = connection.execute("SELECT * FROM meetings WHERE id=?", (args.meeting_id,)).fetchone()
    if not meeting:
        raise SystemExit("meeting not found")
    latest_job = connection.execute(
        "SELECT * FROM jobs WHERE meeting_id=? ORDER BY created_at DESC LIMIT 1", (args.meeting_id,)
    ).fetchone()
    assets = connection.execute("SELECT * FROM assets WHERE meeting_id=?", (args.meeting_id,)).fetchall()
    evidence = connection.execute("SELECT * FROM evidence WHERE meeting_id=?", (args.meeting_id,)).fetchall()
    evidence_by_id = {row["id"]: row for row in evidence}
    facts = connection.execute("SELECT * FROM facts WHERE meeting_id=?", (args.meeting_id,)).fetchall()
    formal = [row for row in facts if float(row["confidence"]) >= 0.55]
    heuristic_formal = [row["id"] for row in formal if row["source_backend"] in {"heuristic", "degraded"}]
    normalized_glossary = re.sub(r"\W+", "", meeting["glossary"] or "").casefold()
    image_only_formal: list[str] = []
    isolated_term_formal: list[str] = []
    fact_issues: list[dict[str, Any]] = []
    for fact in formal:
        ids = [value for value in decode(fact["evidence_ids_json"], []) if value in evidence_by_id]
        kinds = {evidence_by_id[value]["kind"] for value in ids}
        if kinds == {"image"} and not fact["user_locked"]:
            image_only_formal.append(fact["id"])
        normalized_statement = re.sub(r"\W+", "", fact["statement"]).casefold()
        if (
            fact["kind"] == "term"
            and not fact["user_locked"]
            and len(ids) < 2
            and (not normalized_statement or normalized_statement not in normalized_glossary)
        ):
            isolated_term_formal.append(fact["id"])
        supporting = "\n".join(str(evidence_by_id[value]["text"] or "") for value in ids)
        statement_numbers = set(re.findall(r"\d+(?:\.\d+)?", fact["statement"]))
        evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", supporting))
        issues = []
        if not ids:
            issues.append("missing-evidence")
        if not statement_numbers.issubset(evidence_numbers):
            issues.append("number-mismatch")
        if fact["owner"] and fact["owner"] not in fact["statement"] and fact["owner"] not in supporting:
            issues.append("owner-mismatch")
        if fact["due_at"] and fact["due_at"] not in fact["statement"] and fact["due_at"] not in supporting:
            issues.append("due-at-mismatch")
        if issues:
            fact_issues.append({"fact_id": fact["id"], "issues": issues})

    latest_version_row = connection.execute(
        "SELECT MAX(version) AS version FROM reports WHERE meeting_id=?", (args.meeting_id,)
    ).fetchone()
    latest_version = latest_version_row["version"] if latest_version_row else None
    reports = connection.execute(
        "SELECT * FROM reports WHERE meeting_id=? AND version=?", (args.meeting_id, latest_version)
    ).fetchall() if latest_version is not None else []
    image_assets = [row for row in assets if row["kind"] == "image"]
    image_asset_ids = {row["id"] for row in image_assets}
    image_evidence_assets = {row["asset_id"] for row in evidence if row["kind"] == "image"}
    degraded_image_evidence = [
        row["id"] for row in evidence if row["kind"] == "image" and row["source_backend"] == "degraded"
    ]
    audio_metadata = [decode(row["metadata_json"], {}) for row in assets if row["kind"] == "audio"]
    failed_asr = sum(len(value.get("asr_failed_segments") or []) for value in audio_metadata)
    transcript_count = connection.execute(
        "SELECT COUNT(*) FROM transcript_segments WHERE meeting_id=? AND source_backend<>'degraded'",
        (args.meeting_id,),
    ).fetchone()[0]
    article_row = next((row for row in reports if row["section"] == "article"), None)
    media_row = next((row for row in reports if row["section"] == "media"), None)
    manuscript = str(article_row["content_markdown"] if article_row else "")
    manuscript_chapters = len(re.findall(r"^## (?!附录)", manuscript, re.MULTILINE))
    manuscript_subsections = len(re.findall(r"^### ", manuscript, re.MULTILINE))
    manuscript_sources = set(re.findall(r"\[证据:([0-9a-fA-F]{8,32})\]", manuscript))
    rich_detailed = meeting["report_detail"] == "DETAILED" and len(image_assets) >= 8
    media_links = connection.execute(
        "SELECT * FROM media_links WHERE meeting_id=?", (args.meeting_id,)
    ).fetchall()
    article_html = str(article_row["content_html"] if article_row else "")
    if article_row and str(article_row["updated_at"]) != str(article_row["created_at"]):
        editorial_state = "user_edited"
    elif "本地模型分章生成" in article_html:
        editorial_state = "automatic_model_draft"
    else:
        editorial_state = "legacy_state_not_persisted"
    manuscript_audit = audit_detailed_manuscript(
        manuscript,
        evidence=[dict(row) for row in evidence],
        image_asset_ids=image_asset_ids,
        media_markdown=str(media_row["content_markdown"] if media_row else ""),
        media_links=[dict(row) for row in media_links],
        editorial_state=editorial_state,
    ) if rich_detailed else None
    checks = {
        "job_done": bool(latest_job and latest_job["status"] == "DONE"),
        "non_degraded_transcript": transcript_count > 0,
        "asr_failed_segments_zero": failed_asr == 0,
        "all_images_have_evidence": image_asset_ids == image_evidence_assets,
        "image_evidence_not_degraded": not degraded_image_evidence,
        "all_report_sections_present": {row["section"] for row in reports} == REQUIRED_SECTIONS,
        "formal_fact_evidence_coverage_100": not fact_issues,
        "formal_facts_not_heuristic": not heuristic_formal,
        "no_unconfirmed_image_only_formal_facts": not image_only_formal,
        "no_isolated_unverified_formal_terms": not isolated_term_formal,
        "detailed_manuscript_depth": not rich_detailed or (
            len(manuscript) >= 3_000 and manuscript_chapters >= 5 and manuscript_subsections >= 5
        ),
        "detailed_image_source_coverage": not rich_detailed or (
            len(manuscript_sources) >= (len(image_assets) * 9 + 9) // 10
        ),
    }
    if manuscript_audit:
        checks.update({f"detailed_{name}": value for name, value in manuscript_audit.checks.items()})

    structure_check_names = {
        "job_done", "non_degraded_transcript", "asr_failed_segments_zero",
        "all_report_sections_present", "detailed_manuscript_depth",
    }
    provenance_check_names = {
        "all_images_have_evidence", "image_evidence_not_degraded",
        "formal_fact_evidence_coverage_100", "formal_facts_not_heuristic",
        "no_unconfirmed_image_only_formal_facts", "no_isolated_unverified_formal_terms",
        "detailed_image_source_coverage", "detailed_all_references_resolve",
        "detailed_critical_claims_have_evidence", "detailed_critical_numbers_have_reviewable_source",
        "detailed_all_images_cited", "detailed_no_media_relation_conflicts",
    }
    readability_check_names = {
        "detailed_no_empty_labels", "detailed_no_isolated_arrows",
        "detailed_no_truncated_sentences", "detailed_no_internal_state_tokens",
        "detailed_no_unconfirmed_candidate_content",
    }

    def layer(names: set[str]) -> dict[str, Any]:
        selected = {name: checks[name] for name in names if name in checks}
        return {"passed": all(selected.values()), "checks": selected}

    structure_layer = layer(structure_check_names)
    readability_layer = layer(readability_check_names)
    provenance_layer = layer(provenance_check_names)
    delivery_gate_passed = all(checks.values())
    payload = {
        "meeting_id": args.meeting_id,
        "meeting_status": meeting["status"],
        "latest_job": dict(latest_job) if latest_job else None,
        "latest_report_version": latest_version,
        "counts": {
            "assets": len(assets),
            "images": len(image_assets),
            "transcript_segments": transcript_count,
            "evidence": len(evidence),
            "facts": len(facts),
            "formal_facts": len(formal),
            "asr_failed_segments": failed_asr,
            "report_sections": len(reports),
            "manuscript_characters": len(manuscript),
            "manuscript_chapters": manuscript_chapters,
            "manuscript_subsections": manuscript_subsections,
            "manuscript_sources": len(manuscript_sources),
        },
        "checks": checks,
        "quality_layers": {
            "structure": structure_layer,
            "readability": readability_layer,
            "provenance_and_numeric_consistency": provenance_layer,
            "semantic_verification": {
                "level": manuscript_audit.verification_level if manuscript_audit else "not_applicable",
                "full_text_semantically_verified": False,
                "human_domain_review_required": bool(rich_detailed),
                "editorial_state": editorial_state if rich_detailed else "not_applicable",
                "limitations": manuscript_audit.limitations if manuscript_audit else [],
            },
        },
        "metrics": {
            "manuscript": manuscript_audit.metrics if manuscript_audit else {},
            "formal_fact_issue_count": len(fact_issues),
            "confirmed_media_links": sum(bool(row["confirmed"]) for row in media_links),
            "candidate_media_links": len(media_links),
        },
        "fact_issues": fact_issues,
        "heuristic_formal_fact_ids": heuristic_formal,
        "degraded_image_evidence_ids": degraded_image_evidence,
        "image_only_formal_fact_ids": image_only_formal,
        "isolated_term_formal_fact_ids": isolated_term_formal,
        "delivery_gate_passed": delivery_gate_passed,
        "publication_readiness": "human_semantic_review_required" if rich_detailed else "delivery_gate_only",
        # Backward-compatible alias. This means deterministic delivery checks,
        # never full-text semantic verification; see quality_layers above.
        "passed": delivery_gate_passed,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded)
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
