from __future__ import annotations

import asyncio
import hashlib
import html
import json
import os
import re
import threading
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from .config import Settings
from .db import Database, classify_asset, new_id, now_iso
from .exporter import ReportExporter
from .pipeline import PipelineRunner, REPORT_SECTIONS
from .schemas import FactPatch, MediaLinkPatch, MeetingCreate, MeetingPatch, ReportPatch, ReportApproval, SegmentPatch


def _json_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "metadata_json", "locator_json", "participants_json", "evidence_ids_json", "contradiction_ids_json",
        "checks_json", "metrics_json", "errors_json",
    )
    for row in rows:
        for field in fields:
            if field in row:
                key = field.removesuffix("_json")
                try:
                    row[key] = json.loads(row[field] or "null")
                except json.JSONDecodeError:
                    row[key] = None
                row.pop(field, None)
        for key in ("needs_review", "user_locked", "confirmed", "rejected"):
            if key in row:
                row[key] = bool(row[key])
    return rows


class ApiContext:
    def __init__(self, settings: Settings, db: Database, pipeline: PipelineRunner):
        self.settings = settings
        self.db = db
        self.pipeline = pipeline
        self.exporter = ReportExporter(settings, db)

    def auth(self, x_meeting_token: Annotated[str | None, Header()] = None) -> None:
        if self.settings.api_token and x_meeting_token != self.settings.api_token:
            raise HTTPException(status_code=401, detail="访问令牌无效")


def build_router(context: ApiContext) -> APIRouter:
    router = APIRouter(prefix="/api", dependencies=[Depends(context.auth)])
    db = context.db

    def remember(meeting_id: str, kind: str, row: dict[str, Any]) -> None:
        db.insert("revision_history", {
            "meeting_id": meeting_id, "entity_type": kind, "entity_id": row["id"],
            "previous_json": db.json(row), "created_at": now_iso(),
        })

    def invalidate_reports(meeting_id: str) -> None:
        db.execute("UPDATE reports SET needs_review=1 WHERE meeting_id=?", (meeting_id,))

    @router.get("/health")
    def health() -> dict[str, Any]:
        from .inference import mem_available_mb

        return {
            "status": "ok",
            "version": "0.1.0",
            "mem_available_mb": mem_available_mb(),
            "local_llm_model": context.settings.local_llm_model,
            "local_llm_model_path": str(context.settings.local_llm_model_path) if context.settings.local_llm_model_path else None,
            "asr_configured": bool(context.settings.asr_command),
            "diarization_configured": bool(context.settings.diarization_command),
            "ocr_configured": bool(context.settings.ocr_command),
            "vlm_configured": bool(
                context.settings.vlm_command
                or (context.settings.local_vlm_model_path and context.settings.local_vlm_model_dir)
            ),
            "cloud_configured": bool(context.settings.cloud_base_url and context.settings.cloud_api_key),
        }

    @router.get("/meetings")
    def meetings() -> list[dict[str, Any]]:
        return db.all(
            "SELECT m.*, (SELECT COUNT(*) FROM assets a WHERE a.meeting_id=m.id) AS asset_count, "
            "(SELECT COUNT(*) FROM facts f WHERE f.meeting_id=m.id) AS fact_count "
            "FROM meetings m ORDER BY created_at DESC"
        )

    @router.post("/meetings", status_code=201)
    def create_meeting(payload: MeetingCreate) -> dict[str, Any]:
        timestamp = now_iso()
        meeting_id = db.insert(
            "meetings",
            {
                "title": payload.title.strip(),
                "mode": payload.mode,
                "report_detail": payload.report_detail,
                "glossary": payload.glossary.strip(),
                "status": "DRAFT",
                "created_at": timestamp,
                "updated_at": timestamp,
            },
        )
        return db.one("SELECT * FROM meetings WHERE id=?", (meeting_id,)) or {}

    @router.patch("/meetings/{meeting_id}")
    def patch_meeting(meeting_id: str, payload: MeetingPatch) -> dict[str, Any]:
        if not db.one("SELECT id FROM meetings WHERE id=?", (meeting_id,)):
            raise HTTPException(404, "会议不存在")
        values = payload.model_dump(exclude_none=True)
        if values:
            values["updated_at"] = now_iso()
            sql = ", ".join(f"{key}=?" for key in values)
            db.execute(f"UPDATE meetings SET {sql} WHERE id=?", (*values.values(), meeting_id))
        return db.one("SELECT * FROM meetings WHERE id=?", (meeting_id,)) or {}

    @router.get("/meetings/{meeting_id}")
    def meeting_detail(meeting_id: str) -> dict[str, Any]:
        meeting = db.one("SELECT * FROM meetings WHERE id=?", (meeting_id,))
        if not meeting:
            raise HTTPException(404, "会议不存在")
        version = db.one("SELECT MAX(version) AS version FROM reports WHERE meeting_id=?", (meeting_id,))
        latest_version = version["version"] if version else None
        return {
            "meeting": meeting,
            "assets": _json_fields(db.all("SELECT * FROM assets WHERE meeting_id=? ORDER BY created_at", (meeting_id,))),
            "speakers": db.all("SELECT * FROM speakers WHERE meeting_id=? ORDER BY label", (meeting_id,)),
            "segments": _json_fields(db.all(
                "SELECT ts.*, s.label AS speaker_label, s.display_name AS speaker_name FROM transcript_segments ts "
                "LEFT JOIN speakers s ON s.id=ts.speaker_id WHERE ts.meeting_id=? ORDER BY ts.start_ms", (meeting_id,)
            )),
            "evidence": _json_fields(db.all("SELECT * FROM evidence WHERE meeting_id=? ORDER BY created_at", (meeting_id,))),
            "facts": _json_fields(db.all("SELECT * FROM facts WHERE meeting_id=? ORDER BY created_at", (meeting_id,))),
            "media_links": _json_fields(db.all(
                "SELECT ml.*, a.filename, f.statement AS fact_statement FROM media_links ml "
                "JOIN assets a ON a.id=ml.asset_id LEFT JOIN facts f ON f.id=ml.fact_id "
                "WHERE ml.meeting_id=? ORDER BY ml.score DESC", (meeting_id,)
            )),
            "reports": _json_fields(db.all(
                "SELECT * FROM reports WHERE meeting_id=? AND version=? ORDER BY rowid",
                (meeting_id, latest_version),
            )) if latest_version else [],
            "latest_version": latest_version,
            "jobs": db.all("SELECT * FROM jobs WHERE meeting_id=? ORDER BY created_at DESC", (meeting_id,)),
            "cloud_audit": db.all("SELECT * FROM cloud_audit WHERE meeting_id=? ORDER BY created_at DESC", (meeting_id,)),
            "export_audits": _json_fields(db.all(
                "SELECT * FROM export_audits WHERE meeting_id=? ORDER BY created_at DESC LIMIT 20", (meeting_id,)
            )),
        }

    @router.post("/meetings/{meeting_id}/assets", status_code=201)
    async def upload_assets(meeting_id: str, files: Annotated[list[UploadFile], File()]) -> list[dict[str, Any]]:
        if not db.one("SELECT id FROM meetings WHERE id=?", (meeting_id,)):
            raise HTTPException(404, "会议不存在")
        meeting_dir = context.settings.data_dir / "meetings" / meeting_id / "uploads"
        meeting_dir.mkdir(parents=True, exist_ok=True)
        created: list[dict[str, Any]] = []
        for upload in files:
            original = Path(upload.filename or "upload.bin").name
            suffix = Path(original).suffix.lower()
            asset_id = new_id()
            target = meeting_dir / f"{asset_id}{suffix}"
            digest = hashlib.sha256()
            size = 0
            try:
                with target.open("wb") as handle:
                    while chunk := await upload.read(1024 * 1024):
                        size += len(chunk)
                        if size > context.settings.max_upload_bytes:
                            raise HTTPException(413, "文件超过上传大小限制")
                        digest.update(chunk)
                        handle.write(chunk)
            except Exception:
                target.unlink(missing_ok=True)
                raise
            timestamp = now_iso()
            kind = classify_asset(original, upload.content_type)
            db.insert(
                "assets",
                {
                    "id": asset_id,
                    "meeting_id": meeting_id,
                    "kind": kind,
                    "filename": original,
                    "content_type": upload.content_type,
                    "path": str(target),
                    "sha256": digest.hexdigest(),
                    "size_bytes": size,
                    "metadata_json": "{}",
                    "created_at": timestamp,
                },
            )
            created.append(db.one("SELECT * FROM assets WHERE id=?", (asset_id,)) or {})
        db.execute("UPDATE meetings SET updated_at=? WHERE id=?", (now_iso(), meeting_id))
        return _json_fields(created)

    @router.get("/assets/{asset_id}")
    def asset_file(asset_id: str) -> FileResponse:
        asset = db.one("SELECT * FROM assets WHERE id=?", (asset_id,))
        if not asset or not Path(asset["path"]).exists():
            raise HTTPException(404, "文件不存在")
        return FileResponse(asset["path"], media_type=asset["content_type"], filename=asset["filename"])

    @router.post("/meetings/{meeting_id}/process", status_code=202)
    def process_meeting(meeting_id: str, background_tasks: BackgroundTasks) -> dict[str, Any]:
        if not db.one("SELECT id FROM meetings WHERE id=?", (meeting_id,)):
            raise HTTPException(404, "会议不存在")
        active = db.one("SELECT id FROM jobs WHERE status IN ('QUEUED','RUNNING') LIMIT 1")
        if active:
            raise HTTPException(409, f"已有任务运行：{active['id']}")
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
        background_tasks.add_task(context.pipeline.run, job_id)
        return db.one("SELECT * FROM jobs WHERE id=?", (job_id,)) or {}

    @router.get("/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        row = db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
        if not row:
            raise HTTPException(404, "任务不存在")
        return row

    @router.get("/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        if not db.one("SELECT id FROM jobs WHERE id=?", (job_id,)):
            raise HTTPException(404, "任务不存在")

        async def stream():
            last_id = 0
            while True:
                events = db.all("SELECT * FROM job_events WHERE job_id=? AND id>? ORDER BY id", (job_id, last_id))
                for event in events:
                    last_id = event["id"]
                    yield f"id: {last_id}\nevent: progress\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                current = db.one("SELECT status, error FROM jobs WHERE id=?", (job_id,))
                if current and current["status"] in {"DONE", "FAILED", "CANCELLED"}:
                    yield f"event: complete\ndata: {json.dumps(current, ensure_ascii=False)}\n\n"
                    break
                yield ": keep-alive\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @router.patch("/transcript-segments/{segment_id}")
    def patch_segment(segment_id: str, payload: SegmentPatch) -> dict[str, Any]:
        segment = db.one("SELECT * FROM transcript_segments WHERE id=?", (segment_id,))
        if not segment:
            raise HTTPException(404, "转写片段不存在")
        updates: dict[str, Any] = {}
        if payload.user_text is not None:
            updates["user_text"] = payload.user_text.strip()
        if payload.speaker_id is not None:
            if not db.one("SELECT id FROM speakers WHERE id=? AND meeting_id=?", (payload.speaker_id, segment["meeting_id"])):
                raise HTTPException(422, "说话人不属于本会议")
            updates["speaker_id"] = payload.speaker_id
        if payload.needs_review is not None:
            updates["needs_review"] = int(payload.needs_review)
        with db.transaction():
            remember(segment["meeting_id"], "transcript", segment)
            if payload.speaker_name is not None and segment.get("speaker_id"):
                db.execute("UPDATE speakers SET display_name=? WHERE id=?", (payload.speaker_name.strip() or None, segment["speaker_id"]))
            if updates:
                updates["updated_at"] = now_iso()
                sql = ", ".join(f"{key}=?" for key in updates)
                db.execute(f"UPDATE transcript_segments SET {sql} WHERE id=?", (*updates.values(), segment_id))
            if payload.user_text is not None:
                # Empty edits are intentional too; never silently restore raw ASR.
                effective = payload.user_text.strip()
                sources = db.all("SELECT * FROM evidence WHERE segment_id=?", (segment_id,))
                for source in sources:
                    remember(segment["meeting_id"], "evidence", source)
                    db.execute("UPDATE evidence SET text=?, revision=revision+1 WHERE id=?", (effective, source["id"]))
                db.execute("DELETE FROM transcript_fts WHERE segment_id=?", (segment_id,))
                db.execute("INSERT INTO transcript_fts(segment_id,meeting_id,text) VALUES(?,?,?)", (segment_id, segment["meeting_id"], effective))
                source_ids = {s["id"] for s in sources}
                for fact in db.all("SELECT * FROM facts WHERE meeting_id=?", (segment["meeting_id"],)):
                    if source_ids.intersection(json.loads(fact["evidence_ids_json"])):
                        remember(segment["meeting_id"], "fact", fact)
                        db.execute("UPDATE facts SET review_state='stale', status='needs_review', updated_at=? WHERE id=?", (now_iso(), fact["id"]))
            invalidate_reports(segment["meeting_id"])
        return db.one(
            "SELECT ts.*, s.label AS speaker_label, s.display_name AS speaker_name FROM transcript_segments ts "
            "LEFT JOIN speakers s ON s.id=ts.speaker_id WHERE ts.id=?", (segment_id,)
        ) or {}

    @router.patch("/facts/{fact_id}")
    def patch_fact(fact_id: str, payload: FactPatch) -> dict[str, Any]:
        fact = db.one("SELECT * FROM facts WHERE id=?", (fact_id,))
        if not fact:
            raise HTTPException(404, "事实不存在")
        values = payload.model_dump(exclude_unset=True)
        values.setdefault("user_locked", True)
        if payload.review_state == "accepted":
            ids = payload.evidence_ids if payload.evidence_ids is not None else json.loads(fact["evidence_ids_json"])
            valid_ids = {e["id"] for e in db.all("SELECT id FROM evidence WHERE meeting_id=? AND source_backend<>'degraded' AND text<>''", (fact["meeting_id"],))}
            if not ids or not set(ids).issubset(valid_ids):
                raise HTTPException(422, "采用事实必须引用本会议的有效证据")
            values["status"] = payload.status or "confirmed"
        elif any(k in values for k in ("statement", "owner", "due_at", "evidence_ids", "kind")):
            values["review_state"] = payload.review_state or "pending"
        if payload.evidence_ids is not None:
            valid_ids = {e["id"] for e in db.all("SELECT id FROM evidence WHERE meeting_id=?", (fact["meeting_id"],))}
            if not set(payload.evidence_ids).issubset(valid_ids):
                raise HTTPException(422, "引用不属于本会议")
        mapping = {"participants": "participants_json", "evidence_ids": "evidence_ids_json"}
        updates: dict[str, Any] = {}
        for key, value in values.items():
            target = mapping.get(key, key)
            updates[target] = db.json(value) if key in mapping else int(value) if key == "user_locked" else value
        updates["updated_at"] = now_iso()
        sql = ", ".join(f"{key}=?" for key in updates)
        with db.transaction():
            remember(fact["meeting_id"], "fact", fact)
            db.execute(f"UPDATE facts SET {sql} WHERE id=?", (*updates.values(), fact_id))
            db.execute("DELETE FROM facts_fts WHERE fact_id=?", (fact_id,))
            db.execute("INSERT INTO facts_fts(fact_id,meeting_id,statement) VALUES(?,?,?)", (fact_id, fact["meeting_id"], updates.get("statement",fact["statement"])))
            invalidate_reports(fact["meeting_id"])
        return _json_fields([db.one("SELECT * FROM facts WHERE id=?", (fact_id,)) or {}])[0]

    @router.patch("/media-links/{link_id}")
    def patch_link(link_id: str, payload: MediaLinkPatch) -> dict[str, Any]:
        if payload.confirmed and payload.rejected:
            raise HTTPException(422, "关联不能同时确认和拒绝")
        if not db.one("SELECT id FROM media_links WHERE id=?", (link_id,)):
            raise HTTPException(404, "图片关联不存在")
        db.execute(
            "UPDATE media_links SET confirmed=?, rejected=?, updated_at=? WHERE id=?",
            (int(payload.confirmed), int(payload.rejected), now_iso(), link_id),
        )
        return db.one("SELECT * FROM media_links WHERE id=?", (link_id,)) or {}

    @router.patch("/reports/{report_id}")
    def patch_report(report_id: str, payload: ReportPatch) -> dict[str, Any]:
        report = db.one("SELECT * FROM reports WHERE id=?", (report_id,))
        if not report:
            raise HTTPException(404, "报告章节不存在")
        content = payload.content_markdown
        if report["section"] == "article":
            content_html = PipelineRunner._editorial_html(content)
            evidence_ids = list(dict.fromkeys(re.findall(r"\[证据:([0-9a-fA-F]{8,32})\]", content)))
        else:
            content_html = "".join(f"<p>{html.escape(line)}</p>" for line in content.splitlines() if line.strip())
            evidence_ids = None
        with db.transaction():
            latest = db.one("SELECT MAX(version) AS version FROM reports WHERE meeting_id=?", (report["meeting_id"],))
            if latest["version"] != report["version"]:
                raise HTTPException(409, "报告已有新版本，请刷新后编辑")
            if payload.expected_updated_at and payload.expected_updated_at != report["updated_at"]:
                raise HTTPException(409, "报告已被修改，请刷新后编辑")
            rows = db.all("SELECT * FROM reports WHERE meeting_id=? AND version=?", (report["meeting_id"], report["version"]))
            result_id = None
            for row in rows:
                row.pop("id")
                row["version"] = report["version"] + 1
                row["created_at"] = row["updated_at"] = now_iso()
                row["needs_review"] = 1
                if row["section"] == report["section"]:
                    row.update(content_markdown=content, content_html=content_html,
                               evidence_ids_json=db.json(evidence_ids) if evidence_ids is not None else row["evidence_ids_json"],
                               editorial_state="user_edited")
                new_id = db.insert("reports", row)
                if row["section"] == report["section"]:
                    result_id = new_id
        return db.one("SELECT * FROM reports WHERE id=?", (result_id,)) or {}

    @router.post("/meetings/{meeting_id}/reports/approve")
    def approve_reports(meeting_id: str, payload: ReportApproval) -> dict[str, Any]:
        if not payload.confirm_semantic_review:
            raise HTTPException(422, "需确认已核对正文、原始证据及待核实项")
        with db.transaction():
            latest = db.one("SELECT MAX(version) AS version FROM reports WHERE meeting_id=?", (meeting_id,))
            if not latest or latest["version"] != payload.version:
                raise HTTPException(409, "请审核最新报告版本")
            rows = db.all("SELECT * FROM reports WHERE meeting_id=? AND version=?", (meeting_id,payload.version))
            if {r["section"] for r in rows} != set(REPORT_SECTIONS):
                raise HTTPException(409, "报告章节不完整")
            from .report_quality import audit_detailed_manuscript
            evidence = db.all("SELECT * FROM evidence WHERE meeting_id=?", (meeting_id,))
            article = next(r for r in rows if r["section"] == "article")
            check = audit_detailed_manuscript(article["content_markdown"], evidence=evidence, editorial_state="automatic_model_draft")
            if not check.passed:
                raise HTTPException(409, "正文证据检查未通过，请先处理：" + "；".join(check.errors))
            for row in rows:
                remember(meeting_id, "report_approval", row)
            db.execute("UPDATE reports SET editorial_state='human_reviewed', needs_review=0 WHERE meeting_id=? AND version=?", (meeting_id,payload.version))
        return {"version":payload.version,"publication_readiness":"human_reviewed"}

    @router.post("/meetings/{meeting_id}/reports/{section}/regenerate")
    def regenerate(meeting_id: str, section: str) -> dict[str, Any]:
        if section not in REPORT_SECTIONS:
            raise HTTPException(422, "未知报告章节")
        try:
            return context.pipeline.regenerate_section(meeting_id, section)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/meetings/{meeting_id}/export")
    def export_report(
        meeting_id: str,
        format: str = Query(pattern="^(docx|pdf)$"),
        detail: str | None = Query(default=None, pattern="^(BRIEF|STANDARD|DETAILED)$"),
        purpose: str = Query(default="draft", pattern="^(draft|final)$"),
    ) -> FileResponse:
        latest = db.one("SELECT MAX(version) AS version FROM reports WHERE meeting_id=?", (meeting_id,))
        unreviewed = db.one("SELECT id FROM reports WHERE meeting_id=? AND version=? AND (needs_review=1 OR editorial_state<>'human_reviewed') LIMIT 1", (meeting_id,latest["version"] if latest else None))
        if purpose == "final" and (not latest or latest["version"] is None or unreviewed):
            raise HTTPException(409, "当前报告仍是草稿，请完成逐项核对并确认发布")
        try:
            resolved_detail = context.exporter.detail_for(meeting_id, detail)
            path = (
                context.exporter.docx(meeting_id, detail=resolved_detail)
                if format == "docx"
                else context.exporter.pdf(meeting_id, detail=resolved_detail)
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if format == "docx" else "application/pdf"
        return FileResponse(
            path,
            media_type=media_type,
            filename=path.name,
            headers={
                "X-Report-Quality": "passed",
                "X-Publication-State": "draft" if unreviewed else "human_reviewed",
                "X-Report-Detail": resolved_detail,
            },
        )

    return router
