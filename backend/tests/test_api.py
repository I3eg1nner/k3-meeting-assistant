from __future__ import annotations

import io
import zipfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from meeting_assistant.api import ApiContext, build_router
from meeting_assistant.config import Settings
from meeting_assistant.db import Database
from meeting_assistant.inference import InferenceRegistry
from meeting_assistant.pipeline import PipelineRunner

from .test_pipeline import settings_for


def client_for(tmp_path: Path) -> TestClient:
    settings: Settings = settings_for(tmp_path)
    settings.ensure_directories()
    db = Database(settings.db_path)
    db.initialize()
    pipeline = PipelineRunner(settings, db, InferenceRegistry(settings))
    app = FastAPI()
    app.include_router(build_router(ApiContext(settings, db, pipeline)))
    return TestClient(app)


def test_create_upload_process_and_edit(tmp_path: Path):
    client = client_for(tmp_path)
    created = client.post(
        "/api/meetings",
        json={"title": "接口测试会议", "mode": "LOCAL_ONLY", "glossary": "K3"},
    )
    assert created.status_code == 201
    assert created.json()["report_detail"] == "STANDARD"
    meeting_id = created.json()["id"]

    preference = client.patch(
        f"/api/meetings/{meeting_id}", json={"report_detail": "DETAILED"}
    )
    assert preference.status_code == 200
    assert preference.json()["report_detail"] == "DETAILED"

    upload = client.post(
        f"/api/meetings/{meeting_id}/assets",
        files={"files": ("meeting.txt", "会议决定使用 K3。\n李工负责完成验证。", "text/plain")},
    )
    assert upload.status_code == 201

    processed = client.post(f"/api/meetings/{meeting_id}/process")
    assert processed.status_code == 202
    detail = client.get(f"/api/meetings/{meeting_id}").json()
    assert detail["jobs"][0]["status"] == "DONE"
    assert detail["segments"]
    assert detail["reports"]

    segment = detail["segments"][0]
    patched = client.patch(
        f"/api/transcript-segments/{segment['id']}",
        json={"user_text": "会议决定在 K3 上运行。", "needs_review": False},
    )
    assert patched.status_code == 200
    assert patched.json()["user_text"] == "会议决定在 K3 上运行。"

    exported = client.get(
        f"/api/meetings/{meeting_id}/export?format=docx&detail=DETAILED"
    )
    assert exported.status_code == 200
    assert exported.headers["x-report-quality"] == "passed"
    assert exported.headers["x-report-detail"] == "DETAILED"
    assert exported.content.startswith(b"PK")
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    # Without a model or accepted facts, a sparse input must not become a
    # fabricated technical article merely because DETAILED was selected.
    assert "议题与逻辑脉络" in document_xml
    assert "草稿 · 待人工核对" in document_xml
    assert exported.headers["x-publication-state"] == "draft"
    assert client.get(f"/api/meetings/{meeting_id}/export?format=docx&purpose=final").status_code == 409
    assert "正式事实与证据附录" in document_xml
    audited = client.get(f"/api/meetings/{meeting_id}").json()["export_audits"]
    assert audited and audited[0]["status"] == "PASSED"
    assert audited[0]["checks"]["required_sections"] is True
    assert audited[0]["metrics"]["report_detail"] == "DETAILED"


def test_optional_token_auth(tmp_path: Path):
    settings = settings_for(tmp_path)
    object.__setattr__(settings, "api_token", "secret")
    settings.ensure_directories()
    db = Database(settings.db_path)
    db.initialize()
    app = FastAPI()
    app.include_router(build_router(ApiContext(settings, db, PipelineRunner(settings, db, InferenceRegistry(settings)))))
    client = TestClient(app)
    assert client.get("/api/health").status_code == 401
    assert client.get("/api/health", headers={"X-Meeting-Token": "secret"}).status_code == 200


def test_meeting_title_can_be_corrected_after_creation(tmp_path: Path):
    client = client_for(tmp_path)
    meeting = client.post("/api/meetings", json={"title": "临时标题"}).json()

    response = client.patch(
        f"/api/meetings/{meeting['id']}",
        json={"title": "Synopsys HAV 技术开放日 2026"},
    )

    assert response.status_code == 200
    assert response.json()["title"] == "Synopsys HAV 技术开放日 2026"
