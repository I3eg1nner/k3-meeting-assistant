from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from meeting_assistant.db import Database, now_iso
from meeting_assistant.inference import InferenceRegistry
from meeting_assistant.pipeline import PipelineRunner
from .test_pipeline import settings_for


def setup_runner(tmp_path: Path, *, asr: bool = False, hybrid: bool = False):
    settings = replace(settings_for(tmp_path), asr_command="local-test {input}" if asr else None)
    settings.ensure_directories()
    db = Database(settings.db_path)
    db.initialize()
    mid = db.insert("meetings", {
        "title": "恢复验收", "mode": "HYBRID_REVIEW" if hybrid else "LOCAL_ONLY",
        "created_at": now_iso(), "updated_at": now_iso(),
    })
    return PipelineRunner(settings, db, InferenceRegistry(settings)), mid


def asset(runner, mid, name="part.wav", kind="audio"):
    return runner.db.insert("assets", {
        "meeting_id": mid, "kind": kind, "filename": name, "path": "/not-read/" + name,
        "sha256": name, "size_bytes": 1, "metadata_json": json.dumps({"normalized_path": "/not-read/" + name}),
        "created_at": now_iso(),
    })


def fact(runner, mid, eid, *, confidence=.8, review="pending", statement="预算 100 万元"):
    return runner.db.insert("facts", {
        "meeting_id": mid, "kind": "decision", "statement": statement,
        "evidence_ids_json": json.dumps([eid]), "confidence": confidence,
        "review_state": review, "created_at": now_iso(), "updated_at": now_iso(),
    })


def evidence(runner, mid, aid, text="预算 100 万元", kind="audio"):
    return runner.db.insert("evidence", {
        "meeting_id": mid, "asset_id": aid, "kind": kind, "text": text,
        "confidence": .8, "created_at": now_iso(),
    })


def test_asr_recovers_each_asset_and_preserves_completed_ids(tmp_path, monkeypatch):
    runner, mid = setup_runner(tmp_path, asr=True)
    first, second = asset(runner, mid, "first.wav"), asset(runner, mid, "second.wav")
    calls = []
    def recognize(_template, **paths):
        calls.append(str(paths["input"]))
        if len(calls) == 2:
            raise RuntimeError("second asset interrupted")
        return {"segments": [{"start": 0, "end": 2, "text": "预算 100 万元"}]}
    monkeypatch.setattr(runner, "_run_json_command", recognize)
    with pytest.raises(RuntimeError):
        runner._asr(mid, "test")
    existing = runner.db.one("SELECT id FROM transcript_segments WHERE asset_id=?", (first,))
    assert existing
    assert not runner.db.one("SELECT id FROM transcript_segments WHERE asset_id=?", (second,))
    assert not runner._asr(mid, "retry").degraded
    assert runner.db.one("SELECT id FROM transcript_segments WHERE asset_id=?", (first,)) == existing
    assert len(calls) == 3 and calls.count("/not-read/first.wav") == 1
    runner._asr(mid, "repeat")
    assert len(calls) == 3


def test_asr_segment_evidence_fts_and_completion_commit_atomically(tmp_path, monkeypatch):
    runner, mid = setup_runner(tmp_path, asr=True)
    aid = asset(runner, mid)
    monkeypatch.setattr(runner, "_run_json_command", lambda *a, **kw: {"segments": [{"text": "有效内容"}]})
    original = runner.db.insert
    def interrupted(table, values):
        if table == "evidence":
            raise RuntimeError("power loss before evidence")
        return original(table, values)
    monkeypatch.setattr(runner.db, "insert", interrupted)
    with pytest.raises(RuntimeError):
        runner._asr(mid, "test")
    for table in ("transcript_segments", "evidence", "transcript_fts"):
        assert not runner.db.all(f"SELECT * FROM {table}")
    metadata = json.loads(runner.db.one("SELECT metadata_json FROM assets WHERE id=?", (aid,))["metadata_json"])
    assert not metadata.get("asr_complete")


def test_failed_segments_retry_without_duplication(tmp_path, monkeypatch):
    runner, mid = setup_runner(tmp_path, asr=True)
    aid = asset(runner, mid)
    responses = iter([
        {"segments": [{"text": "第一句"}], "failed_segments": [{"start": 5, "end": 10}]},
        {"segments": [{"text": "第一句"}, {"text": "第二句"}]},
    ])
    monkeypatch.setattr(runner, "_run_json_command", lambda *a, **kw: next(responses))
    assert runner._asr(mid, "first").degraded
    assert not runner._asr(mid, "retry").degraded
    assert len(runner.db.all("SELECT * FROM transcript_segments WHERE asset_id=?", (aid,))) == 2
    assert len(runner.db.all("SELECT * FROM evidence WHERE asset_id=?", (aid,))) == 2


def test_partial_asr_with_user_edits_is_not_silently_declared_complete(tmp_path, monkeypatch):
    runner, mid = setup_runner(tmp_path, asr=True)
    aid = asset(runner, mid)
    runner._insert_segments(mid, aid, [{"text": "机器原稿"}], "local-asr")
    runner.db.execute("UPDATE transcript_segments SET user_text='人工修订' WHERE asset_id=?", (aid,))
    monkeypatch.setattr(runner, "_run_json_command", lambda *a, **kw: pytest.fail("must preserve user work"))
    assert runner._asr(mid, "retry").degraded
    row = runner.db.one("SELECT raw_text,user_text FROM transcript_segments WHERE asset_id=?", (aid,))
    assert row == {"raw_text": "机器原稿", "user_text": "人工修订"}


def test_hybrid_never_implicitly_uploads_complete_audio_or_image(tmp_path, monkeypatch):
    runner, mid = setup_runner(tmp_path, hybrid=True)
    aid = asset(runner, mid)
    iid = asset(runner, mid, "slide.jpg", "image")
    monkeypatch.setattr(runner.inference, "cloud_backend", lambda: pytest.fail("implicit external request"))
    assert runner._asr(mid, "test").degraded
    assert runner._ocr_vlm(mid, "test").degraded
    for ident in (aid, iid):
        meta = json.loads(runner.db.one("SELECT metadata_json FROM assets WHERE id=?", (ident,))["metadata_json"])
        assert meta["cloud_review_required"]
    assert not runner.db.all("SELECT * FROM cloud_audit")


@pytest.mark.parametrize("regenerate", [False, True])
def test_report_publication_rolls_back_partial_version(tmp_path, monkeypatch, regenerate):
    runner, mid = setup_runner(tmp_path)
    runner._report(mid, "initial")
    original = runner.db.insert
    count = 0
    def interrupted(table, values):
        nonlocal count
        if table == "reports":
            count += 1
            if count == 2:
                raise RuntimeError("power loss while publishing")
        return original(table, values)
    monkeypatch.setattr(runner.db, "insert", interrupted)
    with pytest.raises(RuntimeError):
        if regenerate:
            runner.regenerate_section(mid, "summary")
        else:
            runner._report(mid, "next")
    assert runner.db.all("SELECT version,COUNT(*) AS n FROM reports GROUP BY version") == [{"version": 1, "n": 9}]


def test_regenerate_shares_global_pipeline_lock(tmp_path):
    runner, mid = setup_runner(tmp_path)
    other = PipelineRunner(runner.settings, runner.db, runner.inference)
    assert runner._run_lock.acquire(False)
    try:
        with pytest.raises(ValueError, match="任务运行"):
            other.regenerate_section(mid, "summary")
    finally:
        runner._run_lock.release()


@pytest.mark.parametrize("choice", ["confirmed", "rejected"])
def test_manual_media_choice_survives_fact_reextraction(tmp_path, choice):
    runner, mid = setup_runner(tmp_path)
    aid = asset(runner, mid, "slide.jpg", "image")
    eid = evidence(runner, mid, aid, kind="image")
    fid = fact(runner, mid, eid)
    lid = runner.db.insert("media_links", {
        "meeting_id": mid, "asset_id": aid, "fact_id": fid, choice: 1,
        "created_at": now_iso(), "updated_at": now_iso(),
    })
    runner._fact_extract(mid, "repeat")
    runner._link(mid, "repeat")
    saved = runner.db.one("SELECT * FROM media_links WHERE id=?", (lid,))
    assert saved and saved[choice] == 1 and saved["fact_id"] == fid


def test_editorial_sources_cannot_promote_unreviewed_image_candidates(tmp_path):
    runner, mid = setup_runner(tmp_path)
    aid = asset(runner, mid, "slide.jpg", "image")
    eid = evidence(runner, mid, aid, "可靠标题\n[视觉理解候选]\n吞吐量提高 99 倍", "image")
    fid = fact(runner, mid, eid, confidence=.5, statement="可靠标题")
    rows = runner.db.all("SELECT * FROM facts")
    assert runner._editorial_sources(mid, rows) == []
    runner.db.execute("UPDATE facts SET review_state='accepted' WHERE id=?", (fid,))
    rows = runner.db.all("SELECT * FROM facts")
    sources = runner._editorial_sources(mid, rows)
    assert len(sources) == 1 and sources[0]["text"] == "可靠标题"
    assert "99" not in sources[0]["text"]


def test_verify_invalidates_old_article_even_for_accepted_fact(tmp_path):
    runner, mid = setup_runner(tmp_path)
    aid = asset(runner, mid)
    eid = evidence(runner, mid, aid)
    fact(runner, mid, eid, confidence=.4, review="accepted", statement="预算 999 万元")
    runner._report(mid, "initial")
    runner.db.execute("UPDATE reports SET content_markdown='错误正文预算999万元', editorial_state='user_reviewed', needs_review=0 WHERE section='article'")
    assert runner._verify(mid, "verify").degraded
    latest = runner.db.one("SELECT * FROM reports WHERE section='article' ORDER BY version DESC LIMIT 1")
    assert latest["editorial_state"] == "invalidated" and latest["needs_review"] == 1
    assert "999" not in latest["content_markdown"]
    assert "已失效" in latest["content_markdown"]
