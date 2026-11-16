from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('LOCAL_ONLY', 'HYBRID_REVIEW')),
    report_detail TEXT NOT NULL DEFAULT 'STANDARD' CHECK (report_detail IN ('BRIEF', 'STANDARD', 'DETAILED')),
    glossary TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'DRAFT',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    filename TEXT NOT NULL,
    content_type TEXT,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS speakers (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    display_name TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transcript_segments (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    asset_id TEXT REFERENCES assets(id) ON DELETE SET NULL,
    speaker_id TEXT REFERENCES speakers(id) ON DELETE SET NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    corrected_text TEXT,
    user_text TEXT,
    confidence REAL NOT NULL DEFAULT 0,
    needs_review INTEGER NOT NULL DEFAULT 0,
    source_backend TEXT NOT NULL DEFAULT 'local',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    asset_id TEXT REFERENCES assets(id) ON DELETE SET NULL,
    segment_id TEXT REFERENCES transcript_segments(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    locator_json TEXT NOT NULL DEFAULT '{}',
    text TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0,
    source_backend TEXT NOT NULL DEFAULT 'local',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    statement TEXT NOT NULL,
    participants_json TEXT NOT NULL DEFAULT '[]',
    owner TEXT,
    due_at TEXT,
    status TEXT,
    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0,
    contradiction_ids_json TEXT NOT NULL DEFAULT '[]',
    source_backend TEXT NOT NULL DEFAULT 'local',
    user_locked INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS media_links (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    fact_id TEXT REFERENCES facts(id) ON DELETE CASCADE,
    score REAL NOT NULL DEFAULT 0,
    confirmed INTEGER NOT NULL DEFAULT 0,
    rejected INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    section TEXT NOT NULL,
    content_markdown TEXT NOT NULL,
    content_html TEXT NOT NULL,
    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(meeting_id, version, section)
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    current_stage TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    progress REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cloud_audit (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    operation TEXT NOT NULL,
    payload_summary TEXT NOT NULL,
    provider TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS export_audits (
    id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    format TEXT NOT NULL CHECK (format IN ('docx', 'pdf')),
    status TEXT NOT NULL CHECK (status IN ('PASSED', 'FAILED')),
    path TEXT,
    checks_json TEXT NOT NULL DEFAULT '{}',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    errors_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS assets_meeting_idx ON assets(meeting_id);
CREATE INDEX IF NOT EXISTS segments_meeting_time_idx ON transcript_segments(meeting_id, start_ms);
CREATE INDEX IF NOT EXISTS facts_meeting_kind_idx ON facts(meeting_id, kind);
CREATE INDEX IF NOT EXISTS reports_meeting_idx ON reports(meeting_id, version);
CREATE INDEX IF NOT EXISTS job_events_job_idx ON job_events(job_id, id);
CREATE INDEX IF NOT EXISTS export_audits_meeting_idx ON export_audits(meeting_id, created_at);
CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts USING fts5(
    segment_id UNINDEXED, meeting_id UNINDEXED, text, tokenize='unicode61'
);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    fact_id UNINDEXED, meeting_id UNINDEXED, statement, tokenize='unicode61'
);
"""


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


class Database:
    def __init__(self, path: Path):
        self.path = path
        self._init_lock = threading.Lock()
        self._initialized = False
        self._local = threading.local()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """One atomic write unit, shared by nested helpers on this thread."""
        active = getattr(self._local, "connection", None)
        if active is not None:
            # Savepoints allow callers to catch a nested error without committing
            # a half-written nested operation.
            name = "sp_" + uuid.uuid4().hex
            active.execute(f"SAVEPOINT {name}")
            try:
                yield active
                active.execute(f"RELEASE SAVEPOINT {name}")
            except BaseException:
                active.execute(f"ROLLBACK TO SAVEPOINT {name}")
                active.execute(f"RELEASE SAVEPOINT {name}")
                raise
            return
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._local.connection = conn
            try:
                yield conn
            finally:
                self._local.connection = None

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        active = getattr(self._local, "connection", None)
        if active is not None:
            yield active
            return
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self._init_lock:
            if self._initialized:
                return
            with self.connect() as conn:
                conn.executescript(SCHEMA)
                columns = {row[1] for row in conn.execute("PRAGMA table_info(meetings)")}
                if "report_detail" not in columns:
                    conn.execute(
                        "ALTER TABLE meetings ADD COLUMN report_detail TEXT NOT NULL DEFAULT 'STANDARD'"
                    )
                additions = {
                    "facts": {"review_state": "TEXT NOT NULL DEFAULT 'pending'"},
                    "reports": {
                        "editorial_state": "TEXT NOT NULL DEFAULT 'automatic_model_draft'",
                        "needs_review": "INTEGER NOT NULL DEFAULT 1",
                    },
                    "evidence": {"revision": "INTEGER NOT NULL DEFAULT 1"},
                }
                for table, fields in additions.items():
                    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                    for field, definition in fields.items():
                        if field not in existing:
                            conn.execute(f"ALTER TABLE {table} ADD COLUMN {field} {definition}")
                conn.execute("""CREATE TABLE IF NOT EXISTS revision_history (
                    id TEXT PRIMARY KEY, meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
                    entity_type TEXT NOT NULL, entity_id TEXT NOT NULL,
                    previous_json TEXT NOT NULL, created_at TEXT NOT NULL
                )""")
            self._initialized = True

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self.connect() as conn:
            conn.execute(sql, params)

    def execute_many(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        with self.connect() as conn:
            conn.executemany(sql, rows)

    def one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def insert(self, table: str, values: dict[str, Any]) -> str:
        values = dict(values)
        values.setdefault("id", new_id())
        keys = list(values)
        sql = f"INSERT INTO {table} ({', '.join(keys)}) VALUES ({', '.join('?' for _ in keys)})"
        self.execute(sql, tuple(values[key] for key in keys))
        return str(values["id"])

    @staticmethod
    def json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def decode_rows(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
        for row in rows:
            for field in fields:
                if field in row and isinstance(row[field], str):
                    try:
                        row[field.removesuffix("_json")] = json.loads(row[field])
                    except json.JSONDecodeError:
                        row[field.removesuffix("_json")] = None
        return rows


def classify_asset(filename: str, content_type: str | None) -> str:
    suffix = Path(filename).suffix.lower()
    if content_type and content_type.startswith("audio/") or suffix in {
        ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".mp4",
    }:
        return "audio"
    if content_type and content_type.startswith("image/") or suffix in {
        ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff",
    }:
        return "image"
    if suffix in {".txt", ".md", ".srt", ".vtt"}:
        return "transcript"
    return "other"
