from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


MeetingMode = Literal["LOCAL_ONLY", "HYBRID_REVIEW"]
ReportDetail = Literal["BRIEF", "STANDARD", "DETAILED"]
FactKind = Literal["topic", "decision", "action", "risk", "question", "metric", "term"]


class MeetingCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    mode: MeetingMode = "LOCAL_ONLY"
    report_detail: ReportDetail = "STANDARD"
    glossary: str = Field(default="", max_length=20_000)


class MeetingPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    report_detail: ReportDetail | None = None


class SegmentPatch(BaseModel):
    user_text: str | None = Field(default=None, max_length=20_000)
    speaker_id: str | None = None
    speaker_name: str | None = Field(default=None, max_length=100)
    needs_review: bool | None = None


class FactPatch(BaseModel):
    kind: FactKind | None = None
    statement: str | None = Field(default=None, max_length=20_000)
    participants: list[str] | None = None
    owner: str | None = Field(default=None, max_length=200)
    due_at: str | None = None
    status: str | None = Field(default=None, max_length=100)
    evidence_ids: list[str] | None = None
    user_locked: bool = True
    review_state: Literal['pending', 'accepted', 'rejected'] | None = None


class MediaLinkPatch(BaseModel):
    confirmed: bool = False
    rejected: bool = False


class ReportPatch(BaseModel):
    content_markdown: str = Field(max_length=200_000)
    expected_updated_at: str | None = None


class ReportApproval(BaseModel):
    version: int = Field(ge=1)
    confirm_semantic_review: bool


class FactCandidate(BaseModel):
    kind: FactKind
    statement: str
    participants: list[str] = []
    owner: str | None = None
    due_at: str | None = None
    status: str | None = None
    evidence_ids: list[str] = []
    confidence: float = Field(default=0.6, ge=0, le=1)


class FactBatch(BaseModel):
    facts: list[FactCandidate] = []
