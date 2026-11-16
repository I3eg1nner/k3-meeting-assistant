from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import shlex
import shutil
import threading
from collections import Counter
from difflib import SequenceMatcher
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from .config import Settings
from .db import Database, new_id, now_iso
from .inference import InferenceRegistry, run_guarded_command
from .schemas import FactBatch


LOGGER = logging.getLogger(__name__)

STAGES = [
    "INGEST",
    "AUDIO_PREP",
    "DIARIZE",
    "ASR",
    "ALIGN",
    "IMAGE_PREP",
    "OCR_VLM",
    "FACT_EXTRACT",
    "LINK",
    "REPORT",
    "VERIFY",
]

STAGE_PROGRESS = {stage: (index + 1) / (len(STAGES) + 1) for index, stage in enumerate(STAGES)}
REPORT_SECTIONS = ["summary", "article", "topics", "decisions", "actions", "risks", "questions", "media", "review"]


def _atomic(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.db.transaction():
            return method(self, *args, **kwargs)
    return wrapped


def _formal_fact(fact: dict[str, Any]) -> bool:
    review = fact.get("review_state", "pending")
    return (
        review not in {"rejected", "stale"}
        and fact.get("status") != "needs_review"
        and (review == "accepted" or float(fact.get("confidence") or 0) >= 0.55)
    )


def _effective_text(row: dict[str, Any]) -> str:
    return (row.get("user_text") or row.get("corrected_text") or row.get("raw_text") or "").strip()


def _safe_sentence_split(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[。！？!?；;])\s*|\n+", text) if item.strip()]


def _apply_known_terms(
    text: str,
    canonical_terms: list[str],
    trusted_short_terms: set[str] | None = None,
) -> str:
    """Correct one-character ASR variants only when a user/title term is available."""
    tokens = set(
        re.findall(
            r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9.-]{1,15}(?![A-Za-z0-9])",
            text,
        )
    )
    for token in tokens:
        if not any(character.isupper() for character in token):
            continue
        for canonical in canonical_terms:
            if token.casefold() == canonical.casefold():
                if token != canonical:
                    text = re.sub(
                        rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])",
                        canonical,
                        text,
                    )
                break
            if len(token) == len(canonical) and SequenceMatcher(
                None, token.casefold(), canonical.casefold()
            ).ratio() >= 2 / 3:
                if len(token) <= 3 and trusted_short_terms is not None and canonical not in trusted_short_terms:
                    continue
                text = re.sub(
                    rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])",
                    canonical,
                    text,
                )
                break
    return text


def _canonical_terms(
    title: str,
    glossary: str,
    evidence: list[dict[str, Any]],
) -> list[str]:
    pattern = r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9.-]{1,15}(?![A-Za-z0-9])"
    explicit = re.findall(pattern, f"{title}\n{glossary}")
    image_tokens = Counter(
        token
        for item in evidence
        if item.get("kind") == "image"
        for token in re.findall(
            pattern,
            str(item.get("text") or "").split("[视觉理解候选]", 1)[-1]
            if "[视觉理解候选]" in str(item.get("text") or "")
            else "",
        )
        if sum(character.isupper() for character in token) >= 2
    )
    corroborated = [token for token, count in image_tokens.most_common() if count >= 2]
    return list(dict.fromkeys(explicit + corroborated))


def _timestamp_seconds(value: str) -> int:
    parts = [int(item) for item in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"不支持的时间格式：{value}")


def _parse_feishu_transcript(text: str) -> list[dict[str, Any]]:
    """Parse Feishu's plain-text transcript while preserving speaker turns and time."""
    header = re.compile(
        r"^说话人\s+(?P<speaker>\S+)\s+(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\s*$"
    )
    total_match = re.search(r"(?:(\d+)小时)?\s*(\d+)分钟\s*(\d+)秒", text[:300])
    total_seconds = None
    if total_match:
        total_seconds = (
            int(total_match.group(1) or 0) * 3600
            + int(total_match.group(2)) * 60
            + int(total_match.group(3))
        )

    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    body: list[str] = []
    for raw_line in text.splitlines():
        match = header.match(raw_line.strip())
        if match:
            if current and body:
                current["text"] = "\n".join(line for line in body if line).strip()
                if current["text"]:
                    turns.append(current)
            current = {
                "speaker": f"Speaker {match.group('speaker')}",
                "start": _timestamp_seconds(match.group("time")),
                "confidence": 0.85,
            }
            body = []
        elif current is not None:
            body.append(raw_line.strip())
    if current and body:
        current["text"] = "\n".join(line for line in body if line).strip()
        if current["text"]:
            turns.append(current)

    for index, turn in enumerate(turns):
        if index + 1 < len(turns):
            turn["end"] = max(turn["start"] + 1, turns[index + 1]["start"])
        elif total_seconds is not None:
            turn["end"] = max(turn["start"] + 1, total_seconds)
        else:
            turn["end"] = turn["start"] + max(5, min(120, len(turn["text"]) // 4))
    return turns


def _normalize_fact_kind(value: str) -> str:
    aliases = {
        "议题": "topic", "主题": "topic", "结论": "decision", "决定": "decision",
        "行动": "action", "待办": "action", "风险": "risk", "问题": "question",
        "指标": "metric", "数字": "metric", "术语": "term",
    }
    value = value.strip().lower()
    return aliases.get(value, value if value in {"topic", "decision", "action", "risk", "question", "metric", "term"} else "topic")


def _evidence_support_score(statement: str, evidence_text: str) -> float:
    clean_statement = re.sub(r"[^\w\u4e00-\u9fff]+", "", statement.lower())
    clean_evidence = re.sub(r"[^\w\u4e00-\u9fff]+", "", evidence_text.lower())
    if len(clean_statement) < 2 or not clean_evidence:
        return 0.0
    grams = {clean_statement[index:index + 2] for index in range(len(clean_statement) - 1)}
    return sum(gram in clean_evidence for gram in grams) / max(1, len(grams))


def _lexical_tokens(text: str) -> set[str]:
    """Mixed Chinese/Latin tokens for deterministic linking and conflict checks."""
    lowered = text.lower()
    tokens = set(re.findall(r"[a-z0-9][a-z0-9_.+-]{1,}", lowered))
    for run in re.findall(r"[\u4e00-\u9fff]+", lowered):
        if len(run) == 1:
            tokens.add(run)
        else:
            tokens.update(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


# These words are useful in prose, but are too common in technical decks to prove
# that a particular image supports a particular fact.  Removing them prevents a
# generic "system/platform/technology" slide from becoming a seemingly strong
# link merely because a fact uses the same boilerplate.
_GENERIC_LINK_TOKENS = {
    "技术", "系统", "平台", "方案", "能力", "产品", "应用", "支持", "实现",
    "进行", "使用", "提供", "介绍", "会议", "内容", "核心", "相关", "基于",
    "system", "platform", "technology", "solution", "product", "support",
}


def _link_tokens(text: str) -> set[str]:
    return _lexical_tokens(text) - _GENERIC_LINK_TOKENS


def _link_entities(text: str) -> set[str]:
    """Return high-signal Latin terms and numbers used by media linking."""
    lowered = text.casefold()
    entities = set(re.findall(r"(?<![a-z0-9])[a-z][a-z0-9_.+-]{1,}(?![a-z0-9])", lowered))
    entities.update(re.findall(r"\d+(?:\.\d+)?(?:%|mhz|ghz|tb|gb|hrs?|小时)?", lowered))
    return entities - _GENERIC_LINK_TOKENS


def _focused_image_text(text: str) -> str:
    """Prefer a slide title and the VLM description without discarding OCR text."""
    raw = str(text or "")
    before, marker, after = raw.partition("[视觉理解候选]")
    title_lines = [
        line.strip()
        for line in before.splitlines()
        if re.match(r"^\s*(?:标题|title)\s*[：:]", line, re.IGNORECASE)
    ]
    visual = after if marker else raw
    return "\n".join(title_lines + [visual[:320]])


def _media_fact_link_score(image_text: str, fact_statement: str, *, direct_source: bool = False) -> float:
    """Score an image/fact candidate using containment, focus text and entities.

    Jaccard similarity unfairly penalizes a long OCR result paired with a short
    fact.  Coverage of the shorter fact is the relevant signal here.  An exact
    evidence citation creates a strong *candidate*, but confirmation remains a
    separate trust decision in ``_link``.
    """
    fact_tokens = _link_tokens(fact_statement)
    image_tokens = _link_tokens(image_text)
    focused_tokens = _link_tokens(_focused_image_text(image_text))
    if not fact_tokens or not image_tokens:
        return 0.72 if direct_source else 0.0

    lexical = len(fact_tokens & image_tokens) / max(1, len(fact_tokens))
    focused = len(fact_tokens & focused_tokens) / max(1, len(fact_tokens))
    fact_entities = _link_entities(fact_statement)
    image_entities = _link_entities(image_text)
    entity_coverage = (
        len(fact_entities & image_entities) / len(fact_entities)
        if fact_entities else 0.0
    )
    score = 0.52 * lexical + 0.23 * focused + 0.25 * entity_coverage

    # A mismatched number is much more important than a few matching topic words.
    fact_numbers = set(re.findall(r"\d+(?:\.\d+)?", fact_statement))
    image_numbers = set(re.findall(r"\d+(?:\.\d+)?", image_text))
    if fact_numbers and not fact_numbers.issubset(image_numbers):
        score *= 0.25
    if direct_source:
        score = max(score, 0.72)
    return round(min(1.0, score), 4)


@dataclass
class StageResult:
    message: str
    degraded: bool = False


class PipelineRunner:
    _global_run_lock = threading.Lock()

    def __init__(self, settings: Settings, db: Database, inference: InferenceRegistry):
        self.settings = settings
        self.db = db
        self.inference = inference
        self._run_lock = self._global_run_lock

    def event(self, job_id: str, stage: str, message: str, progress: float, level: str = "info") -> None:
        self.db.execute(
            "INSERT INTO job_events(job_id, stage, level, message, progress, created_at) VALUES(?,?,?,?,?,?)",
            (job_id, stage, level, message, progress, now_iso()),
        )
        self.db.execute(
            "UPDATE jobs SET current_stage=?, progress=?, updated_at=? WHERE id=?",
            (stage, progress, now_iso(), job_id),
        )

    def run(self, job_id: str) -> None:
        if not self._run_lock.acquire(blocking=False):
            self._fail(job_id, "另一个会议任务正在运行")
            return
        job = self.db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
        if not job:
            self._run_lock.release()
            return
        meeting_id = job["meeting_id"]
        try:
            timestamp = now_iso()
            self.db.execute(
                "UPDATE jobs SET status='RUNNING', started_at=COALESCE(started_at, ?), error=NULL, updated_at=? WHERE id=?",
                (timestamp, timestamp, job_id),
            )
            self.db.execute(
                "UPDATE meetings SET status='PROCESSING', updated_at=? WHERE id=?",
                (timestamp, meeting_id),
            )
            handlers: dict[str, Callable[[str, str], StageResult]] = {
                "INGEST": self._ingest,
                "AUDIO_PREP": self._audio_prep,
                "DIARIZE": self._diarize,
                "ASR": self._asr,
                "ALIGN": self._align,
                "IMAGE_PREP": self._image_prep,
                "OCR_VLM": self._ocr_vlm,
                "FACT_EXTRACT": self._fact_extract,
                "LINK": self._link,
                "REPORT": self._report,
                "VERIFY": self._verify,
            }
            stages = STAGES
            if job["status"] == "RUNNING" and job["current_stage"] in STAGES:
                resume_index = STAGES.index(job["current_stage"])
                stages = STAGES[resume_index:]
                self.event(
                    job_id,
                    job["current_stage"],
                    f"服务重启，从 {job['current_stage']} 阶段恢复",
                    float(job["progress"]),
                    "warning",
                )
            for stage in stages:
                self.event(job_id, stage, f"开始：{stage}", STAGE_PROGRESS[stage] - 0.03)
                result = handlers[stage](meeting_id, job_id)
                self.event(
                    job_id,
                    stage,
                    result.message,
                    STAGE_PROGRESS[stage],
                    "warning" if result.degraded else "info",
                )
            finished = now_iso()
            self.db.execute(
                "UPDATE jobs SET status='DONE', current_stage='DONE', progress=1, finished_at=?, updated_at=? WHERE id=?",
                (finished, finished, job_id),
            )
            self.db.execute(
                "UPDATE meetings SET status='DONE', updated_at=? WHERE id=?",
                (finished, meeting_id),
            )
            self.event(job_id, "DONE", "处理完成", 1)
        except Exception as exc:
            self._fail(job_id, str(exc))
            self.db.execute(
                "UPDATE meetings SET status='FAILED', updated_at=? WHERE id=?",
                (now_iso(), meeting_id),
            )
        finally:
            self.inference.model_manager.stop()
            self.inference.vlm_manager.stop()
            self._run_lock.release()

    def _fail(self, job_id: str, message: str) -> None:
        timestamp = now_iso()
        self.db.execute(
            "UPDATE jobs SET status='FAILED', error=?, finished_at=?, updated_at=? WHERE id=?",
            (message, timestamp, timestamp, job_id),
        )
        job = self.db.one("SELECT current_stage, progress FROM jobs WHERE id=?", (job_id,))
        if job:
            self.event(job_id, job["current_stage"], message, float(job["progress"]), "error")

    def _meeting_dir(self, meeting_id: str) -> Path:
        path = self.settings.data_dir / "meetings" / meeting_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _ingest(self, meeting_id: str, _job_id: str) -> StageResult:
        assets = self.db.all("SELECT * FROM assets WHERE meeting_id=?", (meeting_id,))
        if not assets:
            raise RuntimeError("会议没有上传任何材料")
        audio = [a for a in assets if a["kind"] == "audio"]
        images = [a for a in assets if a["kind"] == "image"]
        return StageResult(f"已校验 {len(audio)} 个音频、{len(images)} 张图片和 {len(assets)-len(audio)-len(images)} 个辅助文件")

    def _audio_prep(self, meeting_id: str, _job_id: str) -> StageResult:
        assets = self.db.all("SELECT * FROM assets WHERE meeting_id=? AND kind='audio'", (meeting_id,))
        if not assets:
            return StageResult("未上传音频，跳过音频预处理", True)
        derived = self._meeting_dir(meeting_id) / "derived" / "audio"
        derived.mkdir(parents=True, exist_ok=True)
        converted = 0
        for asset in assets:
            output = derived / f"{asset['id']}.wav"
            command = [
                "ffmpeg", "-y", "-v", "error", "-i", asset["path"],
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output),
            ]
            result = run_guarded_command(command, self.settings, timeout=3600)
            if result.returncode:
                raise RuntimeError(f"音频转换失败：{asset['filename']}：{result.stderr[-500:]}")
            metadata = json.loads(asset.get("metadata_json") or "{}")
            metadata["normalized_path"] = str(output)
            self.db.execute("UPDATE assets SET metadata_json=? WHERE id=?", (self.db.json(metadata), asset["id"]))
            converted += 1
        return StageResult(f"已转换 {converted} 个音频为 16kHz 单声道 PCM")

    def _run_json_command(self, template: str, **paths: Path | str) -> dict[str, Any]:
        rendered = template.format(**{key: shlex.quote(str(value)) for key, value in paths.items()})
        result = run_guarded_command(shlex.split(rendered), self.settings, timeout=3600)
        if result.returncode:
            raise RuntimeError(result.stderr[-1000:] or f"命令退出码 {result.returncode}")
        output_path = paths.get("output")
        if output_path and Path(str(output_path)).exists():
            return json.loads(Path(str(output_path)).read_text())
        return json.loads(result.stdout)

    def _ensure_default_speaker(self, meeting_id: str) -> str:
        speaker = self.db.one("SELECT id FROM speakers WHERE meeting_id=? ORDER BY created_at LIMIT 1", (meeting_id,))
        if speaker:
            return speaker["id"]
        return self.db.insert(
            "speakers",
            {"meeting_id": meeting_id, "label": "Speaker 1", "display_name": None, "created_at": now_iso()},
        )

    def _diarize(self, meeting_id: str, _job_id: str) -> StageResult:
        self._ensure_default_speaker(meeting_id)
        assets = self.db.all("SELECT * FROM assets WHERE meeting_id=? AND kind='audio'", (meeting_id,))
        if not assets:
            return StageResult("无音频，保留默认说话人", True)
        if not self.settings.diarization_command:
            return StageResult("未配置 CAM++ 分离模型，暂按 Speaker 1 处理，可在工作台人工改名", True)
        labels: set[str] = set()
        segment_count = 0
        for asset in assets:
            metadata = json.loads(asset["metadata_json"])
            output = self._meeting_dir(meeting_id) / "derived" / f"diarization-{asset['id']}.json"
            result = self._run_json_command(
                self.settings.diarization_command,
                input=metadata["normalized_path"],
                output=output,
            )
            segments = result.get("segments", [])
            metadata["diarization_segments"] = segments
            metadata["diarization_metrics"] = result.get("metrics", {})
            self.db.execute(
                "UPDATE assets SET metadata_json=? WHERE id=?",
                (self.db.json(metadata), asset["id"]),
            )
            segment_count += len(segments)
            labels.update(str(item.get("speaker", "Speaker 1")) for item in segments)
        for label in labels:
            if not self.db.one("SELECT id FROM speakers WHERE meeting_id=? AND label=?", (meeting_id, label)):
                self.db.insert("speakers", {"meeting_id": meeting_id, "label": label, "display_name": None, "created_at": now_iso()})
        return StageResult(f"已识别 {max(1, len(labels))} 个说话人、{segment_count} 个说话区间")

    @staticmethod
    def _assign_diarized_speakers(
        segments: list[dict[str, Any]], diarization: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Assign the speaker with the greatest temporal overlap to each ASR segment."""
        if not diarization:
            return segments
        assigned: list[dict[str, Any]] = []
        for segment in segments:
            start = float(segment.get("start", 0))
            end = float(segment.get("end", start))
            best_label = None
            best_overlap = 0.0
            for turn in diarization:
                turn_start = float(turn.get("start", turn.get("start_ms", 0)))
                turn_end = float(turn.get("end", turn.get("end_ms", turn_start)))
                if "start_ms" in turn:
                    turn_start /= 1000
                if "end_ms" in turn:
                    turn_end /= 1000
                overlap = max(0.0, min(end, turn_end) - max(start, turn_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label = str(turn.get("speaker") or "Speaker 1")
            item = dict(segment)
            if best_label is not None:
                item["speaker"] = best_label
            assigned.append(item)
        return assigned

    @_atomic
    def _insert_segments(self, meeting_id: str, asset_id: str | None, segments: list[dict[str, Any]], backend: str) -> int:
        speaker_map = {
            row["label"]: row["id"]
            for row in self.db.all("SELECT id, label FROM speakers WHERE meeting_id=?", (meeting_id,))
        }
        self._ensure_default_speaker(meeting_id)
        count = 0
        for index, segment in enumerate(segments):
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            raw_text = str(segment.get("raw_text") or text).strip()
            corrected_text = text if text != raw_text else None
            speaker_label = str(segment.get("speaker") or "Speaker 1")
            speaker_id = speaker_map.get(speaker_label)
            if not speaker_id:
                speaker_id = self.db.insert(
                    "speakers",
                    {
                        "meeting_id": meeting_id,
                        "label": speaker_label,
                        "display_name": None,
                        "created_at": now_iso(),
                    },
                )
                speaker_map[speaker_label] = speaker_id
            start_ms = int(float(segment.get("start_ms", segment.get("start", index * 30))) * (1 if "start_ms" in segment else 1000))
            end_ms = int(float(segment.get("end_ms", segment.get("end", (index + 1) * 30))) * (1 if "end_ms" in segment else 1000))
            confidence = float(segment.get("confidence", 0.75 if backend != "degraded" else 0.0))
            timestamp = now_iso()
            segment_id = self.db.insert(
                "transcript_segments",
                {
                    "meeting_id": meeting_id,
                    "asset_id": asset_id,
                    "speaker_id": speaker_id,
                    "start_ms": start_ms,
                    "end_ms": max(end_ms, start_ms + 1),
                    "raw_text": raw_text,
                    "corrected_text": corrected_text,
                    "user_text": None,
                    "confidence": confidence,
                    "needs_review": int(confidence < 0.55),
                    "source_backend": backend,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                },
            )
            evidence_id = self.db.insert(
                "evidence",
                {
                    "meeting_id": meeting_id,
                    "asset_id": asset_id,
                    "segment_id": segment_id,
                    "kind": "audio",
                    "locator_json": self.db.json({"start_ms": start_ms, "end_ms": max(end_ms, start_ms + 1)}),
                    "text": text,
                    "confidence": confidence,
                    "source_backend": backend,
                    "created_at": timestamp,
                },
            )
            self.db.execute("INSERT INTO transcript_fts(segment_id, meeting_id, text) VALUES(?,?,?)", (segment_id, meeting_id, text))
            count += 1
        return count

    def _asr(self, meeting_id: str, _job_id: str) -> StageResult:
        transcript_assets = self.db.all("SELECT * FROM assets WHERE meeting_id=? AND kind='transcript'", (meeting_id,))
        meeting = self.db.one("SELECT * FROM meetings WHERE id=?", (meeting_id,)) or {}
        audio_assets = self.db.all("SELECT * FROM assets WHERE meeting_id=? AND kind='audio'", (meeting_id,))
        # An uploaded transcript is the explicit textual source for this meeting.
        # Completion is nevertheless tracked per asset, so additional transcripts
        # and interrupted multi-audio jobs cannot be silently skipped.
        assets = transcript_assets or audio_assets
        if not assets:
            return StageResult("未上传音频或文字转写", True)
        total = 0
        degraded = False
        for asset in assets:
            metadata = json.loads(asset.get("metadata_json") or "{}")
            fingerprint = hashlib.sha256(
                (asset["sha256"] + "\n" + ("uploaded-transcript-v1" if asset["kind"] == "transcript" else self.settings.asr_command or "unconfigured")).encode()
            ).hexdigest()
            existing = self.db.all("SELECT * FROM transcript_segments WHERE asset_id=?", (asset["id"],))
            if metadata.get("asr_complete") and metadata.get("asr_fingerprint") == fingerprint:
                total += len(existing)
                continue
            protected_evidence = {
                evidence_id
                for fact in self.db.all(
                    "SELECT evidence_ids_json FROM facts WHERE meeting_id=? AND (user_locked=1 OR review_state='accepted' OR id IN (SELECT fact_id FROM media_links WHERE confirmed=1 OR rejected=1))",
                    (meeting_id,),
                )
                for evidence_id in json.loads(fact["evidence_ids_json"] or "[]")
            }
            asset_evidence = self.db.all("SELECT id FROM evidence WHERE asset_id=?", (asset["id"],))
            if existing and (
                any(row.get("user_text") is not None for row in existing)
                or any(row["id"] in protected_evidence for row in asset_evidence)
            ):
                metadata["asr_review_required"] = "未完成资产含人工修订或已确认引用，保留原结果；需人工确认覆盖范围后重新识别。"
                self.db.execute("UPDATE assets SET metadata_json=? WHERE id=?", (self.db.json(metadata), asset["id"]))
                total += len(existing)
                degraded = True
                continue
            result: dict[str, Any] | None = None
            backend = "local-asr"
            if asset["kind"] == "transcript":
                text = Path(asset["path"]).read_text(errors="replace")
                segments = _parse_feishu_transcript(text)
                if not segments:
                    segments = [{"start": i * 30, "end": (i + 1) * 30, "text": line} for i, line in enumerate(_safe_sentence_split(text))]
                result = {"segments": segments}
                backend = "uploaded-transcript"
            elif self.settings.asr_command:
                normalized = Path(metadata["normalized_path"])
                output = self._meeting_dir(meeting_id) / "derived" / f"asr-{asset['id']}.json"
                result = self._run_json_command(self.settings.asr_command, input=normalized, output=output)
            elif meeting.get("mode") == "HYBRID_REVIEW":
                metadata["cloud_review_required"] = "尚未实现经授权的局部音频复核，未发送整段音频。"
            if result is not None:
                segments = result.get("segments") or [{"start": 0, "end": 30, "text": result.get("text", "")}]
                segments = self._assign_diarized_speakers(
                    segments,
                    metadata.get("diarization_segments") or [],
                )
                failures = result.get("failed_segments") or []
                metadata["asr_failed_segments"] = failures
                metadata["asr_metrics"] = result.get("metrics") or {}
                metadata["vad"] = result.get("vad") or {}
                has_text = any(str(row.get("text") or "").strip() for row in segments)
                # Empty output needs an explicit silence result to count as complete.
                complete = not failures and (has_text or result.get("silence") is True)
                degraded = degraded or not complete
            else:
                degraded = True
                complete = False
                backend = "degraded"
                segments = [{
                    "start": 0,
                    "end": 1,
                    "text": f"[{asset['filename']}] 本地 ASR 尚未配置，需人工补录或配置模型；材料未外发。",
                    "confidence": 0,
                }]
            with self.db.transaction():
                self.db.execute("DELETE FROM transcript_fts WHERE segment_id IN (SELECT id FROM transcript_segments WHERE asset_id=?)", (asset["id"],))
                self.db.execute("DELETE FROM evidence WHERE asset_id=? AND segment_id IS NOT NULL", (asset["id"],))
                self.db.execute("DELETE FROM transcript_segments WHERE asset_id=?", (asset["id"],))
                total += self._insert_segments(meeting_id, asset["id"], segments, backend)
                metadata["asr_complete"] = complete
                metadata["asr_fingerprint"] = fingerprint
                self.db.execute("UPDATE assets SET metadata_json=? WHERE id=?", (self.db.json(metadata), asset["id"]))
        return StageResult(f"已保留 {total} 条带证据转写" + ("；存在未完成或待人工复核资产" if degraded else ""), degraded)

    def _align(self, meeting_id: str, _job_id: str) -> StageResult:
        segments = self.db.all("SELECT * FROM transcript_segments WHERE meeting_id=?", (meeting_id,))
        review = sum(1 for item in segments if item["needs_review"])
        return StageResult(f"已保留 {len(segments)} 条句段级时间戳，{review} 条需要复核", review > 0)

    def _image_prep(self, meeting_id: str, _job_id: str) -> StageResult:
        assets = self.db.all("SELECT * FROM assets WHERE meeting_id=? AND kind='image'", (meeting_id,))
        if not assets:
            return StageResult("未上传图片，跳过图片预处理")
        derived = self._meeting_dir(meeting_id) / "derived" / "images"
        derived.mkdir(parents=True, exist_ok=True)
        enhanced_count = 0
        for asset in assets:
            source = Path(asset["path"])
            output = derived / f"{asset['id']}.png"
            metadata = json.loads(asset["metadata_json"])
            try:
                from PIL import Image, ImageEnhance, ImageFilter, ImageOps

                with Image.open(source) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    image = ImageOps.autocontrast(image, cutoff=1)
                    image = ImageEnhance.Contrast(image).enhance(1.15)
                    image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=110, threshold=3))
                    image.save(output, format="PNG", optimize=True)
                metadata["enhanced_path"] = str(output)
                metadata["enhancement"] = "deterministic-autocontrast-unsharp"
                enhanced_count += 1
            except Exception as exc:
                shutil.copy2(source, output)
                metadata["enhanced_path"] = str(output)
                metadata["enhancement_error"] = str(exc)
            self.db.execute("UPDATE assets SET metadata_json=? WHERE id=?", (self.db.json(metadata), asset["id"]))
        return StageResult(f"已生成 {enhanced_count}/{len(assets)} 张确定性增强图，原图完整保留", enhanced_count < len(assets))

    def _ocr_vlm(self, meeting_id: str, _job_id: str) -> StageResult:
        assets = self.db.all("SELECT * FROM assets WHERE meeting_id=? AND kind='image'", (meeting_id,))
        if not assets:
            return StageResult("无图片，跳过 OCR/VLM")
        meeting = self.db.one("SELECT * FROM meetings WHERE id=?", (meeting_id,)) or {}
        locked_evidence_ids: set[str] = set()
        for fact in self.db.all("SELECT evidence_ids_json FROM facts WHERE meeting_id=? AND user_locked=1", (meeting_id,)):
            locked_evidence_ids.update(json.loads(fact["evidence_ids_json"] or "[]"))
        recognized = 0
        degraded = False
        work: list[dict[str, Any]] = []
        for asset in assets:
            metadata = json.loads(asset["metadata_json"])
            # The original upload is the evidentiary source.  The deterministic
            # enhancement is retained for human comparison, but real K3 tests
            # showed that it reduced key-line OCR recall (36.8% -> 31.6%).
            source = Path(asset["path"])
            enhanced = Path(metadata.get("enhanced_path") or asset["path"])
            ocr_result: dict[str, Any] = {"text": "", "regions": []}
            if self.settings.ocr_command:
                output = self._meeting_dir(meeting_id) / "derived" / f"ocr-{asset['id']}.json"
                ocr_result = self._run_json_command(self.settings.ocr_command, input=source, output=output)
            text = str(ocr_result.get("text") or "").strip()
            work.append(
                {
                    "asset": asset,
                    "source": source,
                    "enhanced": enhanced,
                    "ocr": ocr_result,
                    "text": text,
                    "vlm": {},
                }
            )

        vlm_ready = False
        vlm_errors: list[str] = []
        with self.inference.vlm_manager.session() as ready:
            vlm_ready = ready
            for item in work:
                asset = item["asset"]
                source = item["source"]
                text = item["text"]
                try:
                    if ready:
                        self.inference.vlm_manager.assert_resources()
                        item["vlm"] = self.inference.local_vision_backend().analyze_image(
                            source,
                            text,
                            (
                                "你在核对一张会议PPT照片。返回JSON对象，字段必须包含title、key_points、metrics、"
                                "chart_or_table、unreadable。只写图中直接可见的信息；OCR有明显错字时可纠正，"
                                "但不可猜测被遮挡或模糊内容；metrics只保留含阿拉伯数字、百分号或明确单位的可见原文，"
                                "无数字指标时返回空数组；无图表时chart_or_table返回空字符串；unreadable只列出无法辨认的"
                                "幻灯片文字/图表区域，不描述人物、会场或背景。key_points最多5项，metrics最多5项，"
                                "unreadable最多3项，每项不超过100字；不要重复，不要解释。"
                            ),
                        )
                        self.inference.vlm_manager.assert_resources()
                    elif self.settings.vlm_command:
                        output = self._meeting_dir(meeting_id) / "derived" / f"vlm-{asset['id']}.json"
                        item["vlm"] = self._run_json_command(
                            self.settings.vlm_command,
                            input=source,
                            output=output,
                            ocr=text,
                        )
                except Exception as exc:
                    vlm_errors.append(f"{asset['filename']}: {type(exc).__name__}: {exc}")

        for item in work:
            asset = item["asset"]
            ocr_result = item["ocr"]
            text = item["text"]
            vlm_result = item["vlm"]
            if not vlm_result and meeting.get("mode") == "HYBRID_REVIEW" and (not text or float(ocr_result.get("confidence", 0)) < 0.6):
                # HYBRID_REVIEW authorizes local-first review, not implicit full
                # image upload. Keep this pending until scoped consent exists.
                metadata = json.loads(asset.get("metadata_json") or "{}")
                metadata["cloud_review_required"] = "图片需复核；局部裁剪授权尚未实现，原图未外发。"
                self.db.execute("UPDATE assets SET metadata_json=? WHERE id=?", (self.db.json(metadata), asset["id"]))
                degraded = True
            semantic_parts: list[str] = []
            if vlm_result:
                title = vlm_result.get("title")
                if title:
                    semantic_parts.append(f"标题：{title}")
                key_points = vlm_result.get("key_points") or []
                if isinstance(key_points, list):
                    semantic_parts.extend(str(value) for value in key_points[:5] if value)
                metrics = vlm_result.get("metrics") or []
                if isinstance(metrics, list):
                    # Small VLMs sometimes fill a required metrics array with
                    # ordinary prose. Only numerically checkable values may become evidence.
                    semantic_parts.extend(
                        str(value) for value in metrics[:5]
                        if value and re.search(
                            r"\d|[%％]|(?:GB|MB|GHz|MHz|fps|帧|秒|分钟|小时|万元|亿元|台)",
                            str(value), re.I,
                        )
                    )
                chart = vlm_result.get("chart_or_table") or vlm_result.get("summary")
                if chart and str(chart).strip().lower() not in {
                    "text", "none", "n/a", "无", "无图表", "无表格",
                }:
                    semantic_parts.append(str(chart))
                unreadable = vlm_result.get("unreadable") or []
                if isinstance(unreadable, list):
                    relevant = [
                        str(value) for value in unreadable[:3]
                        if value and not re.search(
                            r"人物|人脸|上半身|下半身|观众|会场|背景", str(value)
                        )
                    ]
                    if relevant:
                        semantic_parts.append("无法辨认：" + "；".join(relevant))
            evidence_text = text
            if semantic_parts:
                evidence_text = (evidence_text + "\n[视觉理解候选]\n" + "\n".join(semantic_parts)).strip()
            if not evidence_text:
                evidence_text = f"图片 {asset['filename']} 尚未配置 OCR/VLM，需人工检查原图。"
                degraded = True
            else:
                recognized += 1
            confidence = float(ocr_result.get("confidence", 0.7 if text else (0.6 if vlm_result else 0.0)))
            evidence_values = {
                "locator_json": self.db.json({"regions": ocr_result.get("regions", []), "vlm": vlm_result}),
                "text": evidence_text,
                "confidence": confidence,
                "source_backend": (
                    "local-ocr+vlm" if text and vlm_result else
                    "local-ocr" if text else
                    "local-vlm" if vlm_result and vlm_ready else
                    "local-vlm" if vlm_result else
                    "degraded"
                ),
            }
            existing_evidence = self.db.all(
                "SELECT id FROM evidence WHERE meeting_id=? AND asset_id=? AND kind='image' ORDER BY created_at",
                (meeting_id, asset["id"]),
            )
            if existing_evidence:
                primary_id = existing_evidence[0]["id"]
                self.db.execute(
                    "UPDATE evidence SET locator_json=?, text=?, confidence=?, source_backend=? WHERE id=?",
                    (*evidence_values.values(), primary_id),
                )
                for duplicate in existing_evidence[1:]:
                    if duplicate["id"] not in locked_evidence_ids:
                        self.db.execute("DELETE FROM evidence WHERE id=?", (duplicate["id"],))
            else:
                self.db.insert(
                    "evidence",
                    {
                        "meeting_id": meeting_id,
                        "asset_id": asset["id"],
                        "segment_id": None,
                        "kind": "image",
                        **evidence_values,
                        "created_at": now_iso(),
                    },
                )
        detail = f"已处理 {len(assets)} 张图片，{recognized} 张获得可检索内容"
        if vlm_errors:
            detail += f"；{len(vlm_errors)} 张本地 VLM 失败，首个错误：{vlm_errors[0][:160]}"
        return StageResult(detail, degraded or bool(vlm_errors))

    def _fact_schema(
        self,
        *,
        max_items: int = 3,
        min_items: int = 0,
        kinds: list[str] | None = None,
    ) -> dict[str, Any]:
        allowed_kinds = kinds or ["topic", "decision", "action", "risk", "question", "metric", "term"]
        return {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "minItems": min_items,
                    "maxItems": max_items,
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "enum": allowed_kinds},
                            "statement": {"type": "string"},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["kind", "statement", "evidence_ids", "confidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["facts"],
            "additionalProperties": False,
        }

    def _windows(self, evidence: list[dict[str, Any]], max_chars: int = 2000) -> list[list[dict[str, Any]]]:
        overlap_target = int(max_chars * 0.15)
        chunk_limit = max_chars - overlap_target
        expanded: list[dict[str, Any]] = []
        for item in evidence:
            text = item["text"]
            if len(text) <= chunk_limit:
                expanded.append(item)
                continue
            chunk = ""
            for sentence in _safe_sentence_split(text):
                if chunk and len(chunk) + len(sentence) > chunk_limit:
                    expanded.append({**item, "text": chunk})
                    chunk = ""
                while len(sentence) > chunk_limit:
                    available = chunk_limit - len(chunk)
                    chunk += sentence[:available]
                    expanded.append({**item, "text": chunk})
                    chunk = ""
                    sentence = sentence[available:]
                chunk += sentence
            if chunk:
                expanded.append({**item, "text": chunk})

        windows: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        length = 0
        for item in expanded:
            item_len = len(item["text"])
            if current and length + item_len > max_chars:
                windows.append(current)
                overlap: list[dict[str, Any]] = []
                overlap_len = 0
                for previous in reversed(current):
                    remaining = overlap_target - overlap_len
                    if remaining <= 0:
                        break
                    previous_text = previous["text"]
                    piece = previous_text[-remaining:]
                    overlap.insert(0, {**previous, "text": piece})
                    overlap_len += len(piece)
                current = overlap
                length = overlap_len
                if length + item_len > max_chars:
                    current = []
                    length = 0
            current.append(item)
            length += item_len
        if current:
            windows.append(current)
        return windows

    def _heuristic_facts(self, window: list[dict[str, Any]]) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        rules = [
            ("action", ("负责", "需要", "待办", "完成", "跟进", "行动")),
            ("decision", ("决定", "确定", "结论", "同意", "采用")),
            ("risk", ("风险", "担心", "可能失败", "阻塞", "延期")),
            ("question", ("问题", "疑问", "是否", "？", "?")),
            ("metric", ("%", "目标", "预算", "万元", "小时", "分钟")),
        ]
        for item in window:
            for sentence in _safe_sentence_split(item["text"]):
                kind = next((name for name, keywords in rules if any(k in sentence for k in keywords)), None)
                if kind:
                    facts.append({
                        "kind": kind,
                        "statement": sentence,
                        "participants": [],
                        "owner": None,
                        "due_at": None,
                        "status": None,
                        "evidence_ids": [item["id"]],
                        "confidence": 0.45,
                    })
        if not facts and window:
            facts.append({
                "kind": "topic",
                "statement": window[0]["text"][:240],
                "participants": [], "owner": None, "due_at": None, "status": None,
                "evidence_ids": [window[0]["id"]], "confidence": 0.35,
            })
        diverse: list[dict[str, Any]] = []
        seen_kinds: set[str] = set()
        for fact in facts:
            if fact["kind"] not in seen_kinds:
                diverse.append(fact)
                seen_kinds.add(fact["kind"])
        return diverse[:7]

    def _fact_extract(self, meeting_id: str, _job_id: str) -> StageResult:
        locked_ids = {row["id"] for row in self.db.all("SELECT id FROM facts WHERE meeting_id=? AND user_locked=1", (meeting_id,))}
        self.db.execute("DELETE FROM facts_fts WHERE meeting_id=?", (meeting_id,))
        self.db.execute(
            "DELETE FROM facts WHERE meeting_id=? AND user_locked=0 AND review_state NOT IN ('accepted','rejected','stale') "
            "AND id NOT IN (SELECT fact_id FROM media_links WHERE meeting_id=? AND (confirmed=1 OR rejected=1) AND fact_id IS NOT NULL)",
            (meeting_id, meeting_id),
        )
        evidence = self.db.all(
            "SELECT * FROM evidence WHERE meeting_id=? AND text<>'' AND source_backend<>'degraded' ORDER BY created_at",
            (meeting_id,),
        )
        if not evidence:
            return StageResult("没有可靠证据可用于事实抽取；降级证据仅进入待核实项", True)
        meeting = self.db.one("SELECT title, glossary FROM meetings WHERE id=?", (meeting_id,)) or {}
        normalized_glossary = re.sub(r"\W+", "", str(meeting.get("glossary") or "")).casefold()
        canonical_terms = _canonical_terms(
            str(meeting.get("title") or ""),
            str(meeting.get("glossary") or ""),
            evidence,
        )
        trusted_short_terms = set(
            _canonical_terms(
                str(meeting.get("title") or ""),
                str(meeting.get("glossary") or ""),
                [],
            )
        )
        all_candidates: list[dict[str, Any]] = []
        used_llm = False
        fallback_errors: list[str] = []
        retried_windows = 0
        windows = self._windows(evidence)
        with self.inference.model_manager.session() as ready:
            for window in windows:
                evidence_text = "\n".join(f"[EVIDENCE:{item['id']}] {item['text']}" for item in window)
                evidence_kinds_in_window = {str(item.get("kind") or "") for item in window}
                audio_window = evidence_kinds_in_window == {"audio"}
                candidates: list[dict[str, Any]]
                candidate_backend = "heuristic"
                if ready:
                    system_prompt = (
                        "你是严谨的会议事实抽取器。只提取证据明确支持的事实，不补充常识，不生成报告正文。"
                        "topic 用于背景、既往成果和讨论议题；decision 只用于会议明确通过、同意或确定的结论；"
                        "action 只用于会议后需要执行的未来任务，不能把已经完成的研究成果标为 action；"
                        "risk 只用于当前阻塞、隐患或明确风险；question 只用于仍未解决的问题；"
                        "metric 用于包含明确数字的重要指标，term 用于需要解释的术语。"
                        "模糊图片的单一推断和只出现一次、疑似音译错误的术语只能作为待核实候选，confidence 填 0.5。"
                    )
                    if audio_window:
                        fact_prompt = (
                            "从以下连续音频证据提取 1 至 2 条最重要事实。第一条必须是 topic，"
                            "用 20 至 80 个字概括本窗口主旨；第二条仅在存在明确决定、行动、风险、"
                            "未决问题或重要数字时输出。不要把单个术语当作主要事实。"
                            "每条必须引用给出的完整证据ID；直接证据 confidence 填 0.8。\n" + evidence_text
                        )
                        initial_schema = self._fact_schema(max_items=2, min_items=1)
                    else:
                        fact_prompt = (
                            "从以下图片证据提取议题、指标和术语。只输出最重要的 1 至 3 条；"
                            "每条事实必须引用给出的完整证据ID；图片单独支持的内容 confidence 填 0.5。\n"
                            + evidence_text
                        )
                        initial_schema = self._fact_schema(max_items=3)
                    try:
                        self.inference.model_manager.assert_resources()
                        data = self.inference.local_backend().complete_json(
                            system_prompt,
                            fact_prompt,
                            initial_schema,
                            temperature=0.0,
                            max_tokens=640,
                        )
                        self.inference.model_manager.assert_resources()
                        candidates = [item.model_dump() for item in FactBatch.model_validate(data).facts]
                        candidates = candidates[: (2 if audio_window else 3)]
                        if audio_window and (not candidates or candidates[0].get("kind") == "term"):
                            raise ValueError("音频窗口未返回主题事实")
                        used_llm = True
                        candidate_backend = "local-llm"
                    except Exception as first_exc:
                        try:
                            self.inference.model_manager.assert_resources()
                            retry_schema = self._fact_schema(
                                max_items=1,
                                min_items=1 if audio_window else 0,
                                kinds=["topic"] if audio_window else None,
                            )
                            retry_instruction = (
                                "上次输出无法解析。只输出 1 条 topic，用 20 至 80 个字概括本音频窗口主旨，"
                                "并引用最直接的一条完整证据ID。\n"
                                if audio_window
                                else "上次输出无法解析。只提取 1 条最重要且证据最明确的事实。\n"
                            )
                            data = self.inference.local_backend().complete_json(
                                system_prompt,
                                retry_instruction + evidence_text,
                                retry_schema,
                                temperature=0.0,
                                max_tokens=320,
                            )
                            self.inference.model_manager.assert_resources()
                            candidates = [item.model_dump() for item in FactBatch.model_validate(data).facts]
                            used_llm = True
                            candidate_backend = "local-llm"
                            retried_windows += 1
                        except Exception as retry_exc:
                            fallback_errors.append(
                                f"{type(first_exc).__name__}: {first_exc}; retry {type(retry_exc).__name__}: {retry_exc}"
                            )
                            candidates = self._heuristic_facts(window)
                else:
                    candidates = self._heuristic_facts(window)
                allowed = {item["id"] for item in window}
                evidence_lookup = {item["id"]: item for item in window}
                for item in candidates:
                    item["kind"] = _normalize_fact_kind(str(item.get("kind", "topic")))
                    statement = _apply_known_terms(
                        str(item.get("statement", "")).strip(),
                        canonical_terms,
                        trusted_short_terms,
                    )
                    item["statement"] = statement
                    if len(statement) < 8 or re.search(r"(?:从|引用|根据)?证据(?:中|里的|中的|信息|[：:]?[0-9a-f]{8,})", statement, re.IGNORECASE):
                        continue
                    future_markers = ("负责", "后续", "下一步", "计划", "将会", "将要", "需要", "待", "跟进", "希望", "建议")
                    past_markers = ("提出了", "构建了", "发表", "获得", "发现了", "已完成", "已交付", "录用")
                    if item["kind"] == "action" and any(marker in statement for marker in past_markers) and not any(marker in statement for marker in future_markers):
                        item["kind"] = "metric" if re.search(r"\d", statement) else "topic"
                    if item["kind"] == "action":
                        if not any(marker in statement for marker in future_markers) or any(marker in statement for marker in ("进入下一个议题", "进入第四个议题", "本次议题", "技术介绍")):
                            item["kind"] = "topic"
                        elif re.search(r"举办|报名|workshop|活动|开放日", statement, re.IGNORECASE) and "负责" not in statement:
                            item["kind"] = "topic"
                    if item["kind"] == "decision":
                        if any(marker in statement for marker in ("是否", "能否", "看一下是否", "需要决策")):
                            item["kind"] = "question"
                        elif not re.search(r"(?:会议|议题|委员|代表|工作组)?.{0,16}(?:决定|通过|同意|批准|确定采用|结论)", statement):
                            item["kind"] = "topic"
                    if item["kind"] == "risk" and not any(marker in statement for marker in ("风险", "阻塞", "隐患", "困难", "担心", "不足", "缺少", "可能", "尚未")):
                        item["kind"] = "topic"
                    raw_ids = list(item.get("evidence_ids", []))
                    ids = [value for value in raw_ids if value in allowed]
                    references_valid = bool(ids)
                    item["evidence_ids"] = ids or [window[0]["id"]]
                    supporting_rows = [evidence_lookup[value] for value in item["evidence_ids"] if value in evidence_lookup]
                    supported_text = "\n".join(str(row.get("text") or "") for row in supporting_rows)
                    stated_numbers = set(re.findall(r"\d+(?:\.\d+)?", statement))
                    evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", supported_text))
                    support_score = _evidence_support_score(statement, supported_text)
                    if candidate_backend == "local-llm" and (not references_valid or support_score < 0.32 or not stated_numbers.issubset(evidence_numbers)):
                        item["confidence"] = min(float(item.get("confidence", 0.8)), 0.45)
                        item["status"] = "needs_review"
                    # A citation proves traceability, not that noisy source recognition is correct.
                    # Pure slide-photo claims and isolated ASR/OCR terms stay candidates until
                    # cross-modal corroboration or a user confirms them.
                    item["confidence"] = min(float(item.get("confidence", 0.8)), 0.8)
                    if item["kind"] == "metric" and not stated_numbers:
                        item["kind"] = "topic"
                    evidence_kinds = {str(row.get("kind") or "") for row in supporting_rows}
                    normalized_statement = re.sub(r"\W+", "", statement).casefold()
                    isolated_unverified_term = (
                        item["kind"] == "term"
                        and len(item["evidence_ids"]) < 2
                        and (not normalized_statement or normalized_statement not in normalized_glossary)
                    )
                    if evidence_kinds == {"image"} or isolated_unverified_term:
                        item["confidence"] = min(float(item["confidence"]), 0.5)
                        item["status"] = "needs_review"
                    participants = list(item.get("participants") or [])
                    speaker = re.match(r"^([^：:]{1,24})[：:]", statement)
                    if speaker and speaker.group(1) not in participants:
                        participants.append(speaker.group(1))
                    owner_match = re.search(r"(?:^|[，,。；;：:\s])([\u4e00-\u9fffA-Za-z·]{1,6}?)(?:是|将)?负责(?!人)", statement)
                    item["participants"] = participants
                    item["owner"] = item.get("owner") or (owner_match.group(1) if owner_match else None)
                    due_match = re.search(r"((?:本|下)?周[一二三四五六日天](?:前)?|\d{1,2}月\d{1,2}日(?:前)?|\d{4}[-年]\d{1,2}[-月]\d{1,2}日?(?:前)?)", statement)
                    item["due_at"] = item.get("due_at") or (due_match.group(1) if due_match else None)
                    item["_source_backend"] = candidate_backend
                    all_candidates.append(item)
        deduped: list[dict[str, Any]] = []
        seen: set[str] = {
            re.sub(r"\W+", "", row["statement"]).lower()[:160]
            for row in self.db.all("SELECT statement FROM facts WHERE meeting_id=?", (meeting_id,))
        }
        for item in all_candidates:
            key = re.sub(r"\W+", "", item["statement"]).lower()[:160]
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        for item in deduped:
            fact_id = self.db.insert(
                "facts",
                {
                    "meeting_id": meeting_id,
                    "kind": item["kind"],
                    "statement": item["statement"],
                    "participants_json": self.db.json(item.get("participants", [])),
                    "owner": item.get("owner"),
                    "due_at": item.get("due_at"),
                    "status": item.get("status"),
                    "evidence_ids_json": self.db.json(item["evidence_ids"]),
                    "confidence": float(item.get("confidence", 0.5)),
                    "contradiction_ids_json": "[]",
                    "source_backend": item.get("_source_backend", "heuristic"),
                    "user_locked": 0,
                    "created_at": now_iso(),
                    "updated_at": now_iso(),
                },
            )
            self.db.execute("INSERT INTO facts_fts(fact_id, meeting_id, statement) VALUES(?,?,?)", (fact_id, meeting_id, item["statement"]))
        detail = f"已从 {len(windows)} 个证据窗口提取 {len(deduped)} 条事实，保留 {len(locked_ids)} 条用户锁定事实"
        if fallback_errors:
            detail += f"；{len(fallback_errors)} 个窗口回退：{fallback_errors[0][:180]}"
        if retried_windows:
            detail += f"；{retried_windows} 个窗口经单事实约束重试成功"
        return StageResult(detail, not used_llm)

    def _link(self, meeting_id: str, _job_id: str) -> StageResult:
        # Rebuild only machine candidates.  A rejection is an explicit human
        # decision and must survive reprocessing just like a confirmation.
        self.db.execute(
            "DELETE FROM media_links WHERE meeting_id=? AND confirmed=0 AND rejected=0",
            (meeting_id,),
        )
        image_evidence = self.db.all("SELECT * FROM evidence WHERE meeting_id=? AND kind='image'", (meeting_id,))
        facts = self.db.all("SELECT * FROM facts WHERE meeting_id=?", (meeting_id,))
        evidence_kinds = {
            row["id"]: row["kind"]
            for row in self.db.all("SELECT id, kind FROM evidence WHERE meeting_id=?", (meeting_id,))
        }
        candidate_threshold = 0.22
        automatic_threshold = 0.86
        count = 0
        confirmed_count = 0
        unmatched = 0
        seen_assets: set[str] = set()
        for image in image_evidence:
            if not image.get("asset_id") or image["asset_id"] in seen_assets:
                continue
            seen_assets.add(image["asset_id"])
            existing_choice = self.db.one(
                "SELECT id FROM media_links WHERE meeting_id=? AND asset_id=? AND (confirmed=1 OR rejected=1) LIMIT 1",
                (meeting_id, image["asset_id"]),
            )
            if existing_choice:
                continue
            best: tuple[float, dict[str, Any], bool] | None = None
            for fact in facts:
                try:
                    cited_ids = set(json.loads(fact.get("evidence_ids_json") or "[]"))
                except (TypeError, json.JSONDecodeError):
                    cited_ids = set()
                direct_source = image["id"] in cited_ids
                has_independent_source = any(
                    evidence_kinds.get(evidence_id) not in {None, "image"}
                    for evidence_id in cited_ids
                )
                semantic_score = _media_fact_link_score(
                    str(image.get("text") or ""),
                    str(fact.get("statement") or ""),
                    direct_source=False,
                )
                trusted_direct_source = (
                    direct_source
                    and semantic_score >= 0.45
                    and (
                        bool(fact.get("user_locked"))
                        or (
                            has_independent_source
                            and float(fact.get("confidence") or 0) >= 0.75
                            and str(fact.get("status") or "") != "needs_review"
                        )
                    )
                )
                score = max(semantic_score, 0.72) if direct_source else semantic_score
                # Only a user lock or corroborated, high-confidence direct citation
                # can cross the automatic threshold.  Lexical similarity alone is
                # always shown as a candidate, never promoted to a formal fact.
                if trusted_direct_source:
                    score = max(score, 0.9)
                if best is None or score > best[0]:
                    best = (score, fact, trusted_direct_source)
            if best and best[0] >= candidate_threshold:
                confirmed = int(best[0] >= automatic_threshold and best[2])
                self.db.insert(
                    "media_links",
                    {
                        "meeting_id": meeting_id,
                        "asset_id": image["asset_id"],
                        "fact_id": best[1]["id"],
                        "score": round(best[0], 4),
                        "confirmed": confirmed,
                        "rejected": 0,
                        "created_at": now_iso(),
                        "updated_at": now_iso(),
                    },
                )
                count += 1
                confirmed_count += confirmed
            else:
                unmatched += 1
        return StageResult(
            f"已生成 {count} 条图片—事实候选，其中 {confirmed_count} 条由直接引用和独立证据自动确认；"
            f"{unmatched} 张保持未关联。相似度达到 {candidate_threshold:.2f} 才展示候选，达到 "
            f"{automatic_threshold:.2f} 且来源可信才自动确认",
            unmatched > 0,
        )

    def _render_section(self, title: str, facts: list[dict[str, Any]]) -> tuple[str, str, list[str]]:
        lines = [f"## {title}"]
        evidence_ids: list[str] = []
        if not facts:
            lines.append("暂无已确认内容。")
        for fact in facts:
            ids = json.loads(fact["evidence_ids_json"] or "[]")
            evidence_ids.extend(ids)
            suffix = " ".join(f"[证据:{item[:8]}]" for item in ids)
            detail = fact["statement"]
            if fact.get("owner"):
                detail += f"（负责人：{fact['owner']}）"
            if fact.get("due_at"):
                detail += f"（截止：{fact['due_at']}）"
            lines.append(f"- {detail} {suffix}".rstrip())
        markdown = "\n".join(lines)
        items = "".join(f"<li>{html.escape(line[2:])}</li>" for line in lines[1:] if line.startswith("- "))
        empty = "".join(f"<p>{html.escape(line)}</p>" for line in lines[1:] if not line.startswith("- "))
        return markdown, f"<section><h2>{html.escape(title)}</h2>{'<ul>'+items+'</ul>' if items else empty}</section>", list(dict.fromkeys(evidence_ids))

    def _render_media_section(self, meeting_id: str) -> tuple[str, str, list[str]]:
        rows = self.db.all(
            "SELECT e.id AS evidence_id, e.text, a.filename, ml.score, ml.confirmed, "
            "ml.rejected, f.statement AS fact_statement FROM evidence e "
            "JOIN assets a ON a.id=e.asset_id "
            "LEFT JOIN media_links ml ON ml.meeting_id=e.meeting_id AND ml.asset_id=e.asset_id "
            "LEFT JOIN facts f ON f.id=ml.fact_id "
            "WHERE e.meeting_id=? AND e.kind='image' ORDER BY a.created_at",
            (meeting_id,),
        )
        lines = ["## 关联图片"]
        html_items: list[str] = []
        evidence_ids: list[str] = []
        seen: set[str] = set()
        for row in rows:
            if row["evidence_id"] in seen or bool(row.get("rejected")):
                continue
            seen.add(row["evidence_id"])
            evidence_ids.append(row["evidence_id"])
            caption = next((line.strip() for line in str(row.get("text") or "").splitlines() if line.strip()), "无可靠图片说明")
            caption = caption[:180]
            if bool(row.get("confirmed")) and row.get("fact_statement"):
                relation = f"已确认关联：{row['fact_statement']}"
            elif row.get("fact_statement"):
                relation = f"待确认关联（{float(row.get('score') or 0):.2f}）：{row['fact_statement']}"
            else:
                # An image can be a cited manuscript source without also being
                # attached to a normalized FactRecord.  Keep those two
                # provenance relationships explicit so readers do not mistake
                # an unconfirmed structured-fact link for an unusable source.
                relation = "图片材料待核对；图片—结构化事实：尚未单独确认"
            detail = f"{row['filename']}｜{caption}｜{relation} [证据:{row['evidence_id'][:8]}]"
            lines.append(f"- {detail}")
            html_items.append(f"<li>{html.escape(detail)}</li>")
        if len(lines) == 1:
            lines.append("暂无会议图片。")
        body = f"<ul>{''.join(html_items)}</ul>" if html_items else "<p>暂无会议图片。</p>"
        return "\n".join(lines), f"<section><h2>关联图片</h2>{body}</section>", evidence_ids

    def _review_items(self, meeting_id: str, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items = [fact for fact in facts if not _formal_fact(fact)]
        degraded = self.db.all(
            "SELECT id, kind, asset_id FROM evidence WHERE meeting_id=? AND source_backend='degraded' ORDER BY created_at",
            (meeting_id,),
        )
        for kind, unit, label in (("audio", "个", "音频片段尚未完成可靠 ASR，需配置模型或人工补录"), ("image", "张", "图片尚未获得可靠 OCR/VLM 内容，需配置模型或人工校对")):
            by_asset: dict[str, str] = {}
            for item in degraded:
                if item["kind"] == kind:
                    by_asset.setdefault(item["asset_id"] or item["id"], item["id"])
            ids = list(by_asset.values())
            if ids:
                items.append(
                    {
                        "kind": "topic",
                        "statement": f"{len(ids)} {unit}{label}。",
                        "owner": None,
                        "due_at": None,
                        "evidence_ids_json": self.db.json(ids),
                        "confidence": 0,
                        "user_locked": 0,
                    }
                )
        return items

    @staticmethod
    def _editorial_source_text(value: str, limit: int = 420) -> str:
        text = str(value or "")
        if "[视觉理解候选]" in text:
            text = text.split("[视觉理解候选]", 1)[0]
        lines: list[str] = []
        for raw in text.splitlines():
            line = re.sub(r"\s+", " ", raw).strip(" -•；;")
            if not line or line.startswith("无法辨认") or line in lines:
                continue
            lines.append(line)
        compact = "；".join(lines)
        return compact[:limit]

    def _editorial_sources(self, meeting_id: str, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        eligible_statements: dict[str, list[str]] = {}
        for fact in facts:
            if _formal_fact(fact):
                for evidence_id in json.loads(fact.get("evidence_ids_json") or "[]"):
                    eligible_statements.setdefault(evidence_id, []).append(fact["statement"])
        images = self.db.all(
            "SELECT e.id, e.text, e.confidence, a.filename, a.created_at "
            "FROM assets a JOIN evidence e ON e.asset_id=a.id "
            "WHERE a.meeting_id=? AND a.kind='image' AND e.source_backend<>'degraded' "
            "ORDER BY a.created_at, e.created_at",
            (meeting_id,),
        )
        sources: list[dict[str, Any]] = []
        for index, row in enumerate(images, start=1):
            if row["id"] not in eligible_statements:
                continue
            # Only admitted facts enter the prose prompt. An image citation is
            # not permission to promote every OCR/VLM candidate on that image.
            text = self._editorial_source_text("；".join(eligible_statements[row["id"]]))
            if text:
                sources.append({
                    "id": row["id"],
                    "kind": "image",
                    "order": index,
                    "label": f"图 {index:02d}",
                    "text": text,
                    "confidence": float(row.get("confidence") or 0),
                })
        seen = {item["id"] for item in sources}
        for fact in facts:
            if not _formal_fact(fact):
                continue
            try:
                evidence_ids = json.loads(fact.get("evidence_ids_json") or "[]")
            except (TypeError, json.JSONDecodeError):
                evidence_ids = []
            for evidence_id in evidence_ids[:2]:
                if evidence_id in seen:
                    continue
                evidence = self.db.one("SELECT kind, text FROM evidence WHERE id=?", (evidence_id,))
                if not evidence or evidence.get("kind") != "audio":
                    continue
                statement = str(fact.get("statement") or "").strip()
                source_text = self._editorial_source_text(str(evidence.get("text") or ""), 320)
                if not statement or not source_text:
                    continue
                sources.append({
                    "id": evidence_id,
                    "kind": "audio",
                    "order": len(sources) + 1,
                    "label": "演讲原话",
                    "text": f"主题：{statement}；原话：{source_text}",
                    "confidence": float(fact.get("confidence") or 0),
                })
                seen.add(evidence_id)
        return sources

    @staticmethod
    def _fallback_editorial_outline(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        images = [item for item in sources if item["kind"] == "image"]
        if not images:
            images = sources
        size = 4 if len(images) >= 20 else 3
        chapters: list[dict[str, Any]] = []
        for start in range(0, len(images), size):
            group = images[start : start + size]
            if not group:
                continue
            first = re.sub(r"^标题[：:]", "", group[0]["text"].split("；", 1)[0]).strip()
            chapters.append({
                "title": first or f"主题材料 {len(chapters) + 1}",
                "evidence_ids": [item["id"] for item in group],
            })
        return chapters[:10]

    @staticmethod
    def _editorial_marker(evidence_ids: list[str]) -> str:
        return "".join(f" [证据:{item}]" for item in evidence_ids)

    @staticmethod
    def _editorial_html(markdown: str) -> str:
        """Render clean editor HTML while retaining evidence in Markdown."""
        marker = re.compile(r"\s*\[证据:[^\]]+\]")
        parts: list[str] = ["<article>"]
        in_list = False
        in_table = False

        def close_blocks() -> None:
            nonlocal in_list, in_table
            if in_list:
                parts.append("</ul>")
                in_list = False
            if in_table:
                parts.append("</div>")
                in_table = False

        for raw in markdown.splitlines():
            line = marker.sub("", raw).strip()
            if not line:
                close_blocks()
                continue
            if re.fullmatch(r"[-:| ]+", line):
                continue
            heading = re.match(r"^(#{1,4})\s+(.+)$", line)
            if heading:
                close_blocks()
                level = min(3, len(heading.group(1)))
                parts.append(f"<h{level}>{html.escape(heading.group(2))}</h{level}>")
                continue
            if line.startswith("> "):
                close_blocks()
                parts.append(f"<blockquote><p>{html.escape(line[2:])}</p></blockquote>")
                continue
            if line.startswith("- "):
                if in_table:
                    parts.append("</div>")
                    in_table = False
                if not in_list:
                    parts.append("<ul>")
                    in_list = True
                parts.append(f"<li><p>{html.escape(line[2:])}</p></li>")
                continue
            if line.startswith("|") and line.endswith("|"):
                if in_list:
                    parts.append("</ul>")
                    in_list = False
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if not in_table:
                    parts.append('<div class="report-table">')
                    in_table = True
                parts.append(f"<p>{html.escape(' ｜ '.join(cells))}</p>")
                continue
            close_blocks()
            parts.append(f"<p>{html.escape(line)}</p>")
        close_blocks()
        parts.append("</article>")
        return "".join(parts)

    def _fallback_editorial_chapter(
        self,
        title: str,
        sources: list[dict[str, Any]],
    ) -> dict[str, Any]:
        sections: list[dict[str, Any]] = []
        for source in sources[:8]:
            pieces = [item.strip() for item in source["text"].split("；") if item.strip()]
            section_title = re.sub(r"^标题[：:]", "", pieces[0] if pieces else source["label"]).strip()
            points = [
                {"text": item, "evidence_ids": [source["id"]]}
                for item in pieces[1:5]
            ]
            explanation = "；".join(pieces[1:3])
            if explanation:
                summary = f"{section_title or source['label']}的核心信息是：{explanation}。"
            else:
                summary = f"现场材料将{section_title or source['label']}列为需要进一步核对的技术主题。"
            sections.append({
                "title": section_title or source["label"],
                "summary": summary,
                "evidence_ids": [source["id"]],
                "points": points,
                "table": {"headers": [], "rows": []},
            })
        return {"title": title, "sections": sections}

    def _editorial_article(
        self,
        meeting_id: str,
        meeting: dict[str, Any],
        facts: list[dict[str, Any]],
        *,
        generate: bool,
    ) -> tuple[str, str, list[str]]:
        sources = self._editorial_sources(meeting_id, facts)
        if not sources:
            markdown = (
                f"# {meeting.get('title', '会议')} — 内容整理文稿\n\n"
                "> 当前没有足够的可靠文字或图片材料，无法形成技术内容整理文稿。"
            )
            return markdown, f"<article><p>{html.escape(markdown)}</p></article>", []
        lookup = {item["id"]: item for item in sources}
        outline = self._fallback_editorial_outline(sources)
        generated_chapters: list[dict[str, Any]] = []
        used_llm = False
        if generate:
            with self.inference.model_manager.session() as ready:
                if ready:
                    outline_lines = []
                    outline_sources = [item for item in sources if item["kind"] == "image"] or sources[:24]
                    for source in outline_sources:
                        summary = source["text"][:42]
                        outline_lines.append(
                            f"[EVIDENCE:{source['id']}] {source['label']}｜{summary}"
                        )
                    minimum_chapters = 8 if len(outline_sources) >= 20 else 4
                    outline_schema = {
                        "type": "object",
                        "properties": {
                            "chapters": {
                                "type": "array",
                                "minItems": minimum_chapters,
                                "maxItems": 10,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string", "maxLength": 48},
                                        "evidence_ids": {
                                            "type": "array", "minItems": 1, "maxItems": 8,
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": ["title", "evidence_ids"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["chapters"],
                        "additionalProperties": False,
                    }
                    try:
                        data = self.inference.local_backend().complete_json(
                            "你是技术会议内容编辑。任务是把演讲和幻灯片按因果和知识依赖重组成可独立阅读的技术文章目录。"
                            "目录应覆盖背景与挑战、核心方法或平台、关键技术、应用/案例、未来方向和术语中实际存在的部分。"
                            "章节要形成‘问题—方法—能力边界—应用—后续方向’的阅读路径。"
                            "不要按幻灯片顺序机械分章，不写会议过程，不添加证据之外的行业常识；"
                            "路线图、产品能力和数字必须留在其证据所属章节，不能脱离来源合并。",
                            f"根据以下材料拟定 {minimum_chapters} 至 10 个互不重复、由浅入深的技术章节。"
                            "同一主题可合并不连续的材料；每章只能引用给出的完整证据ID。\n"
                            + "\n".join(outline_lines),
                            outline_schema,
                            temperature=0.0,
                            max_tokens=1200,
                        )
                        candidate_outline = []
                        assigned_ids: set[str] = set()
                        for chapter in data.get("chapters", []):
                            ids = list(dict.fromkeys(
                                value for value in chapter.get("evidence_ids", [])
                                if value in lookup and value not in assigned_ids
                            ))
                            title = str(chapter.get("title") or "").strip()
                            if title and ids:
                                candidate_outline.append({"title": title, "evidence_ids": ids})
                                assigned_ids.update(ids)
                        if len(candidate_outline) >= minimum_chapters:
                            source_positions = {
                                source["id"]: index for index, source in enumerate(outline_sources)
                            }
                            # A polished manuscript must cover the complete slide
                            # sequence, not merely the easiest items selected by a
                            # small model. Assign omissions to the nearest thematic
                            # chapter while keeping every source in only one place.
                            for source in outline_sources:
                                evidence_id = source["id"]
                                if evidence_id in assigned_ids:
                                    continue
                                position = source_positions[evidence_id]
                                eligible = [
                                    (index, chapter) for index, chapter in enumerate(candidate_outline)
                                    if len(chapter["evidence_ids"]) < 8
                                ]
                                if not eligible:
                                    break
                                target_index, target = min(
                                    eligible,
                                    key=lambda pair: min(
                                        abs(position - source_positions.get(item, position))
                                        for item in pair[1]["evidence_ids"]
                                    ),
                                )
                                candidate_outline[target_index]["evidence_ids"].append(evidence_id)
                                assigned_ids.add(evidence_id)
                            outline = candidate_outline[:10]
                    except Exception as exc:
                        LOGGER.warning("技术文稿目录生成失败，使用确定性目录：%s", exc)

                    section_schema = {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "maxLength": 48},
                            "summary": {"type": "string", "maxLength": 260},
                            "points": {
                                "type": "array", "maxItems": 5,
                                "items": {"type": "string", "maxLength": 140},
                            },
                            "table_headers": {
                                "type": "array", "maxItems": 4,
                                "items": {"type": "string", "maxLength": 40},
                            },
                            "table_rows": {
                                "type": "array", "maxItems": 6,
                                "items": {"type": "string", "maxLength": 320},
                            },
                            "evidence_note": {"type": "string", "maxLength": 140},
                        },
                        "required": ["title", "summary", "points", "table_headers", "table_rows", "evidence_note"],
                        "additionalProperties": False,
                    }
                    for chapter in outline:
                        chapter_sources = [lookup[value] for value in chapter["evidence_ids"] if value in lookup][:8]
                        if not chapter_sources:
                            continue
                        valid_sections: list[dict[str, Any]] = []
                        for source in chapter_sources:
                            try:
                                self.inference.model_manager.assert_resources()
                                data = self.inference.local_backend().complete_json(
                                    "你是严谨的中文技术文章编辑。把单份幻灯片或演讲证据改写成可独立阅读的小节，"
                                    "而不是逐条抄录幻灯片。summary 第一两句先给出本节结论，随后解释因果、关系或工程意义，"
                                    "并自然承接所属章节；points 只保留支持结论的完整句。禁止空标签（如‘Copilot：’）、"
                                    "孤立箭头、截断句、中英文无解释拼接、会议过程和‘本节整理了’等模板话术。"
                                    "不得引入外部知识或补全模糊图片；材料矛盾、不完整或不可辨时，只说明待核实，不猜测原文。"
                                    "所有数字必须逐字存在于材料中。数字、路线图或产品能力出现时，evidence_note 必须用"
                                    "‘口径：…；核实状态：…’说明对象/条件；材料没有条件就明确写‘适用条件未说明，待核实’，"
                                    "不得写成已独立验证。没有这类主张时 evidence_note 为空。",
                                    f"所属章节：{chapter['title']}\n材料置信度：{float(source.get('confidence') or 0):.2f}"
                                    f"\n材料：{source['text'][:420]}\n"
                                    "summary 写 80–220 字连续自然中文；points 写 0–4 条完整句。"
                                    "表格行用竖线分隔且列数与 table_headers 一致；无真实可比较字段时两个数组均为空。",
                                    section_schema,
                                    temperature=0.1,
                                    max_tokens=520,
                                )
                                summary = str(data.get("summary") or "").strip()
                                title = str(data.get("title") or "").strip()
                                prose = [summary] + [str(value).strip() for value in data.get("points", [])]
                                if len(summary) < 30 or not title:
                                    raise ValueError("小节正文过短")
                                if any(
                                    not value
                                    or re.search(r"(?:^|\s)[^。！？!?]{1,32}[：:]$|(?:^|\s)[→←↔]+(?:\s|$)", value)
                                    or value[-1] not in "。！？!?）》”’"
                                    for value in prose
                                ):
                                    raise ValueError("小节含空标签、孤立箭头或截断句")
                                generated_numbers = set(re.findall(r"\d+(?:\.\d+)?", "\n".join(prose)))
                                source_numbers = set(re.findall(r"\d+(?:\.\d+)?", source["text"]))
                                if not generated_numbers.issubset(source_numbers):
                                    raise ValueError("小节含来源中不存在的数字")
                                sensitive_claim = bool(
                                    generated_numbers
                                    or re.search(r"路线图|roadmap|(?:计划|预计).{0,12}(?:发布|提供|交付)|"
                                                 r"(?:支持|能够|可实现|提升|加速|性能)", "\n".join(prose), re.IGNORECASE)
                                )
                                evidence_note = str(data.get("evidence_note") or "").strip()
                                if sensitive_claim and not (
                                    "口径" in evidence_note and ("核实" in evidence_note or "待核实" in evidence_note)
                                ):
                                    raise ValueError("数字、路线图或产品能力缺少口径与核实状态")
                                points = [
                                    {"text": str(value).strip(), "evidence_ids": [source["id"]]}
                                    for value in data.get("points", [])
                                    if str(value).strip()
                                ]
                                headers = [
                                    str(value).strip() for value in data.get("table_headers", [])
                                    if str(value).strip()
                                ]
                                rows = []
                                for raw_row in data.get("table_rows", []):
                                    cells = [value.strip() for value in str(raw_row).split("|")]
                                    if headers and len(cells) == len(headers):
                                        rows.append({"cells": cells, "evidence_ids": [source["id"]]})
                                valid_sections.append({
                                    "title": title,
                                    "summary": summary,
                                    "evidence_ids": [source["id"]],
                                    "points": points,
                                    "table": {"headers": headers if rows else [], "rows": rows},
                                    "evidence_note": evidence_note,
                                })
                                used_llm = True
                            except Exception as exc:
                                LOGGER.warning(
                                    "技术文稿小节生成失败，使用证据直排：meeting=%s source=%s error=%s",
                                    meeting_id,
                                    source["label"],
                                    exc,
                                )
                                valid_sections.extend(
                                    self._fallback_editorial_chapter(chapter["title"], [source])["sections"]
                                )
                        generated_chapters.append({"title": chapter["title"], "sections": valid_sections})
        if not generated_chapters:
            for chapter in outline:
                chapter_sources = [lookup[value] for value in chapter["evidence_ids"] if value in lookup]
                if chapter_sources:
                    generated_chapters.append(self._fallback_editorial_chapter(chapter["title"], chapter_sources))

        chinese_numbers = "一二三四五六七八九十"
        title = f"{meeting.get('title', '会议')} — 技术内容整理文稿"
        lines = [f"# {title}", "", f"> 本文稿基于现场演讲、转写与图片材料重组，共形成 {len(generated_chapters)} 个技术章节。正文按知识逻辑组织，证据定位集中保留。", ""]
        used_ids: list[str] = []
        for chapter_index, chapter in enumerate(generated_chapters, start=1):
            numeral = chinese_numbers[chapter_index - 1] if chapter_index <= len(chinese_numbers) else str(chapter_index)
            lines.extend([f"## {numeral}、{chapter['title']}", ""])
            for section_index, section in enumerate(chapter.get("sections", []), start=1):
                lines.extend([f"### {chapter_index}.{section_index} {section['title']}", ""])
                section_ids = list(dict.fromkeys(section.get("evidence_ids", [])))
                used_ids.extend(section_ids)
                lines.append(section["summary"] + self._editorial_marker(section_ids))
                lines.append("")
                for point in section.get("points", []):
                    point_ids = list(dict.fromkeys(point.get("evidence_ids", [])))
                    used_ids.extend(point_ids)
                    lines.append(f"- {point['text']}{self._editorial_marker(point_ids)}")
                if section.get("points"):
                    lines.append("")
                table = section.get("table") or {}
                headers = table.get("headers") or []
                rows = table.get("rows") or []
                if headers and rows:
                    lines.append("| " + " | ".join(headers) + " |")
                    lines.append("| " + " | ".join("---" for _ in headers) + " |")
                    for row in rows:
                        row_ids = list(dict.fromkeys(row.get("evidence_ids", [])))
                        used_ids.extend(row_ids)
                        cells = list(row.get("cells", []))
                        if cells:
                            cells[-1] += self._editorial_marker(row_ids)
                        lines.append("| " + " | ".join(cells) + " |")
                    lines.append("")
                evidence_note = str(section.get("evidence_note") or "").strip()
                if evidence_note:
                    lines.append(f"> {evidence_note}{self._editorial_marker(section_ids)}")
                    lines.append("")
        lines.extend([
            "## 附录：阅读与核对说明",
            "",
            "- 正文中的技术能力、路线图和数值均按现场材料陈述处理，不代表本工具进行了独立复测。",
            "- 模糊、冲突或无法从原始材料定位的信息不写入正式结论，应回到校对工作台或原图确认。",
        ])
        markdown = "\n".join(lines).strip()
        content_html = self._editorial_html(markdown)
        backend = "本地模型分章生成" if used_llm else "确定性降级编排"
        content_html = f"<!-- {backend} -->" + content_html
        return markdown, content_html, list(dict.fromkeys(used_ids))

    def _report(self, meeting_id: str, _job_id: str) -> StageResult:
        meeting = self.db.one("SELECT * FROM meetings WHERE id=?", (meeting_id,)) or {}
        facts = self.db.all("SELECT * FROM facts WHERE meeting_id=? ORDER BY created_at", (meeting_id,))
        confirmed_facts = [fact for fact in facts if _formal_fact(fact)]
        by_kind: dict[str, list[dict[str, Any]]] = {kind: [] for kind in ["topic", "decision", "action", "risk", "question", "metric", "term"]}
        for fact in confirmed_facts:
            by_kind.setdefault(fact["kind"], []).append(fact)
        if _job_id == "verify":
            notice = "## 技术内容整理文稿\n原正文引用的事实已发生变化或未通过证据校验，正文已失效。请校对事实后重新生成并审阅。"
            article = (notice, self._editorial_html(notice), [])
        else:
            article = self._editorial_article(
                meeting_id,
                meeting,
                facts,
                generate=meeting.get("report_detail") == "DETAILED",
            )
        sections: dict[str, tuple[str, str, list[str]]] = {
            "summary": self._render_section("会议摘要", (by_kind["decision"] + by_kind["topic"] + by_kind["action"])[:8]),
            "article": article,
            "topics": self._render_section("议题与背景", by_kind["topic"] + by_kind["term"] + by_kind["metric"]),
            "decisions": self._render_section("关键决定", by_kind["decision"]),
            "actions": self._render_section("行动项", by_kind["action"]),
            "risks": self._render_section("风险与异议", by_kind["risk"]),
            "questions": self._render_section("未决问题", by_kind["question"]),
            "media": self._render_media_section(meeting_id),
            "review": self._render_section("待核实项", self._review_items(meeting_id, facts)),
        }
        rows = []
        for section, (markdown, section_html, evidence_ids) in sections.items():
            rows.append({
                    "section": section,
                    "content_markdown": markdown,
                    "content_html": section_html,
                    "evidence_ids_json": self.db.json(evidence_ids),
                    "editorial_state": "invalidated" if section == "article" and _job_id == "verify" else "automatic_model_draft" if section == "article" else "verified_facts",
                    "needs_review": int(section == "article"),
            })
        version = self._publish_report_rows(meeting_id, rows)
        return StageResult(f"已生成《{meeting.get('title', '会议报告')}》第 {version} 版，共 {len(sections)} 个可独立重生成章节")

    @_atomic
    def _publish_report_rows(self, meeting_id: str, rows: list[dict[str, Any]], expected_version: int | None = None) -> int:
        if {row["section"] for row in rows} != set(REPORT_SECTIONS) or len(rows) != len(REPORT_SECTIONS):
            raise ValueError("报告版本不完整，拒绝发布")
        latest = self.db.one("SELECT COALESCE(MAX(version),0) AS version FROM reports WHERE meeting_id=?", (meeting_id,))
        current = int(latest["version"] if latest else 0)
        if expected_version is not None and current != expected_version:
            raise ValueError("报告版本已变化，请刷新后重试")
        for row in rows:
            self.db.insert("reports", {**row, "meeting_id": meeting_id, "version": current + 1, "created_at": now_iso(), "updated_at": now_iso()})
        return current + 1

    def regenerate_section(self, meeting_id: str, section: str) -> dict[str, Any]:
        if not self._run_lock.acquire(blocking=False):
            raise ValueError("已有会议处理或章节生成任务运行，请稍后重试")
        try:
            self._verify(meeting_id, "regenerate")
            return self._regenerate_section(meeting_id, section)
        finally:
            self.inference.model_manager.stop()
            self.inference.vlm_manager.stop()
            self._run_lock.release()

    def _regenerate_section(self, meeting_id: str, section: str) -> dict[str, Any]:
        if section not in REPORT_SECTIONS:
            raise ValueError(f"未知报告章节：{section}")
        latest = self.db.one("SELECT COALESCE(MAX(version),0) AS version FROM reports WHERE meeting_id=?", (meeting_id,))
        current_version = int(latest["version"] if latest else 0)
        if current_version == 0:
            self._report(meeting_id, "manual")
            current_version = 1
        rows = self.db.all("SELECT * FROM reports WHERE meeting_id=? AND version=?", (meeting_id, current_version))
        new_version = current_version + 1
        facts = self.db.all("SELECT * FROM facts WHERE meeting_id=? ORDER BY created_at", (meeting_id,))
        confirmed_facts = [fact for fact in facts if _formal_fact(fact)]
        kind_map = {
            "summary": ("会议摘要", ["decision", "topic", "action"]),
            "article": ("技术内容整理文稿", []),
            "topics": ("议题与背景", ["topic", "term", "metric"]),
            "decisions": ("关键决定", ["decision"]),
            "actions": ("行动项", ["action"]),
            "risks": ("风险与异议", ["risk"]),
            "questions": ("未决问题", ["question"]),
            "media": ("关联图片", []),
            "review": ("待核实项", []),
        }
        title, kinds = kind_map[section]
        selected = [fact for fact in confirmed_facts if fact["kind"] in kinds]
        if section == "review":
            selected = self._review_items(meeting_id, facts)
        if section == "summary":
            selected = selected[:8]
        if section == "article":
            meeting = self.db.one("SELECT * FROM meetings WHERE id=?", (meeting_id,)) or {}
            markdown, section_html, evidence_ids = self._editorial_article(
                meeting_id, meeting, facts, generate=True
            )
        elif section == "media":
            markdown, section_html, evidence_ids = self._render_media_section(meeting_id)
        else:
            markdown, section_html, evidence_ids = self._render_section(title, selected)
        published_rows = []
        for row in rows:
            values = {
                "section": row["section"],
                "content_markdown": markdown if row["section"] == section else row["content_markdown"],
                "content_html": section_html if row["section"] == section else row["content_html"],
                "evidence_ids_json": self.db.json(evidence_ids) if row["section"] == section else row["evidence_ids_json"],
                "editorial_state": ("automatic_model_draft" if section == "article" else "verified_facts") if row["section"] == section else row.get("editorial_state", "automatic_model_draft"),
                "needs_review": int(section == "article") if row["section"] == section else row.get("needs_review", 1),
            }
            published_rows.append(values)
        if section == "article" and not any(row["section"] == "article" for row in rows):
            published_rows.append({
                    "section": "article",
                    "content_markdown": markdown,
                    "content_html": section_html,
                    "evidence_ids_json": self.db.json(evidence_ids),
                    "editorial_state": "automatic_model_draft",
                    "needs_review": 1,
                })
        new_version = self._publish_report_rows(meeting_id, published_rows, current_version)
        return self.db.one("SELECT * FROM reports WHERE meeting_id=? AND version=? AND section=?", (meeting_id, new_version, section)) or {}

    def _verify(self, meeting_id: str, _job_id: str) -> StageResult:
        facts = self.db.all("SELECT * FROM facts WHERE meeting_id=?", (meeting_id,))
        evidence_rows = self.db.all("SELECT id, text FROM evidence WHERE meeting_id=?", (meeting_id,))
        evidence_text = {row["id"]: str(row.get("text") or "") for row in evidence_rows}
        invalid_ids: set[str] = set()
        reasons: dict[str, list[str]] = {}
        for fact in facts:
            ids = json.loads(fact["evidence_ids_json"] or "[]")
            valid = [item for item in ids if item in evidence_text]
            if not valid:
                reasons.setdefault(fact["id"], []).append("缺少有效证据")
                invalid_ids.add(fact["id"])
                continue
            supporting = "\n".join(evidence_text[item] for item in valid)
            stated_numbers = set(re.findall(r"\d+(?:\.\d+)?", fact["statement"]))
            evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", supporting))
            if not stated_numbers.issubset(evidence_numbers):
                reasons.setdefault(fact["id"], []).append("数字或日期与证据不一致")
                invalid_ids.add(fact["id"])
            owner = str(fact.get("owner") or "").strip()
            if owner and owner not in fact["statement"] and owner not in supporting:
                reasons.setdefault(fact["id"], []).append("负责人缺少证据")
                invalid_ids.add(fact["id"])
            due_at = str(fact.get("due_at") or "").strip()
            if due_at and due_at not in fact["statement"] and due_at not in supporting:
                reasons.setdefault(fact["id"], []).append("截止时间缺少证据")
                invalid_ids.add(fact["id"])

        # Conservative conflict detection: only flag closely matching formal facts
        # of the same kind when their numbers or positive/negative polarity differ.
        formal = [fact for fact in facts if _formal_fact(fact)]
        contradictions: dict[str, set[str]] = {fact["id"]: set() for fact in facts}
        negation = ("不", "未", "否决", "取消", "禁止", "停止")
        for index, left in enumerate(formal):
            left_tokens = _lexical_tokens(left["statement"])
            left_numbers = set(re.findall(r"\d+(?:\.\d+)?", left["statement"]))
            for right in formal[index + 1 :]:
                if left["kind"] != right["kind"]:
                    continue
                right_tokens = _lexical_tokens(right["statement"])
                similarity = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
                if similarity < 0.65:
                    continue
                right_numbers = set(re.findall(r"\d+(?:\.\d+)?", right["statement"]))
                polarity_differs = any(value in left["statement"] for value in negation) != any(
                    value in right["statement"] for value in negation
                )
                numbers_differ = bool(left_numbers and right_numbers and left_numbers != right_numbers)
                if polarity_differs or numbers_differ:
                    contradictions[left["id"]].add(right["id"])
                    contradictions[right["id"]].add(left["id"])
                    invalid_ids.update((left["id"], right["id"]))
                    reasons.setdefault(left["id"], []).append("存在冲突事实")
                    reasons.setdefault(right["id"], []).append("存在冲突事实")

        for fact in facts:
            conflict_ids = sorted(contradictions[fact["id"]])
            if fact["id"] in invalid_ids:
                self.db.execute(
                    "UPDATE facts SET confidence=0, status='needs_review', contradiction_ids_json=?, updated_at=? WHERE id=?",
                    (self.db.json(conflict_ids), now_iso(), fact["id"]),
                )
            elif conflict_ids != json.loads(fact["contradiction_ids_json"] or "[]"):
                self.db.execute(
                    "UPDATE facts SET contradiction_ids_json=?, updated_at=? WHERE id=?",
                    (self.db.json(conflict_ids), now_iso(), fact["id"]),
                )

        # REPORT precedes VERIFY in the resumable state machine. If verification
        # moved anything out of the formal set, publish a new clean version so the
        # latest report can never retain an unsupported conclusion.
        newly_invalid = {fact["id"] for fact in facts if fact["id"] in invalid_ids and fact.get("status") != "needs_review"}
        if newly_invalid:
            self._report(meeting_id, "verify")
        supported = len(facts) - len(invalid_ids)
        eligible_formal = [fact for fact in formal if fact["id"] not in invalid_ids]
        covered_formal = sum(
            bool([item for item in json.loads(fact["evidence_ids_json"] or "[]") if item in evidence_text])
            for fact in eligible_formal
        )
        coverage = 100.0 if not eligible_formal else round(100 * covered_formal / len(eligible_formal), 1)
        detail = (
            f"证据审计完成：{supported}/{len(facts)} 条候选事实通过；最新报告正式结论证据覆盖率 {coverage:.1f}%；"
            f"{len(invalid_ids)} 条移入待核实"
        )
        if reasons:
            first = next(iter(reasons.values()))
            detail += f"（首项：{'、'.join(first)}）"
        return StageResult(detail, bool(invalid_ids))

    def _audit(self, meeting_id: str, operation: str, summary: str, status: str) -> None:
        provider = self.settings.cloud_base_url or "unconfigured"
        self.db.insert(
            "cloud_audit",
            {
                "meeting_id": meeting_id,
                "operation": operation,
                "payload_summary": summary[:500],
                "provider": provider,
                "status": status,
                "created_at": now_iso(),
            },
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
