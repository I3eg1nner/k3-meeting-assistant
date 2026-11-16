from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .config import Settings
from .db import Database, now_iso
from .report_quality import (
    ExportQualityError,
    QualityResult,
    audit_detailed_manuscript,
    audit_docx,
    audit_pdf,
    convert_docx_to_pdf,
)


SECTION_ORDER = ["summary", "article", "topics", "decisions", "actions", "risks", "questions", "media", "review"]
REPORT_DETAILS = {
    "BRIEF": ("精简纪要", "突出结论、行动项与少量关键材料"),
    "STANDARD": ("标准报告", "兼顾议题脉络、证据边界与图片索引"),
    "DETAILED": ("技术整理文稿", "按知识逻辑重组章节、解释、表格、图片与证据附录"),
}
EVIDENCE_TAG = re.compile(r"\[证据:([0-9a-fA-F]{8,32})\]")
FONT = "Noto Sans SC"
CONTENT_DXA = 9360
NAVY = "203748"
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
GOLD = "B4863B"
INK = "202F3A"
MUTED = "65717B"
LIGHT = "F2F4F7"
PALE_BLUE = "E8EEF5"
PALE_GOLD = "FFF8E8"
PALE_RED = "FCECEC"


def _safe_name(value: str) -> str:
    value = re.sub(r"[^\w\u4e00-\u9fff.-]+", "-", value).strip("-.")
    return value[:80] or "meeting-report"


def _json_list(value: str | None) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
        return [str(item) for item in parsed] if isinstance(parsed, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def _normalize(value: str) -> str:
    value = EVIDENCE_TAG.sub("", value)
    value = re.sub(r"（(?:负责人|截止)[^）]*）", "", value)
    value = re.sub(r"^(?:议题|决定|行动项|风险|问题|指标|术语)\s*[：:]\s*", "", value.strip())
    return re.sub(r"[\W_]+", "", value).lower()


def _stamp(milliseconds: int | float | None) -> str:
    seconds = max(0, int(milliseconds or 0) // 1000)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def _set_run(run, *, size=None, color: str | None = None, bold: bool | None = None, italic: bool | None = None) -> None:
    from docx.oxml.ns import qn

    run.font.name = FONT
    rfonts = run._element.get_or_add_rPr().get_or_add_rFonts()
    for key in ("ascii", "hAnsi", "eastAsia"):
        rfonts.set(qn(f"w:{key}"), FONT)
    if size is not None:
        run.font.size = size
    if color:
        from docx.shared import RGBColor

        run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def _shade_cell(cell, fill: str) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    tc_pr = cell._tc.get_or_add_tcPr()
    node = tc_pr.find(qn("w:shd"))
    if node is None:
        node = OxmlElement("w:shd")
        tc_pr.append(node)
    node.set(qn("w:fill"), fill)


def _set_cell_margins(cell, top: int = 80, bottom: int = 80, start: int = 120, end: int = 120) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for tag, value in (("top", top), ("bottom", bottom), ("start", start), ("end", end)):
        node = tc_mar.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def _set_table_geometry(table, widths: list[int], *, repeat_first_row: bool = True) -> None:
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    if sum(widths) != CONTENT_DXA:
        raise ValueError("表格列宽必须合计 9360 DXA")
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    properties = table._tbl.tblPr
    width = properties.find(qn("w:tblW"))
    if width is None:
        width = OxmlElement("w:tblW")
        properties.append(width)
    width.set(qn("w:w"), str(CONTENT_DXA))
    width.set(qn("w:type"), "dxa")
    indent = properties.find(qn("w:tblInd"))
    if indent is None:
        indent = OxmlElement("w:tblInd")
        properties.append(indent)
    indent.set(qn("w:w"), "120")
    indent.set(qn("w:type"), "dxa")
    layout = properties.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        properties.append(layout)
    layout.set(qn("w:type"), "fixed")
    for index, width_dxa in enumerate(widths):
        table._tbl.tblGrid[index].set(qn("w:w"), str(width_dxa))
    for row in table.rows:
        row_properties = row._tr.get_or_add_trPr()
        cant_split = row_properties.find(qn("w:cantSplit"))
        if cant_split is None:
            cant_split = OxmlElement("w:cantSplit")
            row_properties.append(cant_split)
        cant_split.set(qn("w:val"), "1")
        for index, cell in enumerate(row.cells):
            tc_width = cell._tc.get_or_add_tcPr().get_or_add_tcW()
            tc_width.set(qn("w:w"), str(widths[index]))
            tc_width.set(qn("w:type"), "dxa")
            _set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    first_properties = table.rows[0]._tr.get_or_add_trPr()
    header = first_properties.find(qn("w:tblHeader"))
    if header is None:
        header = OxmlElement("w:tblHeader")
        first_properties.append(header)
    header.set(qn("w:val"), "1" if repeat_first_row else "0")


def _add_page_number(paragraph) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    run = paragraph.add_run("第 ")
    _set_run(run, size=Pt(8.5), color=MUTED)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    paragraph._p.append(field)
    run = paragraph.add_run(" 页")
    _set_run(run, size=Pt(8.5), color=MUTED)


class ReportExporter:
    """Build selectable evidence reports and refuse delivery when QA fails."""

    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db

    def _latest(self, meeting_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        meeting = self.db.one("SELECT * FROM meetings WHERE id=?", (meeting_id,))
        if not meeting:
            raise FileNotFoundError("会议不存在")
        row = self.db.one("SELECT MAX(version) AS version FROM reports WHERE meeting_id=?", (meeting_id,))
        if not row or row["version"] is None:
            raise FileNotFoundError("报告尚未生成")
        reports = self.db.all(
            "SELECT * FROM reports WHERE meeting_id=? AND version=?", (meeting_id, row["version"])
        )
        reports.sort(key=lambda item: SECTION_ORDER.index(item["section"]) if item["section"] in SECTION_ORDER else 99)
        return meeting, reports

    def detail_for(self, meeting_id: str, detail: str | None = None) -> str:
        meeting = self.db.one("SELECT report_detail FROM meetings WHERE id=?", (meeting_id,))
        if not meeting:
            raise FileNotFoundError("会议不存在")
        value = str(detail or meeting.get("report_detail") or "STANDARD").upper()
        if value not in REPORT_DETAILS:
            raise ValueError(f"未知报告深度：{value}")
        return value

    def _evidence(self, meeting_id: str) -> dict[str, dict[str, Any]]:
        rows = self.db.all(
            "SELECT e.*, a.filename, "
            "COALESCE(ts.user_text, ts.corrected_text, ts.raw_text) AS segment_text, "
            "ts.start_ms AS segment_start_ms, ts.end_ms AS segment_end_ms, "
            "COALESCE(s.display_name, s.label) AS speaker_name "
            "FROM evidence e LEFT JOIN assets a ON a.id=e.asset_id "
            "LEFT JOIN transcript_segments ts ON ts.id=e.segment_id "
            "LEFT JOIN speakers s ON s.id=ts.speaker_id WHERE e.meeting_id=?",
            (meeting_id,),
        )
        lookup: dict[str, dict[str, Any]] = {}
        for row in rows:
            lookup[row["id"]] = row
            lookup.setdefault(row["id"][:8], row)
        return lookup

    def _facts(self, meeting_id: str) -> list[dict[str, Any]]:
        return self.db.all("SELECT * FROM facts WHERE meeting_id=? ORDER BY created_at", (meeting_id,))

    @staticmethod
    def _report_entries(report: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not report:
            return []
        entries: list[dict[str, Any]] = []
        for raw in str(report.get("content_markdown") or "").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line in {"暂无已确认内容。", "暂无会议图片。"}:
                continue
            if line.startswith("- "):
                line = line[2:].strip()
            references = EVIDENCE_TAG.findall(line)
            statement = EVIDENCE_TAG.sub("", line).strip()
            if statement:
                entries.append({"statement": statement, "evidence_ids": references})
        return entries

    @staticmethod
    def _match_fact(statement: str, facts: list[dict[str, Any]]) -> dict[str, Any] | None:
        target = _normalize(statement)
        if not target:
            return None
        best: tuple[float, dict[str, Any]] | None = None
        for fact in facts:
            candidate = _normalize(str(fact.get("statement") or ""))
            if not candidate:
                continue
            if target == candidate:
                return fact
            if target in candidate or candidate in target:
                score = min(len(target), len(candidate)) / max(len(target), len(candidate))
                if best is None or score > best[0]:
                    best = (score, fact)
        return best[1] if best and best[0] >= 0.62 else None

    def _section_model(
        self,
        report: dict[str, Any] | None,
        facts: list[dict[str, Any]],
        evidence: dict[str, dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        formal: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        seen: set[str] = set()
        for entry in self._report_entries(report):
            key = _normalize(entry["statement"])
            if not key or key in seen:
                continue
            seen.add(key)
            fact = self._match_fact(entry["statement"], facts)
            references = entry["evidence_ids"] or (_json_list(fact.get("evidence_ids_json")) if fact else [])
            rows = [evidence.get(item) or evidence.get(item[:8]) for item in references]
            rows = [row for row in rows if row]
            model = {
                **entry,
                "kind": fact.get("kind") if fact else None,
                "owner": fact.get("owner") if fact else None,
                "due_at": fact.get("due_at") if fact else None,
                "status": fact.get("status") if fact else None,
                "confidence": float(fact.get("confidence") or 0) if fact else 0,
                "user_locked": bool(fact.get("user_locked")) if fact else False,
                "evidence": rows,
            }
            if rows and (not fact or float(fact.get("confidence") or 0) >= 0.55):
                formal.append(model)
            else:
                pending.append(model)
        return formal, pending

    def _images(self, meeting_id: str) -> list[dict[str, Any]]:
        rows = self.db.all(
            "SELECT a.*, e.id AS evidence_id, e.text AS evidence_text, e.confidence AS evidence_confidence, "
            "ml.score, ml.confirmed, ml.rejected, f.statement AS fact_statement "
            "FROM assets a LEFT JOIN evidence e ON e.asset_id=a.id AND e.kind='image' "
            "LEFT JOIN media_links ml ON ml.meeting_id=a.meeting_id AND ml.asset_id=a.id "
            "LEFT JOIN facts f ON f.id=ml.fact_id WHERE a.meeting_id=? AND a.kind='image' "
            "ORDER BY a.created_at, COALESCE(ml.confirmed,0) DESC, COALESCE(ml.score,0) DESC",
            (meeting_id,),
        )
        by_asset: dict[str, dict[str, Any]] = {}
        for row in rows:
            current = by_asset.get(row["id"])
            if not current:
                by_asset[row["id"]] = row
                continue
            current_rank = (bool(current.get("confirmed")), float(current.get("score") or 0))
            row_rank = (bool(row.get("confirmed")), float(row.get("score") or 0))
            if row_rank > current_rank:
                by_asset[row["id"]] = row
        return [item for item in by_asset.values() if not bool(item.get("rejected")) and Path(item["path"]).exists()]

    @staticmethod
    def _lexical_tokens(value: str) -> set[str]:
        lowered = value.lower()
        words = set(re.findall(r"[a-z][a-z0-9.+-]{1,}|\d+(?:\.\d+)?", lowered))
        chinese_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
        chinese = {
            run[index:index + 2]
            for run in chinese_runs
            for index in range(max(1, len(run) - 1))
            if len(run[index:index + 2]) == 2
        }
        return words | chinese

    @staticmethod
    def _select_images(
        images: list[dict[str, Any]], formal_statements: list[str], limit: int = 8
    ) -> list[dict[str, Any]]:
        formal_tokens = [ReportExporter._lexical_tokens(item) for item in formal_statements]

        def relevance(item: dict[str, Any]) -> float:
            tokens = ReportExporter._lexical_tokens(
                f"{item.get('evidence_text') or ''} {item.get('fact_statement') or ''}"
            )
            similarities = [len(tokens & target) / max(1, len(tokens | target)) for target in formal_tokens]
            return max(similarities, default=0)

        ranked = sorted(
            images,
            key=lambda item: (
                bool(item.get("confirmed")),
                relevance(item),
                float(item.get("score") or 0),
                float(item.get("evidence_confidence") or 0),
                len(ReportExporter._image_caption(item)),
            ),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        captions: set[str] = set()
        for item in ranked:
            caption = _normalize(ReportExporter._image_caption(item))[:48]
            if caption and caption in captions:
                continue
            captions.add(caption)
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _image_caption(image: dict[str, Any]) -> str:
        text = str(image.get("evidence_text") or "")
        title_match = re.search(r"(?:^|\n)标题[：:]\s*([^\n]+)", text)
        title = title_match.group(1).strip() if title_match else ""
        generic = {"会议PPT照片", "会议PPT核对", "会议幻灯片", "Hardware", "HAPS"}
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        caption = title if title and title not in generic else (lines[0] if lines else "图片文字未能可靠识别")
        caption = re.sub(r"\s+", " ", caption)
        return caption[:120]

    @staticmethod
    def _language_label(segments: list[dict[str, Any]]) -> str:
        sample = " ".join(
            str(item.get("user_text") or item.get("corrected_text") or item.get("raw_text") or "")
            for item in segments[:300]
        )
        chinese = len(re.findall(r"[\u4e00-\u9fff]", sample))
        english = len(re.findall(r"\b[A-Za-z]{2,}\b", sample))
        if chinese and english >= 20:
            return "中文为主，夹杂英文术语"
        if chinese:
            return "中文"
        if english:
            return "英文"
        return "未能判断"

    @staticmethod
    def _clean_excerpt(value: str, limit: int = 220) -> str:
        text = str(value or "").split("[视觉理解候选]", 1)[0]
        text = re.sub(r"\[EVIDENCE:[^\]]+\]", "", text)
        text = re.sub(r"\s+", " ", text).strip(" ，,；;")
        text = re.sub(
            r"(?:(?<=^)|(?<=[，。！？；]))\s*(?:嗯|呃|啊|那个|然后呢|那么|对吧)\s*[，,、]?",
            "",
            text,
        )
        text = re.sub(r"(?:\s*[，,]\s*){2,}", "，", text).strip(" ，,；;")
        if len(text) <= limit:
            return text
        candidate = text[:limit]
        boundary = max(candidate.rfind("。"), candidate.rfind("；"), candidate.rfind("！"), candidate.rfind("？"))
        return (candidate[: boundary + 1] if boundary >= int(limit * 0.55) else candidate.rstrip("，,；; ") + "……")

    @staticmethod
    def _segment_text(segment: dict[str, Any]) -> str:
        return ReportExporter._clean_excerpt(
            str(segment.get("user_text") or segment.get("corrected_text") or segment.get("raw_text") or ""),
            155,
        )

    @staticmethod
    def _timeline_groups(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reliable = [
            item for item in segments
            if item.get("source_backend") != "degraded" and ReportExporter._segment_text(item)
        ]
        duration = max((int(item.get("end_ms") or 0) for item in reliable), default=0)
        bucket_ms = 15 * 60_000 if duration > 70 * 60_000 else 10 * 60_000
        groups: list[dict[str, Any]] = []
        for start in range(0, max(duration, 1), bucket_ms):
            rows = [item for item in reliable if start <= int(item.get("start_ms") or 0) < start + bucket_ms]
            if not rows:
                continue

            def score(item: dict[str, Any]) -> tuple[int, int]:
                text = ReportExporter._segment_text(item)
                filler = len(re.findall(r"(?:嗯|呃|啊|那个|然后)", text))
                return (min(len(text), 220) - filler * 12, -int(item.get("start_ms") or 0))

            highlights = sorted(sorted(rows, key=score, reverse=True)[:2], key=lambda item: int(item.get("start_ms") or 0))
            groups.append({"start_ms": start, "end_ms": min(start + bucket_ms, duration), "segments": highlights})
        return groups

    @staticmethod
    def _image_insights(image: dict[str, Any], limit: int = 4) -> list[str]:
        text = str(image.get("evidence_text") or "")
        if "[视觉理解候选]" not in text:
            return []
        semantic = text.split("[视觉理解候选]", 1)[1]
        insights: list[str] = []
        for raw in semantic.splitlines():
            line = re.sub(r"\s+", " ", raw).strip(" -•；;")
            if not line or line.startswith("标题") or line.startswith("无法辨认"):
                continue
            if line not in insights:
                insights.append(line[:150])
            if len(insights) >= limit:
                break
        return insights

    @staticmethod
    def _topic_images(topic: dict[str, Any], images: list[dict[str, Any]], limit: int = 2) -> list[dict[str, Any]]:
        target = ReportExporter._lexical_tokens(str(topic.get("statement") or ""))
        ranked: list[tuple[float, dict[str, Any]]] = []
        for image in images:
            tokens = ReportExporter._lexical_tokens(
                f"{image.get('evidence_text') or ''} {image.get('fact_statement') or ''}"
            )
            score = len(target & tokens) / max(1, len(target | tokens))
            if score > 0:
                ranked.append((score, image))
        ranked.sort(key=lambda value: value[0], reverse=True)
        return [item for _, item in ranked[:limit]]

    @staticmethod
    def _citation(entry: dict[str, Any], image_numbers: dict[str, int]) -> str:
        labels: list[str] = []
        for row in entry.get("evidence", [])[:3]:
            try:
                locator = json.loads(row.get("locator_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                locator = {}
            if row.get("kind") == "audio":
                labels.append(f"音频 {_stamp(locator.get('start_ms'))}-{_stamp(locator.get('end_ms'))}")
            elif row.get("kind") == "image":
                number = image_numbers.get(str(row.get("asset_id") or ""))
                labels.append(f"图片 {number:02d}" if number else f"图片 {row.get('filename') or '原图'}")
            else:
                labels.append("文字稿")
        labels = list(dict.fromkeys(labels))
        extra = len(entry.get("evidence", [])) - len(labels)
        suffix = f"等 {len(entry.get('evidence', []))} 条证据" if extra > 0 else "；".join(labels)
        return f"〔{suffix}〕" if suffix else ""

    @staticmethod
    def _document_title(meeting_title: str, detail: str | None = None) -> str:
        if str(detail or "").upper() == "DETAILED":
            base = re.sub(r"(?:会议)?报告$", "", meeting_title).strip(" -—｜|")
            return f"{base} — 演讲内容整理文稿"
        if meeting_title.endswith("报告"):
            return meeting_title
        return f"{meeting_title}报告" if "会议" in meeting_title else f"{meeting_title}会议报告"

    @classmethod
    def _article_chapters(cls, markdown: str) -> list[str]:
        """Return the human-facing chapter names for a deterministic visible contents page."""
        chapters: list[str] = []
        for raw in str(markdown or "").splitlines():
            line = raw.strip()
            if not line.startswith("## "):
                continue
            title = cls._editorial_text(line[3:])
            if not title or "附录" in title or title in {"阅读与核对说明", "说明"}:
                continue
            chapters.append(title)
        return chapters

    @classmethod
    def _article_source_topics(cls, markdown: str) -> dict[str, list[str]]:
        """Map each article evidence tag to the closest human-readable subsection."""
        topics: dict[str, list[str]] = {}
        current_heading = ""
        for raw in str(markdown or "").splitlines():
            line = raw.strip()
            if line.startswith("### "):
                current_heading = cls._editorial_text(line[4:])
            elif line.startswith("#### "):
                current_heading = cls._editorial_text(line[5:])
            elif line.startswith("## "):
                current_heading = cls._editorial_text(line[3:])
            if not current_heading:
                continue
            for evidence_id in EVIDENCE_TAG.findall(line):
                values = topics.setdefault(evidence_id, [])
                if current_heading not in values:
                    values.append(current_heading)
        return topics

    @staticmethod
    def _select_article_images(
        article_source_ids: list[str],
        evidence: dict[str, dict[str, Any]],
        images: list[dict[str, Any]],
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        """Select evenly distributed, actually cited images instead of lexical guesses."""
        images_by_id = {str(item.get("id") or ""): item for item in images}
        cited: list[dict[str, Any]] = []
        seen: set[str] = set()
        for evidence_id in article_source_ids:
            row = evidence.get(evidence_id) or evidence.get(evidence_id[:8]) or {}
            asset_id = str(row.get("asset_id") or "")
            if row.get("kind") != "image" or not asset_id or asset_id in seen:
                continue
            image = images_by_id.get(asset_id)
            if image:
                cited.append(image)
                seen.add(asset_id)
        if len(cited) <= limit:
            return cited
        if limit <= 1:
            return cited[:limit]
        indexes = [round(index * (len(cited) - 1) / (limit - 1)) for index in range(limit)]
        return [cited[index] for index in dict.fromkeys(indexes)]

    def _record_audit(self, meeting_id: str, fmt: str, path: Path | None, result: QualityResult) -> None:
        self.db.insert(
            "export_audits",
            {
                "meeting_id": meeting_id,
                "format": fmt,
                "status": "PASSED" if result.passed else "FAILED",
                "path": str(path) if path else None,
                "checks_json": self.db.json(result.checks),
                "metrics_json": self.db.json(result.metrics),
                "errors_json": self.db.json(result.errors),
                "created_at": now_iso(),
            },
        )

    def _configure_document(self, document, title: str) -> None:
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.ns import qn
        from docx.shared import Inches, Mm, Pt, RGBColor

        section = document.sections[0]
        section.page_width = Mm(210)
        section.page_height = Mm(297)
        section.top_margin = Mm(22)
        section.bottom_margin = Mm(22)
        # Preserve the exact 9,360-DXA table canvas on an A4 page.
        section.left_margin = Mm(22.45)
        section.right_margin = Mm(22.45)
        section.header_distance = Inches(0.492)
        section.footer_distance = Inches(0.492)
        specs = {
            "Normal": (10.5, INK, 0, 6, 1.10),
            "Title": (26, NAVY, 0, 8, 1.0),
            "Subtitle": (14, DARK_BLUE, 0, 8, 1.0),
            "Heading 1": (16, BLUE, 16, 8, 1.0),
            "Heading 2": (13, BLUE, 12, 6, 1.0),
            "Heading 3": (12, DARK_BLUE, 8, 4, 1.0),
        }
        for style_name, (size, color, before, after, spacing) in specs.items():
            style = document.styles[style_name]
            style.font.name = FONT
            style.font.size = Pt(size)
            style.font.color.rgb = RGBColor.from_string(color)
            rfonts = style._element.get_or_add_rPr().get_or_add_rFonts()
            for key in ("ascii", "hAnsi", "eastAsia"):
                rfonts.set(qn(f"w:{key}"), FONT)
            style.paragraph_format.space_before = Pt(before)
            style.paragraph_format.space_after = Pt(after)
            style.paragraph_format.line_spacing = spacing
        for style_name in ("Heading 1", "Heading 2", "Heading 3"):
            document.styles[style_name].font.bold = True
        for style_name in ("List Bullet", "List Number"):
            style = document.styles[style_name]
            style.font.name = FONT
            style.font.size = Pt(10.5)
            style._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
            style.paragraph_format.left_indent = Inches(0.5)
            style.paragraph_format.first_line_indent = Inches(-0.25)
            style.paragraph_format.space_after = Pt(8)
            style.paragraph_format.line_spacing = 1.167

        header = section.header.paragraphs[0]
        header.text = f"{title}    ·    证据可回溯"
        header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for run in header.runs:
            _set_run(run, size=Pt(8.5), color=MUTED)
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_page_number(footer)

    @staticmethod
    def _add_callout(document, label: str, text: str, *, fill: str = PALE_BLUE, color: str = NAVY) -> None:
        from docx.shared import Pt

        table = document.add_table(rows=1, cols=1)
        _set_table_geometry(table, [CONTENT_DXA])
        cell = table.cell(0, 0)
        _shade_cell(cell, fill)
        paragraph = cell.paragraphs[0]
        paragraph.paragraph_format.space_after = Pt(0)
        _set_run(paragraph.add_run(f"{label}  "), size=Pt(10.5), color=color, bold=True)
        _set_run(paragraph.add_run(text), size=Pt(10.5), color=INK)

    @staticmethod
    def _add_bullet(document, text: str, citation: str = "") -> Any:
        from docx.shared import Pt

        paragraph = document.add_paragraph(style="List Bullet")
        _set_run(paragraph.add_run(text), size=Pt(10.5), color=INK)
        if citation:
            _set_run(paragraph.add_run(f"  {citation}"), size=Pt(8.2), color=MUTED)
        return paragraph

    @staticmethod
    def _add_table(document, headers: tuple[str, ...], rows: list[tuple[str, ...]], widths: list[int]):
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Pt

        table = document.add_table(rows=1, cols=len(headers))
        for index, text in enumerate(headers):
            _shade_cell(table.rows[0].cells[index], LIGHT)
            paragraph = table.rows[0].cells[index].paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if len(text) <= 6 else WD_ALIGN_PARAGRAPH.LEFT
            _set_run(paragraph.add_run(text), size=Pt(9.3), color=NAVY, bold=True)
        for row_values in rows:
            cells = table.add_row().cells
            for index, value in enumerate(row_values):
                paragraph = cells[index].paragraphs[0]
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if index < 2 and len(value) <= 12 else WD_ALIGN_PARAGRAPH.LEFT
                _set_run(paragraph.add_run(value), size=Pt(9.1), color=INK, bold=index == 1)
        _set_table_geometry(table, widths)
        return table

    @staticmethod
    def _add_figure(
        document,
        image: dict[str, Any],
        number: int,
        caption: str,
        relation: str,
        *,
        page_break_before: bool = False,
    ) -> None:
        from docx.shared import Pt

        from PIL import Image
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Inches, Pt

        path = Path(image["path"])
        with Image.open(path) as source:
            width_px, height_px = source.size
        scale = min(5.2 / max(width_px, 1), 3.0 / max(height_px, 1))
        shape = document.add_picture(
            str(path), width=Inches(width_px * scale), height=Inches(height_px * scale)
        )
        shape._inline.docPr.set("descr", f"图 {number}：{caption}。{relation}")
        shape._inline.docPr.set("title", path.name)
        picture = document.paragraphs[-1]
        picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
        picture.paragraph_format.keep_with_next = True
        picture.paragraph_format.page_break_before = page_break_before
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(10)
        _set_run(paragraph.add_run(f"图 {number:02d}｜{caption}\n{relation}"), size=Pt(8.2), color=MUTED)

    @staticmethod
    def _add_gallery(document, images: list[dict[str, Any]]) -> None:
        from PIL import Image
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Inches, Pt

        if not images:
            return
        heading = document.add_heading("完整图片索引", level=1)
        heading.paragraph_format.page_break_before = True
        paragraph = document.add_paragraph(
            f"以下按上传顺序保留全部 {len(images)} 张原图缩略索引。需核对文字时应回到原图；视觉模型解释和增强图不作为文字证据。"
        )
        paragraph.paragraph_format.space_after = Pt(8)
        columns = 6
        rows = math.ceil(len(images) / columns)
        table = document.add_table(rows=rows, cols=columns)
        for index, image in enumerate(images):
            cell = table.cell(index // columns, index % columns)
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            path = Path(image["path"])
            with Image.open(path) as source:
                width_px, height_px = source.size
            scale = min(0.86 / max(width_px, 1), 0.62 / max(height_px, 1))
            shape = paragraph.add_run().add_picture(
                str(path), width=Inches(width_px * scale), height=Inches(height_px * scale)
            )
            shape._inline.docPr.set("descr", f"图 {index + 1:02d} 缩略图：{ReportExporter._image_caption(image)}")
            shape._inline.docPr.set("title", path.name)
            caption = cell.add_paragraph()
            caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption.paragraph_format.space_after = Pt(0)
            _set_run(caption.add_run(f"图 {index + 1:02d}"), size=Pt(7.2), color=MUTED)
        _set_table_geometry(table, [1560] * columns)

    @staticmethod
    def _add_evidence_paragraph(document, label: str, text: str) -> None:
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.shared import Inches, Pt

        paragraph = document.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(0.18)
        paragraph.paragraph_format.right_indent = Inches(0.08)
        paragraph.paragraph_format.space_after = Pt(6)
        properties = paragraph._p.get_or_add_pPr()
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "F6F8FA")
        properties.append(shading)
        _set_run(paragraph.add_run(f"{label}  "), size=Pt(9.2), color=DARK_BLUE, bold=True)
        _set_run(paragraph.add_run(text), size=Pt(9.8), color=INK)

    def _add_detailed_discussion(
        self,
        document,
        topics: list[dict[str, Any]],
        images: list[dict[str, Any]],
        image_numbers: dict[str, int],
    ) -> None:
        from docx.shared import Pt

        heading = document.add_heading("详细讨论记录", level=1)
        heading.paragraph_format.page_break_before = True
        self._add_callout(
            document,
            "详细档阅读方式",
            "本章按正式议题展开，只呈现能够回到音频、文字稿或会议原图的讨论上下文。证据摘录保留原意，口语不顺或术语不确定处不自行补全。",
        )
        if not topics:
            document.add_paragraph("最新报告中暂无能够展开的正式议题。")
            return
        for index, item in enumerate(topics[:16], start=1):
            document.add_heading(f"{index}. {item['statement'][:96]}", level=2)
            evidence_rows = list(item.get("evidence", []))
            audio_count = sum(1 for row in evidence_rows if row.get("kind") == "audio")
            image_count = sum(1 for row in evidence_rows if row.get("kind") == "image")
            text_count = max(0, len(evidence_rows) - audio_count - image_count)
            coverage_parts = []
            if audio_count:
                coverage_parts.append(f"{audio_count} 个音频片段")
            if text_count:
                coverage_parts.append(f"{text_count} 条文字依据")
            if image_count:
                coverage_parts.append(f"{image_count} 个图片区域")
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.space_after = Pt(6)
            _set_run(paragraph.add_run("证据覆盖  "), size=Pt(10.2), color=DARK_BLUE, bold=True)
            coverage = "、".join(coverage_parts) if coverage_parts else "未找到可直接引用的原始材料"
            _set_run(paragraph.add_run(f"本议题由{coverage}支撑。"), size=Pt(10.5), color=INK)
            if evidence_rows:
                document.add_heading("代表性原始依据", level=3)
                for evidence_index, row in enumerate(evidence_rows[:3], start=1):
                    try:
                        locator = json.loads(row.get("locator_json") or "{}")
                    except (TypeError, json.JSONDecodeError):
                        locator = {}
                    if row.get("kind") == "audio":
                        start_ms = row.get("segment_start_ms") or locator.get("start_ms")
                        end_ms = row.get("segment_end_ms") or locator.get("end_ms")
                        speaker = str(row.get("speaker_name") or "未命名说话人")
                        label = f"证据 {evidence_index} · {speaker} · {_stamp(start_ms)}-{_stamp(end_ms)}"
                        excerpt = self._clean_excerpt(str(row.get("segment_text") or row.get("text") or ""))
                    elif row.get("kind") == "image":
                        number = image_numbers.get(str(row.get("asset_id") or ""))
                        label = f"证据 {evidence_index} · 图片 {number:02d}" if number else f"证据 {evidence_index} · 会议图片"
                        excerpt = self._clean_excerpt(str(row.get("text") or ""), 220)
                    else:
                        label = f"证据 {evidence_index} · 文字稿"
                        excerpt = self._clean_excerpt(str(row.get("text") or ""))
                    if excerpt:
                        self._add_evidence_paragraph(document, label, excerpt)
            related = self._topic_images(item, images)
            if related:
                document.add_heading("相关会议材料", level=3)
                for image in related:
                    number = image_numbers.get(str(image.get("id") or ""))
                    label = f"图 {number:02d}" if number else "会议图片"
                    self._add_bullet(
                        document,
                        f"{label}｜{self._image_caption(image)}｜仅用于定位原始材料；"
                        "未经人工确认的视觉模型解释不进入正式导出。",
                    )

    def _add_timeline(self, document, segments: list[dict[str, Any]]) -> None:
        from docx.shared import Pt

        document.add_heading("会议进程时间线", level=1)
        self._add_callout(
            document,
            "编排规则",
            "按 10 分钟（长会议按 15 分钟）划分阶段，每阶段仅选取两条信息量较高的原话摘录。时间线用于定位讨论过程，不作为润色后的会议结论，也不代替完整逐句转写。",
        )
        groups = self._timeline_groups(segments)
        if not groups:
            document.add_paragraph("没有能够进入详细时间线的可靠转写。")
            return
        for group in groups:
            document.add_heading(
                f"{_stamp(group['start_ms'])}-{_stamp(group['end_ms'])}",
                level=2,
            )
            for segment in group["segments"]:
                text = self._segment_text(segment)
                if not text:
                    continue
                speaker = str(segment.get("speaker_name") or segment.get("speaker_label") or "未命名说话人")
                paragraph = document.add_paragraph()
                paragraph.paragraph_format.space_after = Pt(6)
                _set_run(
                    paragraph.add_run(f"{_stamp(segment.get('start_ms'))} · {speaker}  "),
                    size=Pt(9.2),
                    color=DARK_BLUE,
                    bold=True,
                )
                _set_run(paragraph.add_run(text), size=Pt(10.2), color=INK)

    def _add_evidence_appendix(
        self,
        document,
        sections: dict[str, list[dict[str, Any]]],
        image_numbers: dict[str, int],
        *,
        page_break_before: bool = True,
    ) -> None:
        from docx.shared import Pt

        labels = {
            "topics": "议题与背景",
            "decisions": "会议决定",
            "actions": "行动项",
            "risks": "风险与异议",
            "questions": "未决问题",
        }
        heading = document.add_heading("正式事实与证据附录", level=1)
        heading.paragraph_format.page_break_before = page_break_before
        document.add_paragraph(
            "以下按事实类型列出正式报告使用的全部陈述及其证据位置，便于审阅者逐项复核。重复出现在摘要中的内容只保留一次。"
        )
        seen: set[str] = set()
        has_items = False
        for section in ("topics", "decisions", "actions", "risks", "questions"):
            items = []
            for item in sections.get(section, []):
                key = _normalize(item["statement"])
                if not key or key in seen:
                    continue
                seen.add(key)
                items.append(item)
            if not items:
                continue
            has_items = True
            document.add_heading(labels[section], level=2)
            for item in items:
                statement = item["statement"]
                evidence_kinds = {row.get("kind") for row in item.get("evidence", [])}
                if "audio" in evidence_kinds and re.search(
                    r"(?:[一二三四五六七八九十百千万\d.,]+\s*倍|\d[\d.,]*\s*[%％])",
                    statement,
                ):
                    statement += "（会议音频陈述；比较基线、对象与适用范围需核实）"
                paragraph = self._add_bullet(
                    document, statement, self._citation(item, image_numbers)
                )
                paragraph.paragraph_format.space_after = Pt(4)
                paragraph.paragraph_format.line_spacing = 1.05
        if not has_items:
            document.add_paragraph("没有能够进入正式事实附录的内容。")

    @staticmethod
    def _editorial_text(value: str) -> str:
        text = EVIDENCE_TAG.sub("", str(value or ""))
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _editorial_citation(value: str, source_labels: dict[str, str]) -> str:
        labels = [source_labels[item] for item in EVIDENCE_TAG.findall(str(value or "")) if item in source_labels]
        return " ".join(f"[{item}]" for item in dict.fromkeys(labels))

    def _add_static_contents(self, document, markdown: str) -> None:
        """Add a visible, update-free contents page; Word fields are deliberately avoided."""
        from docx.shared import Pt

        document.add_heading("目录", level=1)
        self._add_callout(
            document,
            "阅读结构",
            "报告分为技术内容整理文稿、会议纪要与核实边界、来源与证据附录三部分。"
            "目录为静态章节导航，不依赖 Word 手动更新域。",
        )
        document.add_heading("第一部分｜技术内容整理文稿", level=2)
        chapters = self._article_chapters(markdown)
        if chapters:
            for index, chapter in enumerate(chapters, start=1):
                paragraph = document.add_paragraph()
                paragraph.paragraph_format.left_indent = Pt(10)
                paragraph.paragraph_format.space_after = Pt(4)
                _set_run(paragraph.add_run(f"{index:02d}"), size=Pt(9), color=GOLD, bold=True)
                _set_run(paragraph.add_run(f"    {chapter}"), size=Pt(10.5), color=INK)
        else:
            document.add_paragraph("技术正文将在材料完成核对后生成。")
        document.add_heading("第二部分｜会议纪要与核实边界", level=2)
        document.add_paragraph("确认结论与量化边界 · 决策与行动状态 · 风险与待核实项")
        document.add_heading("第三部分｜来源与证据附录", level=2)
        document.add_paragraph("技术文稿来源索引 · 正式事实与证据附录 · 完整图片索引")

    @staticmethod
    def _image_relation(
        image: dict[str, Any],
        evidence_ids: list[str],
        source_labels: dict[str, str],
        source_topics: dict[str, list[str]],
        image_number: int,
    ) -> str:
        cited = [item for item in evidence_ids if item in source_labels]
        labels = "、".join(source_labels[item] for item in cited)
        topics: list[str] = []
        for evidence_id in cited:
            for topic in source_topics.get(evidence_id, []):
                if topic not in topics:
                    topics.append(topic)
        topic_text = "；".join(topics[:2]) or "当前技术小节"
        if labels:
            article_relation = f"正文引用：{labels} · 图{image_number:02d} → {topic_text}"
        else:
            article_relation = f"正文引用：未标注 · 图{image_number:02d}"
        if image.get("confirmed") and image.get("fact_statement"):
            fact_relation = f"图片—结构化事实：已人工确认（{str(image['fact_statement'])[:72]}）"
        elif image.get("fact_statement"):
            fact_relation = "图片—结构化事实：存在自动候选，待人工确认"
        else:
            fact_relation = "图片—结构化事实：尚未单独确认；不影响上述正文来源引用"
        return f"{article_relation}\n{fact_relation}"

    def _add_editorial_article(
        self,
        document,
        markdown: str,
        source_labels: dict[str, str],
        evidence: dict[str, dict[str, Any]] | None = None,
        images: list[dict[str, Any]] | None = None,
        image_numbers: dict[str, int] | None = None,
        selected_image_ids: set[str] | None = None,
    ) -> None:
        from docx.enum.text import WD_LINE_SPACING
        from docx.shared import Pt

        lines = str(markdown or "").splitlines()
        evidence = evidence or {}
        images_by_id = {str(item.get("id") or ""): item for item in (images or [])}
        image_numbers = image_numbers or {}
        selected_image_ids = selected_image_ids or set()
        source_topics = self._article_source_topics(markdown)
        inserted_images: set[str] = set()

        def add_cited_figures(value: str, current_heading: str) -> None:
            for evidence_id in dict.fromkeys(EVIDENCE_TAG.findall(value)):
                row = evidence.get(evidence_id) or evidence.get(evidence_id[:8]) or {}
                asset_id = str(row.get("asset_id") or "")
                if (
                    row.get("kind") != "image"
                    or asset_id not in selected_image_ids
                    or asset_id in inserted_images
                    or asset_id not in images_by_id
                ):
                    continue
                image = images_by_id[asset_id]
                number = image_numbers.get(asset_id)
                if not number:
                    continue
                related_ids = [
                    item for item in source_labels
                    if str((evidence.get(item) or {}).get("asset_id") or "") == asset_id
                ]
                full_relation = self._image_relation(
                    image,
                    related_ids or [evidence_id],
                    source_labels,
                    source_topics,
                    number,
                )
                # The technical narrative only needs the human-facing source
                # relationship. Structured-fact confirmation is a review-state
                # concern and is summarized once in the evidence appendix.
                relation = full_relation.splitlines()[0]
                caption = f"“{current_heading}”对应的现场演示材料" if current_heading else self._image_caption(image)
                # Keep the cited lead-in with the figure.  Without this, Word
                # can strand a subsection title or a one-line label at the foot
                # of the previous page while moving the image to the next one.
                if document.paragraphs:
                    document.paragraphs[-1].paragraph_format.keep_with_next = True
                self._add_figure(document, image, number, caption, relation)
                inserted_images.add(asset_id)

        index = 0
        current_heading = ""
        while index < len(lines):
            raw = lines[index].rstrip()
            line = raw.strip()
            if not line or line == "---":
                index += 1
                continue
            if line.startswith("# "):
                index += 1
                continue
            if line.startswith("## "):
                current_heading = self._editorial_text(line[3:])
                document.add_heading(current_heading, level=1)
                add_cited_figures(line, current_heading)
                index += 1
                continue
            if line.startswith("### "):
                current_heading = self._editorial_text(line[4:])
                document.add_heading(current_heading, level=2)
                add_cited_figures(line, current_heading)
                index += 1
                continue
            if line.startswith("#### "):
                current_heading = self._editorial_text(line[5:])
                document.add_heading(current_heading, level=3)
                add_cited_figures(line, current_heading)
                index += 1
                continue
            if line.startswith(">"):
                citation = self._editorial_citation(line, source_labels)
                self._add_callout(
                    document,
                    "内容导读",
                    " ".join(filter(None, [self._editorial_text(line.lstrip("> ")), citation])),
                    fill=PALE_BLUE,
                    color=DARK_BLUE,
                )
                add_cited_figures(line, current_heading)
                index += 1
                continue
            if line.startswith("|") and index + 1 < len(lines):
                separator = lines[index + 1].strip()
                if separator.startswith("|") and re.fullmatch(r"[|:\-\s]+", separator):
                    headers = [self._editorial_text(item) for item in line.strip("|").split("|")]
                    rows: list[tuple[str, ...]] = []
                    index += 2
                    table_source_lines: list[str] = [line]
                    while index < len(lines) and lines[index].strip().startswith("|"):
                        table_source_lines.append(lines[index])
                        cell_values: list[str] = []
                        for item in lines[index].strip().strip("|").split("|"):
                            value = " ".join(filter(None, [
                                self._editorial_text(item),
                                self._editorial_citation(item, source_labels),
                            ]))
                            cell_values.append(value or "材料未说明")
                        cells = tuple(cell_values)
                        if len(cells) == len(headers):
                            rows.append(cells)
                        index += 1
                    if rows and 1 <= len(headers) <= 4:
                        base, remainder = divmod(CONTENT_DXA, len(headers))
                        widths = [base + (1 if column < remainder else 0) for column in range(len(headers))]
                        self._add_table(document, tuple(headers), rows, widths)
                    add_cited_figures("\n".join(table_source_lines), current_heading)
                    continue
            if re.match(r"^[-*]\s+", line):
                self._add_bullet(
                    document,
                    self._editorial_text(re.sub(r"^[-*]\s+", "", line)),
                    self._editorial_citation(line, source_labels),
                )
                add_cited_figures(line, current_heading)
                index += 1
                continue
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.space_after = Pt(6)
            paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            paragraph.paragraph_format.line_spacing = 1.25
            _set_run(paragraph.add_run(self._editorial_text(line)), size=Pt(10.8), color=INK)
            citation = self._editorial_citation(line, source_labels)
            if citation:
                _set_run(paragraph.add_run(f"  {citation}"), size=Pt(8.2), color=MUTED)
            add_cited_figures(line, current_heading)
            index += 1

    def _add_manuscript_source_index(
        self,
        document,
        source_labels: dict[str, str],
        evidence: dict[str, dict[str, Any]],
        image_numbers: dict[str, int],
        markdown: str,
        *,
        page_break_before: bool = True,
    ) -> None:
        if not source_labels:
            return
        heading = document.add_heading("技术文稿来源索引", level=1)
        heading.paragraph_format.page_break_before = page_break_before
        document.add_paragraph(
            "正文中的 S 编号用于定位本次会议的原始材料。索引只展示清晰的位置关系；"
            "低质量 OCR 原文保留在证据库，不进入人类交付文档。"
        )

        source_topics = self._article_source_topics(markdown)

        rows: list[tuple[str, str, str]] = []
        for evidence_id, label in source_labels.items():
            item = evidence.get(evidence_id) or {}
            kind = str(item.get("kind") or "")
            if kind == "image":
                asset_id = str(item.get("asset_id") or "")
                number = image_numbers.get(asset_id)
                filename = Path(str(item.get("filename") or "会议图片")).name
                location = f"图 {number:02d} · {filename}" if number else filename
            elif kind == "audio":
                location = f"{_stamp(item.get('segment_start_ms'))}–{_stamp(item.get('segment_end_ms'))}"
                speaker = str(item.get("speaker_name") or "").strip()
                if speaker:
                    location += f" · {speaker}"
            else:
                location = str(item.get("filename") or "会议材料")
            topics = source_topics.get(evidence_id) or []
            topic_text = "；".join(topics[:2]) or "原始材料已保留，正文未标注独立小节"
            rows.append((label, location, f"正文引用：{topic_text}"))

        self._add_table(
            document,
            ("编号", "原始材料", "对应正文位置"),
            rows,
            (900, 3100, CONTENT_DXA - 4000),
        )

    def _build_docx(self, meeting_id: str, path: Path, detail: str) -> tuple[int, str]:
        from docx import Document
        from docx.enum.section import WD_SECTION
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Pt

        meeting, reports = self._latest(meeting_id)
        profile_label, profile_description = REPORT_DETAILS[detail]
        title = self._document_title(meeting["title"], detail)
        reports_by_section = {item["section"]: item for item in reports}
        editorial_markdown = str((reports_by_section.get("article") or {}).get("content_markdown") or "").strip()
        has_editorial = detail == "DETAILED" and "## " in editorial_markdown
        evidence = self._evidence(meeting_id)
        article_source_ids = list(dict.fromkeys(EVIDENCE_TAG.findall(editorial_markdown)))
        article_source_labels = {
            evidence_id: f"S{index:02d}" for index, evidence_id in enumerate(article_source_ids, start=1)
        }
        facts = self._facts(meeting_id)
        sections: dict[str, list[dict[str, Any]]] = {}
        pending: list[dict[str, Any]] = []
        for name in ("summary", "topics", "decisions", "actions", "risks", "questions"):
            formal, rejected = self._section_model(reports_by_section.get(name), facts, evidence)
            sections[name] = formal
            pending.extend(rejected)
        review_entries = self._report_entries(reports_by_section.get("review"))
        images = self._images(meeting_id)
        images.sort(key=lambda item: str(item.get("created_at") or ""))
        image_numbers = {item["id"]: index + 1 for index, item in enumerate(images)}
        formal_statements = [
            item["statement"] for name in sections for item in sections[name]
        ]
        image_limit = 4 if detail == "BRIEF" else 6 if has_editorial else 10 if detail == "DETAILED" else 8
        selected_images = (
            self._select_article_images(article_source_ids, evidence, images, limit=image_limit)
            if has_editorial
            else self._select_images(images, formal_statements, limit=image_limit)
        )

        segments = self.db.all(
            "SELECT ts.*, s.label AS speaker_label, s.display_name AS speaker_name "
            "FROM transcript_segments ts LEFT JOIN speakers s ON s.id=ts.speaker_id "
            "WHERE ts.meeting_id=? ORDER BY ts.start_ms",
            (meeting_id,),
        )
        speaker_count = int(
            (self.db.one("SELECT COUNT(*) AS value FROM speakers WHERE meeting_id=?", (meeting_id,)) or {}).get("value", 0)
        )
        duration_ms = max((int(item.get("end_ms") or 0) for item in segments), default=0)
        reliable_segments = sum(1 for item in segments if item.get("source_backend") != "degraded")
        confirmed_count = sum(len(sections[name]) for name in sections if name != "summary")
        decisions = sections["decisions"]
        actions = sections["actions"]
        topics = sections["topics"]
        summary = sections["summary"]
        quantitative_pattern = re.compile(
            r"\d|[一二三四五六七八九十百千万亿]+(?:倍|％|%|小时|分钟|年|月|日|台|兆|GB|TB|MHz|Mhz)",
            re.IGNORECASE,
        )
        metrics = [
            item for item in topics
            if item.get("kind") == "metric"
            or quantitative_pattern.search(item["statement"])
        ]
        terms = [item for item in topics if item.get("kind") == "term"]
        topic_only = [item for item in topics if item not in metrics and item not in terms]
        summary_display = [
            item for item in summary
            if item.get("kind") != "metric" and not quantitative_pattern.search(item["statement"])
        ]

        document = Document()
        self._configure_document(document, title)
        document.core_properties.title = title
        document.core_properties.subject = f"证据可回溯的{profile_label}"
        document.core_properties.author = "K3 会议智记"
        document.core_properties.keywords = "会议记录,证据,行动项,图片"
        document.core_properties.comments = "根据用户提供的会议录音、文字稿与现场演示材料整理"

        if any(row.get("needs_review", 1) or row.get("editorial_state") != "human_reviewed" for row in reports):
            status = document.add_paragraph("草稿 · 待人工核对，不作为已审核结论发布")
            status.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _set_run(status.runs[0], size=Pt(10), color=MUTED)

        spacer = document.add_paragraph()
        spacer.paragraph_format.space_after = Pt(86)
        kicker_text = "演讲内容整理文稿" if has_editorial else "会议纪要"
        kicker = document.add_paragraph(kicker_text)
        kicker.alignment = WD_ALIGN_PARAGRAPH.CENTER
        kicker.paragraph_format.space_after = Pt(18)
        _set_run(kicker.runs[0], size=Pt(10), color=GOLD, bold=True)
        title_paragraph = document.add_paragraph()
        title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_paragraph.paragraph_format.space_after = Pt(10)
        cover_title = title
        if len(title) > 30 and " — " in title:
            heading, descriptor_title = title.split(" — ", 1)
            cover_title = f"{heading}\n{descriptor_title}"
        _set_run(
            title_paragraph.add_run(cover_title),
            size=Pt(24 if len(title) > 22 else 28),
            color=NAVY,
            bold=True,
        )
        subtitle_text = (
            "技术内容梳理｜关键数据核对｜原始材料回溯"
            if has_editorial
            else f"{profile_label}｜{profile_description}"
        )
        subtitle = document.add_paragraph(subtitle_text)
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        subtitle.paragraph_format.space_after = Pt(24)
        _set_run(subtitle.runs[0], size=Pt(14), color=DARK_BLUE)
        descriptor = document.add_paragraph(
            f"基于 {_stamp(duration_ms)} 会议内容、{reliable_segments} 条可靠转写与 {len(images)} 张原图整理"
        )
        descriptor.alignment = WD_ALIGN_PARAGRAPH.CENTER
        descriptor.paragraph_format.space_after = Pt(68)
        _set_run(descriptor.runs[0], size=Pt(10), color=MUTED)
        meta = document.add_paragraph(
            "依据用户提供材料整理 · 非主办方官方出版物"
        )
        meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _set_run(meta.runs[0], size=Pt(9.5), color=MUTED, bold=True)
        document.add_section(WD_SECTION.NEW_PAGE)

        if has_editorial:
            self._add_static_contents(document, editorial_markdown)
            document.add_page_break()

        # A long-form technical manuscript should begin with the manuscript,
        # not with a meeting-style executive summary assembled from the small
        # normalized fact set.  Meeting metadata and status belong to part two.
        if not has_editorial:
            document.add_heading("执行摘要", level=1)
            if decisions:
                conclusion = f"本次会议形成 {len(decisions)} 项有证据支持的决定和 {len(actions)} 项行动；具体负责人、期限与证据见后文状态表。"
            elif summary_display or topic_only:
                lead = (summary_display or topic_only)[0]["statement"].rstrip("。；")
                conclusion = f"本次会议以“{lead[:72]}”为主要内容；未识别到有充分证据支持的正式决议。"
            else:
                conclusion = "现有材料不足以形成正式会议结论；报告仅保留可回溯内容和待核实项。"
            self._add_callout(document, "一句话结论", conclusion, fill=PALE_GOLD, color="7A5A00")
            summary_limit = 4 if detail == "BRIEF" else 8 if detail == "DETAILED" else 6
            for item in summary_display[:summary_limit]:
                self._add_bullet(document, item["statement"], self._citation(item, image_numbers))
            if not summary_display:
                self._add_bullet(document, "暂无适合进入执行摘要的非量化事实，请以议题、量化边界、决定与行动状态为准。")

            document.add_heading("会议概况", level=2)
            meeting_type = "决策 / 工作会议" if decisions or actions else "分享 / 讨论会议"
            metadata_rows = [
                ("会议性质", meeting_type),
                ("内容时长", _stamp(duration_ms)),
                ("语言", self._language_label(segments)),
                ("内容来源", f"{reliable_segments} 条可靠转写；{len(images)} 张会议原图"),
                ("说话人", f"{speaker_count} 个说话人簇；未实名者不推断身份"),
                ("事实状态", f"{confirmed_count} 条正式章节事实；{len(pending) + len(review_entries)} 条/项进入核实边界"),
            ]
            self._add_table(document, ("项目", "说明"), metadata_rows, [2200, 7160])
            document.add_heading("信息边界", level=2)
            self._add_callout(
                document,
                "阅读规则",
                "正文只纳入最新报告版本中能够定位原始证据的内容。图片中的产品能力、路线图和数字按“会议材料陈述”处理；模糊或冲突内容不推测补全。人工编辑但缺少证据的内容进入待核实项。",
            )

        if has_editorial:
            part_heading = document.add_heading("第一部分｜技术内容整理文稿", level=1)
            part_heading.paragraph_format.page_break_before = True
            self._add_editorial_article(
                document,
                editorial_markdown,
                article_source_labels,
                evidence,
                images,
                image_numbers,
                {str(item.get("id") or "") for item in selected_images},
            )
        else:
            document.add_heading("议题与逻辑脉络", level=1)
            if topic_only:
                topic_limit = 4 if detail == "BRIEF" else 16 if detail == "DETAILED" else 12
                for index, item in enumerate(topic_only[:topic_limit], start=1):
                    document.add_heading(f"{index}  {item['statement'][:72]}", level=2)
                    paragraph = document.add_paragraph()
                    _set_run(paragraph.add_run("证据定位与上下文见原始材料。"), size=Pt(10.5), color=INK)
                    citation = self._citation(item, image_numbers)
                    if citation:
                        _set_run(paragraph.add_run(f"  {citation}"), size=Pt(8.2), color=MUTED)
            else:
                document.add_paragraph("最新报告版本中暂无可独立确认的议题。")

            if detail == "DETAILED":
                self._add_detailed_discussion(document, topic_only, images, image_numbers)
                self._add_timeline(document, segments)

        if has_editorial:
            part_heading = document.add_heading("第二部分｜会议纪要与核实边界", level=1)
            part_heading.paragraph_format.page_break_before = True
            self._add_callout(
                document,
                "本部分的作用",
                "将会议中可确认的事实、量化口径和未决事项与技术讲稿分开呈现，"
                "避免把技术分享误读为已经形成的会议决策。",
            )
            document.add_heading("会议记录概况", level=2)
            meeting_type = "决策 / 工作会议" if decisions or actions else "分享 / 讨论会议"
            self._add_table(
                document,
                ("项目", "说明"),
                [
                    ("会议性质", meeting_type),
                    ("内容时长", _stamp(duration_ms)),
                    ("语言", self._language_label(segments)),
                    ("内容来源", f"{reliable_segments} 条可靠转写；{len(images)} 张会议原图"),
                    ("说话人", f"{speaker_count} 个说话人簇；未实名者不推断身份"),
                ],
                [2200, 7160],
            )
            self._add_callout(
                document,
                "核实边界",
                "图片中的产品能力、路线图和数字按“会议材料陈述”处理；"
                "模糊或冲突内容不推测补全。技术解释仍需领域人员对照原图做最终语义复核。",
            )
        document.add_heading("确认结论与量化边界", level=1)
        summary_keys = {_normalize(item["statement"]) for item in summary_display}
        conclusion_items = [item for item in topic_only if _normalize(item["statement"]) not in summary_keys]
        conclusion_limit = 4 if detail == "BRIEF" else 12 if detail == "DETAILED" else 8
        for item in conclusion_items[:conclusion_limit]:
            self._add_bullet(document, item["statement"], self._citation(item, image_numbers))
        if metrics:
            metric_rows: list[tuple[str, ...]] = []
            metric_limit = 5 if detail == "BRIEF" else 16 if detail == "DETAILED" else 10
            for item in metrics[:metric_limit]:
                kinds = {row.get("kind") for row in item.get("evidence", [])}
                if "audio" in kinds and re.search(r"(?:倍|%|％|加速|提升)", item["statement"]):
                    boundary = "音频陈述；比较基线、对象与适用范围需核实"
                elif "audio" in kinds:
                    boundary = "音频陈述；保留上下文"
                else:
                    boundary = "会议材料陈述；非本次复测结果"
                metric_rows.append((item["statement"][:82], self._citation(item, image_numbers), boundary))
            self._add_table(document, ("量化陈述", "证据", "报告处理"), metric_rows, [3900, 2100, 3360])
        else:
            self._add_callout(document, "量化信息", "未发现能够进入正式章节的量化陈述；不从模糊图片或上下文缺失的数字推导结论。")

        document.add_heading("决策与行动状态", level=1)
        status_rows: list[tuple[str, ...]] = []
        if decisions:
            for item in decisions:
                status_rows.append(("会议决策", item.get("status") or "已记录", item["statement"], self._citation(item, image_numbers)))
        else:
            status_rows.append(("会议决策", "未形成", "未发现有充分证据支持的明确表决或通过事项。", "-"))
        if actions:
            for item in actions:
                owner_due = " / ".join(filter(None, [item.get("owner") or "负责人未指定", item.get("due_at") or "期限未指定"]))
                status_rows.append(("行动项", owner_due, item["statement"], self._citation(item, image_numbers)))
        else:
            status_rows.append(("行动项", "未形成", "未识别到同时具备任务内容与可回溯证据的行动项。", "-"))
        self._add_table(document, ("类型", "状态 / 责任", "内容", "证据"), status_rows, [1300, 1900, 4160, 2000])

        document.add_heading("风险、异议与未决问题", level=2)
        risk_question = [("风险", item) for item in sections["risks"]] + [("问题", item) for item in sections["questions"]]
        if risk_question:
            for label, item in risk_question[:12]:
                self._add_bullet(document, f"{label}：{item['statement']}", self._citation(item, image_numbers))
        else:
            document.add_paragraph("最新报告版本中暂无有充分证据支持的风险、异议或未决问题。")

        if terms:
            document.add_heading("术语与专有名词", level=2)
            for item in terms[:12]:
                self._add_bullet(document, item["statement"], self._citation(item, image_numbers))

        document.add_heading("待核实项", level=2)
        review_count = len(pending) + len(review_entries)
        if review_count:
            review_preview = (pending + review_entries)[:4]
            self._add_callout(
                document,
                "不进入正式结论",
                f"共 {review_count} 条人工编辑、低置信或证据不完整内容被隔离。以下仅列 {len(review_preview)} 条代表项，完整内容保留在校对工作台。",
                fill=PALE_RED,
                color="8A2F2F",
            )
            for item in review_preview:
                self._add_bullet(document, item["statement"][:180])
        else:
            document.add_paragraph("没有被质量门禁隔离的待核实内容。")

        if selected_images and not has_editorial:
            heading = document.add_heading("关键图片证据", level=1)
            heading.paragraph_format.page_break_before = True
            intro = document.add_paragraph(
                f"从 {len(images)} 张原图中按人工确认、关联分、识别置信度和内容完整度选取 {len(selected_images)} 张重点展示；其余图片在完整索引中保留。"
            )
            intro.paragraph_format.space_after = Pt(8)
            for index, image in enumerate(selected_images):
                number = image_numbers[image["id"]]
                caption = self._image_caption(image)
                if image.get("confirmed") and image.get("fact_statement"):
                    relation = f"图片—结构化事实：已人工确认（{str(image['fact_statement'])[:100]}）"
                elif image.get("fact_statement") and float(image.get("score") or 0) >= 0.30:
                    relation = "图片—结构化事实：存在自动候选，待人工确认"
                else:
                    relation = "图片—结构化事实：尚未单独确认；图片仍保留在原始材料索引"
                self._add_figure(
                    document,
                    image,
                    number,
                    caption,
                    relation,
                    page_break_before=(bool(index) if detail == "DETAILED" else bool(index and index % 2 == 0)),
                )
        if detail == "DETAILED":
            part_heading = document.add_heading("第三部分｜来源与证据附录", level=1)
            part_heading.paragraph_format.page_break_before = True
            self._add_callout(
                document,
                "证据阅读方式",
                "S 编号对应正文实际引用的原始材料；图片与正文引用、图片与结构化事实的确认状态分别记录。"
                "本版图片可作为正文来源引用，但尚未完成图片—结构化事实的逐条人工确认；"
                "因此不据此自动提升正式事实。未经人工确认的 OCR/VLM 候选文字不会出现在本正式导出中。",
            )
            self._add_manuscript_source_index(
                document,
                article_source_labels,
                evidence,
                image_numbers,
                editorial_markdown,
                page_break_before=False,
            )
            self._add_evidence_appendix(
                document,
                sections,
                image_numbers,
                page_break_before=False,
            )
        if detail != "BRIEF":
            self._add_gallery(document, images)

        path.parent.mkdir(parents=True, exist_ok=True)
        document.save(path)
        if detail == "BRIEF":
            expected_max_pages = max(6, 5 + math.ceil(len(selected_images) / 2))
        elif detail == "DETAILED":
            expected_max_pages = max(
                16,
                10
                + len(selected_images)
                + math.ceil(len(images) / 36)
                + math.ceil(len(article_source_ids) / 8)
                + (
                    math.ceil(len(editorial_markdown) / 900)
                    if has_editorial
                    else len(topic_only[:16]) * 2 + math.ceil(len(self._timeline_groups(segments)) / 2)
                ),
            )
        else:
            expected_max_pages = max(8, 6 + math.ceil(len(selected_images) / 2) + math.ceil(len(images) / 36))
        return expected_max_pages, title

    def docx(self, meeting_id: str, *, detail: str | None = None, _render_gate: bool = True) -> Path:
        meeting, reports = self._latest(meeting_id)
        resolved_detail = self.detail_for(meeting_id, detail)
        if resolved_detail == "DETAILED":
            article = next((row for row in reports if row["section"] == "article"), None) or {}
            media = next((row for row in reports if row["section"] == "media"), None) or {}
            manuscript = str(article.get("content_markdown") or "")
            editorial_state = str(article.get("editorial_state") or "automatic_model_draft")
            evidence_lookup = self._evidence(meeting_id)
            unique_evidence = {
                str(row.get("id") or ""): row
                for row in evidence_lookup.values()
                if row.get("id")
            }
            image_asset_ids = {
                str(row.get("id") or "") for row in self._images(meeting_id) if row.get("id")
            }
            media_links = self.db.all(
                "SELECT ml.*, f.statement AS fact_statement FROM media_links ml "
                "LEFT JOIN facts f ON f.id=ml.fact_id WHERE ml.meeting_id=?",
                (meeting_id,),
            )
            detailed_source_count = int((self.db.one(
                "SELECT COUNT(*) AS value FROM assets WHERE meeting_id=?", (meeting_id,)
            ) or {}).get("value", 0)) + int((self.db.one(
                "SELECT COUNT(*) AS value FROM transcript_segments WHERE meeting_id=?", (meeting_id,)
            ) or {}).get("value", 0))
            manuscript_gate = audit_detailed_manuscript(
                manuscript,
                evidence=list(unique_evidence.values()),
                image_asset_ids=image_asset_ids,
                media_markdown=str(media.get("content_markdown") or ""),
                media_links=media_links,
                editorial_state=editorial_state,
            )
            # A sparse meeting truthfully exports an insufficiency note rather
            # than a publishable technical manuscript. Apply the strict claim
            # gate once the same material-rich threshold used by the depth gate
            # is reached; otherwise structural DOCX QA remains authoritative.
            if detailed_source_count >= 8:
                manuscript_gate.require()
        output_dir = self.settings.data_dir / "exports" / meeting_id
        output_dir.mkdir(parents=True, exist_ok=True)
        profile_label = REPORT_DETAILS[resolved_detail][0]
        path = output_dir / f"{_safe_name(self._document_title(meeting['title'], resolved_detail))}-{profile_label}.docx"
        expected_max_pages, title = self._build_docx(meeting_id, path, resolved_detail)
        structural = audit_docx(path)
        structural.metrics["report_detail"] = resolved_detail
        if resolved_detail == "DETAILED":
            article = next((row for row in reports if row["section"] == "article"), None)
            manuscript = str((article or {}).get("content_markdown") or "")
            source_count = int((self.db.one(
                "SELECT COUNT(*) AS value FROM assets WHERE meeting_id=?", (meeting_id,)
            ) or {}).get("value", 0)) + int((self.db.one(
                "SELECT COUNT(*) AS value FROM transcript_segments WHERE meeting_id=?", (meeting_id,)
            ) or {}).get("value", 0))
            image_source_count = int((self.db.one(
                "SELECT COUNT(*) AS value FROM assets WHERE meeting_id=? AND kind='image'", (meeting_id,)
            ) or {}).get("value", 0))
            cited_source_count = len(set(EVIDENCE_TAG.findall(manuscript)))
            chapter_count = len(re.findall(r"^## (?!附录)", manuscript, re.MULTILINE))
            subsection_count = len(re.findall(r"^### ", manuscript, re.MULTILINE))
            structural.metrics.update({
                "manuscript_characters": len(manuscript),
                "manuscript_chapters": chapter_count,
                "manuscript_subsections": subsection_count,
                "source_items": source_count,
                "image_source_items": image_source_count,
                "cited_sources": cited_source_count,
            })
            # Sparse meetings may truthfully produce a short insufficiency note.
            # Material-rich meetings, however, must never be delivered as a
            # renamed summary when the user selected the manuscript profile.
            if source_count >= 8:
                structural.check(
                    "detailed_manuscript_depth",
                    len(manuscript) >= 3_000 and chapter_count >= 5 and subsection_count >= 5,
                    "技术整理文稿深度不足（需至少 3000 字符、5 个技术章节和 5 个小节）",
                )
            if image_source_count >= 8:
                structural.check(
                    "detailed_image_source_coverage",
                    cited_source_count >= math.ceil(image_source_count * 0.9),
                    "技术整理文稿未覆盖至少 90% 的会议图片材料",
                )
        self._record_audit(meeting_id, "docx", path, structural)
        structural.require()
        if _render_gate and self.settings.export_require_render:
            preview = output_dir / f".{path.stem}.qa.pdf"
            try:
                convert_docx_to_pdf(path, preview)
                rendered = audit_pdf(preview, expected_max_pages=expected_max_pages, required_text=title)
                rendered.metrics["report_detail"] = resolved_detail
                self._record_audit(meeting_id, "docx", path, rendered)
                rendered.require()
            finally:
                preview.unlink(missing_ok=True)
        return path

    def pdf(self, meeting_id: str, *, detail: str | None = None) -> Path:
        meeting, reports = self._latest(meeting_id)
        resolved_detail = self.detail_for(meeting_id, detail)
        docx_path = self.docx(meeting_id, detail=resolved_detail, _render_gate=False)
        image_count = len(self._images(meeting_id))
        selected_count = min(image_count, 4 if resolved_detail == "BRIEF" else 6 if resolved_detail == "DETAILED" else 8)
        if resolved_detail == "BRIEF":
            expected_max_pages = max(6, 5 + math.ceil(selected_count / 2))
        elif resolved_detail == "DETAILED":
            article = next((row for row in reports if row["section"] == "article"), None)
            manuscript = str((article or {}).get("content_markdown") or "")
            manuscript_length = len(manuscript)
            manuscript_sources = len(set(EVIDENCE_TAG.findall(manuscript)))
            expected_max_pages = max(
                16,
                10
                + selected_count
                + math.ceil(image_count / 36)
                + math.ceil(manuscript_sources / 8)
                + math.ceil(manuscript_length / 900),
            )
        else:
            expected_max_pages = max(8, 6 + math.ceil(selected_count / 2) + math.ceil(image_count / 36))
        pdf_path = docx_path.with_suffix(".pdf")
        try:
            convert_docx_to_pdf(docx_path, pdf_path)
            result = audit_pdf(
                pdf_path,
                expected_max_pages=expected_max_pages,
                required_text=self._document_title(meeting["title"], resolved_detail),
            )
            result.metrics["report_detail"] = resolved_detail
            self._record_audit(meeting_id, "pdf", pdf_path, result)
            result.require()
        except (ExportQualityError, RuntimeError) as exc:
            if not isinstance(exc, ExportQualityError):
                failed = QualityResult("pdf", passed=False, errors=[str(exc)])
                self._record_audit(meeting_id, "pdf", pdf_path if pdf_path.exists() else None, failed)
            pdf_path.unlink(missing_ok=True)
            raise
        return pdf_path
