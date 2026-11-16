from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from PIL import Image


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
NS = {"w": W_NS, "wp": WP_NS}
EVIDENCE_TAG = re.compile(r"\[证据:([0-9a-fA-F]{8,32})\]")


class ExportQualityError(RuntimeError):
    """Raised when an export cannot pass the product delivery gate."""


@dataclass
class QualityResult:
    artifact: str
    passed: bool = True
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    verification_level: str = "artifact_integrity_only"
    limitations: list[str] = field(default_factory=list)

    def check(self, name: str, passed: bool, message: str) -> None:
        self.checks[name] = passed
        if not passed:
            self.passed = False
            self.errors.append(message)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))

    def require(self) -> "QualityResult":
        if not self.passed:
            raise ExportQualityError("报告质量门禁未通过：" + "；".join(self.errors))
        return self


def _normalized_numbers(value: str) -> set[str]:
    """Return comparable numeric values, excluding evidence IDs and list numbering."""
    value = EVIDENCE_TAG.sub("", value)
    candidates = re.findall(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?", value)
    normalized: set[str] = set()
    for candidate in candidates:
        token = candidate.replace(",", "")
        try:
            token = f"{float(token):.12g}"
        except ValueError:
            continue
        normalized.add(token)
    return normalized


def _is_critical_claim(value: str) -> bool:
    without_tags = EVIDENCE_TAG.sub("", value)
    if re.search(
        r"(?:\d[\d,.]*\s*(?:%|％|倍|[kKmMgGtT]?Hz|[TGMK]?B|小时|分钟|秒|天|月|年|台|核|亿|万))"
        r"|(?:19|20)\d{2}|(?:路线图|roadmap|计划|预计|目标|吞吐|性能|频率|延迟|加速|功耗|容量|成本|TaT|KA)",
        without_tags,
        re.IGNORECASE,
    ):
        return True
    return False


def _content_units(markdown: str) -> list[tuple[int, str]]:
    units: list[tuple[int, str]] = []
    in_fence = False
    for number, raw_line in enumerate(markdown.splitlines(), 1):
        line = raw_line.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line or re.fullmatch(r"[-|: ]+", line):
            continue
        units.append((number, line))
    return units


def audit_detailed_manuscript(
    markdown: str,
    *,
    evidence: list[dict[str, Any]],
    image_asset_ids: set[str] | None = None,
    media_markdown: str = "",
    media_links: list[dict[str, Any]] | None = None,
    editorial_state: str = "unknown",
) -> QualityResult:
    """Audit a detailed manuscript without claiming model-level semantic truth.

    The gate is deliberately deterministic: it checks reader-facing contamination,
    resolvable citations, and whether numbers in a claim occur in its cited source.
    It cannot decide whether a paraphrase is technically correct, so a passing
    result always retains an explicit human-review requirement.
    """
    result = QualityResult("detailed_manuscript")
    result.verification_level = "deterministic_source_consistency"
    result.limitations = [
        "自动门禁不能判断技术解释、术语翻译或因果关系是否正确",
        "自动门禁不能证明图片 OCR/VLM 转写本身正确",
        "正式对外发布仍需领域人员完成语义复核",
    ]
    image_asset_ids = image_asset_ids or set()
    media_links = media_links or []
    evidence_by_id = {str(item.get("id") or ""): item for item in evidence if item.get("id")}

    def resolve(reference: str) -> tuple[str | None, str | None]:
        matches = [identifier for identifier in evidence_by_id if identifier.startswith(reference)]
        if len(matches) == 1:
            return matches[0], None
        if not matches:
            return None, "missing"
        return None, "ambiguous"

    units = _content_units(markdown)
    empty_labels: list[int] = []
    isolated_arrows: list[int] = []
    truncated_lines: list[int] = []
    internal_state_lines: list[int] = []
    candidate_content_lines: list[int] = []
    missing_references: list[dict[str, Any]] = []
    critical_without_evidence: list[int] = []
    numeric_mismatches: list[dict[str, Any]] = []
    cited_ids: set[str] = set()

    human_edited = editorial_state in {"human_edited", "user_edited", "user_edited_locked"}
    for line_number, line in units:
        references = EVIDENCE_TAG.findall(line)
        clean = EVIDENCE_TAG.sub("", line).strip()
        clean = re.sub(r"^#{1,6}\s+", "", clean).strip()
        content = re.sub(r"^(?:[-*+]\s+|\d+[.)、]\s+)", "", clean).strip()
        # A stand-alone bullet label renders like an empty item even if nested
        # bullets happen to follow it. Non-list lead-ins before a table/list are
        # legitimate section scaffolding and are not treated as empty.
        if re.fullmatch(r"[-*+]\s+[^|]{1,60}[：:]", clean):
            empty_labels.append(line_number)
        if re.match(r"^(?:→|⇒|->|=>|←)", content) or re.search(r"(?:→|⇒|->|=>|←)$", content):
            isolated_arrows.append(line_number)
        if (
            re.search(r"[，,、；;（(【\[]$", content)
            or re.search(r"(?:以及|并且|或者|包括|例如|通过|达到|提升到|为|是|和|与|及)$", content)
            or re.search(r"\b(?:the|and|or|to|of|with|for)$", content, re.IGNORECASE)
            or content.count("(") != content.count(")")
            or content.count("（") != content.count("）")
        ):
            truncated_lines.append(line_number)
        if re.search(
            r"(?:\[\[MAFN|\[EVIDENCE:|LOCAL_ONLY|HYBRID_REVIEW|\bDETAILED\b|\bSTANDARD\b|报告\s*V\d+|llama-server)",
            content,
            re.IGNORECASE,
        ):
            internal_state_lines.append(line_number)
        if re.search(
            r"(?:视觉理解候选|视觉候选信息|OCR\s*候选|需核对原图|仅作为会议材料保留|尚未建立(?:可靠的)?文字事实关联)",
            content,
            re.IGNORECASE,
        ):
            candidate_content_lines.append(line_number)

        resolved_for_line: list[str] = []
        for reference in references:
            identifier, issue = resolve(reference)
            if issue:
                missing_references.append({"line": line_number, "reference": reference, "issue": issue})
            elif identifier:
                cited_ids.add(identifier)
                resolved_for_line.append(identifier)

        # Headings may contain dates as labels; the supporting paragraph/table
        # carries the evidence. Claims in prose and table rows must cite directly.
        is_heading = line.startswith("#")
        is_table_glossary = line.startswith("|") and not _normalized_numbers(content)
        is_editorial_footer = line.startswith("*") and line.endswith("*")
        if not is_heading and not is_table_glossary and not is_editorial_footer and _is_critical_claim(content):
            if not resolved_for_line:
                critical_without_evidence.append(line_number)
            else:
                claim_numbers = _normalized_numbers(content)
                source_text = "\n".join(str(evidence_by_id[item].get("text") or "") for item in resolved_for_line)
                source_numbers = _normalized_numbers(source_text)
                missing_numbers = sorted(claim_numbers - source_numbers)
                if missing_numbers:
                    numeric_mismatches.append(
                        {"line": line_number, "numbers": missing_numbers, "references": references}
                    )

    cited_image_assets = {
        str(evidence_by_id[item].get("asset_id") or "")
        for item in cited_ids
        if str(evidence_by_id[item].get("kind") or "") == "image"
    } - {""}
    missing_image_assets = sorted(image_asset_ids - cited_image_assets)

    relation_conflicts: list[dict[str, Any]] = []
    for line_number, line in _content_units(media_markdown):
        if not re.search(r"(?:仅作为会议材料保留|尚未建立(?:可靠的)?文字事实关联)", line):
            continue
        for reference in EVIDENCE_TAG.findall(line):
            identifier, issue = resolve(reference)
            if not issue and identifier in cited_ids:
                relation_conflicts.append({"line": line_number, "reference": reference})

    linked_assets = {str(item.get("asset_id") or "") for item in media_links if item.get("confirmed")}
    cited_without_confirmed_fact_link = sorted(cited_image_assets - linked_assets)
    numeric_visual_review_allowed = bool(numeric_mismatches) and human_edited and all(
        all(
            (resolved := resolve(reference))[0]
            and str(evidence_by_id[resolved[0]].get("kind") or "") == "image"
            for reference in issue["references"]
        )
        for issue in numeric_mismatches
    )
    if numeric_visual_review_allowed:
        result.verification_level = "human_edited_source_linked_visual_numeric_review_required"
        result.limitations.append(
            "部分数字未出现在 OCR/VLM 文本层；已保留图片来源引用，但发布前仍须对照原图核验"
        )
    result.metrics.update(
        {
            "editorial_state": editorial_state,
            "content_units": len(units),
            "cited_evidence": len(cited_ids),
            "cited_image_assets": len(cited_image_assets),
            "total_image_assets": len(image_asset_ids),
            "image_source_coverage": round(len(cited_image_assets) / max(1, len(image_asset_ids)), 4),
            "empty_label_lines": empty_labels,
            "isolated_arrow_lines": isolated_arrows,
            "truncated_line_candidates": truncated_lines,
            "internal_state_lines": internal_state_lines,
            "unconfirmed_candidate_content_lines": candidate_content_lines,
            "unresolved_references": missing_references,
            "critical_claims_without_evidence_lines": critical_without_evidence,
            "numeric_mismatches": numeric_mismatches,
            "critical_numbers_exact_match": not numeric_mismatches,
            "numeric_visual_review_allowed": numeric_visual_review_allowed,
            "missing_image_asset_ids": missing_image_assets,
            "media_relation_conflicts": relation_conflicts,
            "cited_images_without_confirmed_fact_link": cited_without_confirmed_fact_link,
        }
    )
    result.check("no_empty_labels", not empty_labels, f"发现空标签行：{empty_labels}")
    result.check("no_isolated_arrows", not isolated_arrows, f"发现孤立箭头表达：{isolated_arrows}")
    result.check("no_truncated_sentences", not truncated_lines, f"发现疑似截断句：{truncated_lines}")
    result.check("no_internal_state_tokens", not internal_state_lines, f"发现内部状态词或工具标记：{internal_state_lines}")
    result.check("no_unconfirmed_candidate_content", not candidate_content_lines, f"未经确认的 OCR/VLM 候选进入正式正文：{candidate_content_lines}")
    result.check("all_references_resolve", not missing_references, "正文包含不存在或不唯一的证据引用")
    result.check("critical_claims_have_evidence", not critical_without_evidence, f"关键数字、日期、路线图或性能主张缺少证据：{critical_without_evidence}")
    result.check(
        "critical_numbers_have_reviewable_source",
        not numeric_mismatches or numeric_visual_review_allowed,
        "关键主张中的数字未出现在所引来源，且不是带原图引用的人工修订内容",
    )
    result.check("all_images_cited", not missing_image_assets, f"{len(missing_image_assets)} 张图片未被正文引用")
    result.check("no_media_relation_conflicts", not relation_conflicts, "正文已引用图片，但媒体章节仍声称其未建立可靠关联")
    return result


def audit_docx(path: Path) -> QualityResult:
    result = QualityResult("docx")
    result.check("exists", path.exists() and path.stat().st_size > 1_000, "DOCX 文件不存在或为空")
    if not result.passed:
        return result

    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            result.check("zip_integrity", bad_member is None, f"DOCX 压缩包损坏：{bad_member}")
            names = set(archive.namelist())
            result.check("document_xml", "word/document.xml" in names, "DOCX 缺少 document.xml")
            if "word/document.xml" not in names:
                return result
            root = ET.fromstring(archive.read("word/document.xml"))
    except (OSError, zipfile.BadZipFile, ET.ParseError) as exc:
        result.check("open", False, f"DOCX 无法解析：{type(exc).__name__}")
        return result

    text = "".join(node.text or "" for node in root.findall(".//w:t", NS))
    result.metrics["text_characters"] = len(text)
    result.check("minimum_content", len(text) >= 120, "报告正文过短，无法构成可交付会议报告")
    meeting_report_sections = all(
        label in text for label in ("执行摘要", "会议概况", "信息边界", "决策与行动状态")
    )
    technical_manuscript_sections = all(
        label in text
        for label in (
            "第一部分｜技术内容整理文稿",
            "第二部分｜会议纪要与核实边界",
            "会议记录概况",
            "决策与行动状态",
            "第三部分｜来源与证据附录",
        )
    )
    result.check(
        "required_sections",
        meeting_report_sections or technical_manuscript_sections,
        "报告缺少会议报告必要章节，或技术文稿的三部分阅读结构",
    )
    result.check("no_internal_tokens", "[[MAFN" not in text and "[证据:" not in text, "报告泄露内部证据标记")
    result.check("no_placeholder", "TODO" not in text and "待生成" not in text, "报告含未完成占位符")

    heading_levels: list[int] = []
    for paragraph in root.findall(".//w:body/w:p", NS):
        style = paragraph.find("./w:pPr/w:pStyle", NS)
        value = style.get(f"{{{W_NS}}}val", "") if style is not None else ""
        match = re.fullmatch(r"Heading(\d+)", value)
        if match:
            heading_levels.append(int(match.group(1)))
    result.metrics["heading_count"] = len(heading_levels)
    hierarchy_ok = bool(heading_levels) and all(
        current <= previous + 1 for previous, current in zip(heading_levels, heading_levels[1:])
    )
    result.check("heading_hierarchy", hierarchy_ok, "标题层级缺失或发生跨级")

    drawings = root.findall(".//wp:docPr", NS)
    missing_alt = [node.get("id", "?") for node in drawings if not (node.get("descr") or "").strip()]
    result.metrics["image_count"] = len(drawings)
    result.check("image_alt_text", not missing_alt, f"{len(missing_alt)} 张图片缺少替代文本")

    tables = root.findall(".//w:tbl", NS)
    geometry_errors = 0
    header_errors = 0
    for table in tables:
        width = table.find("./w:tblPr/w:tblW", NS)
        indent = table.find("./w:tblPr/w:tblInd", NS)
        layout = table.find("./w:tblPr/w:tblLayout", NS)
        grid = table.findall("./w:tblGrid/w:gridCol", NS)
        grid_widths = [int(node.get(f"{{{W_NS}}}w", "0")) for node in grid]
        if (
            width is None
            or width.get(f"{{{W_NS}}}type") != "dxa"
            or int(width.get(f"{{{W_NS}}}w", "0")) != 9360
            or indent is None
            or indent.get(f"{{{W_NS}}}type") != "dxa"
            or int(indent.get(f"{{{W_NS}}}w", "0")) != 120
            or layout is None
            or layout.get(f"{{{W_NS}}}type") != "fixed"
            or sum(grid_widths) != 9360
        ):
            geometry_errors += 1
        rows = table.findall("./w:tr", NS)
        if rows and rows[0].find("./w:trPr/w:tblHeader", NS) is None:
            header_errors += 1
    result.metrics["table_count"] = len(tables)
    result.check("table_geometry", geometry_errors == 0, f"{geometry_errors} 个表格未使用精确 DXA 几何")
    result.check("table_headers", header_errors == 0, f"{header_errors} 个表格未标记首行为标题行")
    return result


def convert_docx_to_pdf(docx_path: Path, output_path: Path) -> Path:
    soffice = shutil.which("libreoffice") or shutil.which("soffice")
    if not soffice:
        raise ExportQualityError("系统未安装 LibreOffice，无法执行渲染质量门禁")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ma-lo-", dir=output_path.parent) as temp:
        temp_path = Path(temp)
        profile = temp_path / "profile"
        home = temp_path / "home"
        profile.mkdir()
        home.mkdir()
        env = os.environ.copy()
        env.update(
            {
                "HOME": str(home),
                "SAL_USE_VCLPLUGIN": "svp",
                "QT_QPA_PLATFORM": "offscreen",
            }
        )
        result = subprocess.run(
            [
                soffice,
                f"-env:UserInstallation=file://{profile}",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(temp_path),
                str(docx_path),
            ],
            capture_output=True,
            text=True,
            timeout=240,
            env=env,
        )
        generated = temp_path / f"{docx_path.stem}.pdf"
        if result.returncode or not generated.exists() or generated.stat().st_size < 1_000:
            detail = (result.stderr or result.stdout or "PDF 转换失败")[-1_500:]
            raise ExportQualityError(f"LibreOffice 转换失败：{detail}")
        shutil.copy2(generated, output_path)
    return output_path


def _pdf_pages(path: Path) -> int:
    pdfinfo = shutil.which("pdfinfo")
    if not pdfinfo:
        raise ExportQualityError("系统缺少 pdfinfo，无法验证 PDF 页数")
    result = subprocess.run([pdfinfo, str(path)], capture_output=True, text=True, timeout=60)
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, re.MULTILINE)
    if result.returncode or not match:
        raise ExportQualityError((result.stderr or "pdfinfo 无法读取 PDF")[-1_000:])
    return int(match.group(1))


def audit_pdf(path: Path, *, expected_max_pages: int, required_text: str) -> QualityResult:
    result = QualityResult("pdf")
    result.check("exists", path.exists() and path.stat().st_size > 1_000, "PDF 文件不存在或为空")
    if not result.passed:
        return result
    try:
        pages = _pdf_pages(path)
    except ExportQualityError as exc:
        result.check("pdf_readable", False, str(exc))
        return result
    result.metrics["pages"] = pages
    result.check("page_count_min", pages >= 2, "PDF 少于 2 页，报告结构不完整")
    result.check(
        "page_count_budget",
        pages <= expected_max_pages,
        f"PDF 共 {pages} 页，超过本次材料的可读性页数预算 {expected_max_pages} 页",
    )

    if re.search(r"[\u4e00-\u9fff]", required_text):
        pdffonts = shutil.which("pdffonts")
        result.check("pdffonts_available", bool(pdffonts), "系统缺少 pdffonts，无法验证中文字体嵌入")
        if pdffonts:
            fonts = subprocess.run([pdffonts, str(path)], capture_output=True, text=True, timeout=60)
            cjk_font = re.search(
                r"(?:Noto|SourceHan|CJK|WenQuanYi|DroidSansFallback|SimSun|YaHei|PingFang|ARPL)",
                fonts.stdout,
                re.IGNORECASE,
            )
            result.check(
                "cjk_font_embedded",
                fonts.returncode == 0 and bool(cjk_font),
                "PDF 未嵌入可识别的中文字体，可能出现方框或缺字",
            )

    pdftotext = shutil.which("pdftotext")
    result.check("pdftotext_available", bool(pdftotext), "系统缺少 pdftotext，无法验证中文文本与内部标记")
    if pdftotext:
        extracted = subprocess.run(
            [pdftotext, str(path), "-"], capture_output=True, text=True, timeout=90
        )
        result.check("text_extract", extracted.returncode == 0, "PDF 文本层无法提取")
        text = extracted.stdout
        required_sample = "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", required_text))[:6]
        result.check(
            "title_text",
            not required_sample or required_sample in "".join(text.split()),
            "PDF 文本层缺少会议标题，可能存在字体或渲染异常",
        )
        result.check("no_internal_tokens", "[证据:" not in text and "[[MAFN" not in text, "PDF 泄露内部证据标记")

    renderer = shutil.which("pdftoppm")
    result.check("renderer_available", bool(renderer), "系统缺少 pdftoppm，无法逐页检查空白页")
    if renderer:
        with tempfile.TemporaryDirectory(prefix="ma-pdf-") as temp:
            prefix = Path(temp) / "page"
            rendered = subprocess.run(
                [renderer, "-gray", "-r", "48", str(path), str(prefix)],
                capture_output=True,
                text=True,
                timeout=180,
            )
            pages_png = sorted(Path(temp).glob("page-*.pgm"))
            result.check(
                "render_all_pages",
                rendered.returncode == 0 and len(pages_png) == pages,
                "PDF 未能完整逐页渲染",
            )
            ink_ratios: list[float] = []
            for image_path in pages_png:
                with Image.open(image_path).convert("L") as image:
                    # Ignore the repeated running header/footer. A page carrying
                    # only those elements is blank from a reader's perspective.
                    body = image.crop(
                        (
                            int(image.width * 0.05),
                            int(image.height * 0.08),
                            int(image.width * 0.95),
                            int(image.height * 0.92),
                        )
                    )
                    histogram = body.histogram()
                    non_white = sum(histogram[:245])
                    ink_ratios.append(round(non_white / max(1, body.width * body.height), 5))
            result.metrics["page_ink_ratios"] = ink_ratios
            blank = [index + 1 for index, ratio in enumerate(ink_ratios) if ratio < 0.001]
            result.check("no_blank_pages", not blank, f"检测到空白页：{blank}")
            if pdftotext and len(ink_ratios) == pages:
                page_text_characters: list[int] = []
                for page_number in range(1, pages + 1):
                    page_text = subprocess.run(
                        [
                            pdftotext,
                            "-f",
                            str(page_number),
                            "-l",
                            str(page_number),
                            str(path),
                            "-",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    normalized = re.sub(r"\s+", "", page_text.stdout) if page_text.returncode == 0 else ""
                    page_text_characters.append(len(normalized))
                result.metrics["page_text_characters"] = page_text_characters
                # Image-led pages legitimately contain little text. A page is
                # only near-blank when both its body ink and extracted copy are
                # genuinely tiny; ordinary chapter transitions and compact
                # tables commonly fall around 7–10% ink and 200–300 characters.
                # The cover is intentionally sparse and is exempt.
                sparse = [
                    index + 1
                    for index, (ratio, characters) in enumerate(zip(ink_ratios, page_text_characters))
                    if index > 0 and ratio < 0.025 and characters < 120
                ]
                result.check(
                    "no_sparse_body_pages",
                    not sparse,
                    f"检测到近空白正文页：{sparse}",
                )
    return result
