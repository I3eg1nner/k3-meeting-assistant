export type Evidence = {
  id: string;
  kind: string;
  text: string;
  asset_id?: string | null;
  segment_id?: string | null;
  locator?: { start_ms?: number; end_ms?: number; regions?: unknown[] };
};

export type ReportSource = { content_markdown: string; evidence_ids?: string[] };

/** Markdown is the persisted document, including its citations. Never round-trip via rendered text. */
export function reportSavePayload(markdown: string) {
  return { content_markdown: markdown };
}

export function citationIds(markdown: string): string[] {
  return [...new Set([...markdown.matchAll(/\[证据:([0-9a-fA-F]{8,32})\]/g)].map((match) => match[1]))];
}

export function chapterEvidence(report: ReportSource | undefined, evidence: Evidence[]) {
  if (!report) return [];
  const ids = [...new Set([...citationIds(report.content_markdown), ...(report.evidence_ids || [])])];
  const seen = new Set<string>();
  return ids.flatMap((reference) => {
    const exact = evidence.find((item) => item.id === reference);
    const matches = exact ? [exact] : evidence.filter((item) => item.id.startsWith(reference));
    const item = matches.length === 1 ? matches[0] : undefined;
    const key = item?.id || reference;
    if (seen.has(key)) return [];
    seen.add(key);
    return [{ reference, evidence: item, issue: item ? undefined : matches.length ? "引用不唯一" : "证据不存在" }];
  });
}

export type MarkdownBlock =
  | { kind: "heading"; level: number; text: string }
  | { kind: "paragraph" | "quote" | "code"; text: string }
  | { kind: "list"; ordered: boolean; items: string[] }
  | { kind: "table"; headers: string[]; rows: string[][] };

/** Safe preview model: values are rendered as React text, never injected as HTML. */
export function markdownBlocks(markdown: string): MarkdownBlock[] {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const blocks: MarkdownBlock[] = [];
  const cells = (line: string) => line.trim().replace(/^\||\|$/g, "").split(/(?<!\\)\|/).map((cell) => cell.trim().replace(/\\\|/g, "|"));
  for (let index = 0; index < lines.length;) {
    const line = lines[index];
    if (!line.trim()) { index++; continue; }
    if (line.trimStart().startsWith("```")) {
      const text: string[] = [];
      index++;
      while (index < lines.length && !lines[index].trimStart().startsWith("```")) text.push(lines[index++]);
      if (index < lines.length) index++;
      blocks.push({ kind: "code", text: text.join("\n") });
      continue;
    }
    const heading = /^(#{1,6})\s+(.+)$/.exec(line);
    if (heading) { blocks.push({ kind: "heading", level: heading[1].length, text: heading[2] }); index++; continue; }
    if (line.includes("|") && index + 1 < lines.length && cells(lines[index + 1]).every((cell) => /^:?-{3,}:?$/.test(cell))) {
      const headers = cells(line);
      const rows: string[][] = [];
      index += 2;
      while (index < lines.length && lines[index].trim() && lines[index].includes("|")) rows.push(cells(lines[index++]));
      blocks.push({ kind: "table", headers, rows });
      continue;
    }
    const list = /^\s*(?:([-*+])|\d+[.)])\s+(.+)$/.exec(line);
    if (list) {
      const ordered = !list[1];
      const items: string[] = [];
      while (index < lines.length) {
        const item = /^\s*(?:([-*+])|\d+[.)])\s+(.+)$/.exec(lines[index]);
        if (!item || !item[1] !== ordered) break;
        items.push(item[2]); index++;
      }
      blocks.push({ kind: "list", ordered, items });
      continue;
    }
    blocks.push(line.startsWith("> ") ? { kind: "quote", text: line.slice(2) } : { kind: "paragraph", text: line });
    index++;
  }
  return blocks;
}
