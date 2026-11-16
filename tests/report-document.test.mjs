import assert from "node:assert/strict";
import test from "node:test";
import { chapterEvidence, citationIds, markdownBlocks, reportSavePayload } from "../web/report-document.ts";

const markdown = "# 技术报告\r\n\r\n## 带宽\r\n\r\n| 项目 | 数值 |\r\n| --- | --- |\r\n| **K3** | 8 GB [证据:aaaaaaaa11111111] |\r\n\r\n- 仅适用于本样本。 [证据:bbbbbbbb]\r\n";

test("editing one sentence preserves headings, table, exact citations, Unicode and line endings in saved payload", () => {
  const edited = markdown.replace("仅适用于本样本。", "该结论仅适用于本次近讲样本。");
  const payload = JSON.parse(JSON.stringify(reportSavePayload(edited)));
  assert.equal(payload.content_markdown, edited);
  assert.deepEqual(citationIds(payload.content_markdown), ["aaaaaaaa11111111", "bbbbbbbb"]);
  assert.ok(payload.content_markdown.includes("| **K3** | 8 GB [证据:aaaaaaaa11111111] |\r\n"));
  const blocks = markdownBlocks(payload.content_markdown);
  assert.deepEqual(blocks.filter((block) => block.kind === "heading").map((block) => block.level), [1, 2]);
  assert.deepEqual(blocks.find((block) => block.kind === "table").rows, [["**K3**", "8 GB [证据:aaaaaaaa11111111]"]]);
  assert.deepEqual(blocks.find((block) => block.kind === "list").items, ["该结论仅适用于本次近讲样本。 [证据:bbbbbbbb]"]);
});

test("chapter evidence resolves unique short IDs, deduplicates and excludes unrelated early transcript segments", () => {
  const evidence = [
    { id: "cccccccc33333333", kind: "audio", text: "不属于本章的开场白", asset_id: "audio-other" },
    { id: "aaaaaaaa11111111", kind: "image", text: "K3 8 GB", asset_id: "image-2" },
    { id: "bbbbbbbb22222222", kind: "audio", text: "带宽测试", asset_id: "audio-2", segment_id: "segment-91", locator: { start_ms: 123000, end_ms: 130000 } },
  ];
  const selected = chapterEvidence({ content_markdown: markdown, evidence_ids: ["aaaaaaaa11111111", "bbbbbbbb22222222"] }, evidence);
  assert.deepEqual(selected.map((row) => row.evidence.id), ["aaaaaaaa11111111", "bbbbbbbb22222222"]);
  assert.equal(selected[1].evidence.asset_id, "audio-2");
  assert.equal(selected[1].evidence.locator.start_ms, 123000);
  assert.equal(selected[1].evidence.segment_id, "segment-91");
});

test("ambiguous and missing references stay visible instead of navigating to a random source", () => {
  const evidence = [{ id: "aaaaaaaa11111111", kind: "audio", text: "A" }, { id: "aaaaaaaa22222222", kind: "image", text: "B" }];
  const selected = chapterEvidence({ content_markdown: "内容 [证据:aaaaaaaa] [证据:dddddddd]" }, evidence);
  assert.deepEqual(selected.map((row) => row.issue), ["引用不唯一", "证据不存在"]);
  assert.ok(selected.every((row) => !row.evidence));
  assert.deepEqual(chapterEvidence(undefined, evidence), []);
});

test("preview preserves literal hostile markup as text and handles ordered lists/code without converting persisted source", () => {
  const source = "## 核对\n<script>alert(1)</script>\n1. 第一项\n2. 第二项\n\n```\n<img src=x onerror=alert(1)>\n```";
  const blocks = markdownBlocks(source);
  assert.deepEqual(blocks[1], { kind: "paragraph", text: "<script>alert(1)</script>" });
  assert.deepEqual(blocks[2], { kind: "list", ordered: true, items: ["第一项", "第二项"] });
  assert.equal(blocks[3].kind, "code");
  assert.equal(reportSavePayload(source).content_markdown, source);
});
