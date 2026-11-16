"use client";

import WaveSurfer from "wavesurfer.js";
import { createElement, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { chapterEvidence, markdownBlocks, reportSavePayload, type Evidence } from "./report-document";

type Meeting = {
  id: string;
  title: string;
  mode: "LOCAL_ONLY" | "HYBRID_REVIEW";
  report_detail: ReportDetail;
  status: string;
  glossary: string;
  created_at: string;
  asset_count?: number;
  fact_count?: number;
};

type ReportDetail = "BRIEF" | "STANDARD" | "DETAILED";

type Asset = {
  id: string;
  kind: string;
  filename: string;
  content_type?: string;
  size_bytes: number;
  metadata?: Record<string, unknown>;
};

type Segment = {
  id: string;
  asset_id?: string;
  start_ms: number;
  end_ms: number;
  raw_text: string;
  corrected_text?: string;
  user_text?: string;
  speaker_label?: string;
  speaker_name?: string;
  confidence: number;
  needs_review: boolean;
};

type Fact = {
  id: string;
  kind: string;
  statement: string;
  owner?: string;
  due_at?: string;
  confidence: number;
  evidence_ids: string[];
  user_locked: boolean;
  review_state?: "accepted" | "rejected" | "pending";
};

type Report = {
  id: string;
  section: string;
  content_markdown: string;
  content_html: string;
  version: number;
  evidence_ids?: string[];
  editorial_state?: string;
  needs_review?: boolean;
};

type Job = {
  id: string;
  status: string;
  current_stage: string;
  progress: number;
  error?: string;
};

type Detail = {
  meeting: Meeting;
  assets: Asset[];
  segments: Segment[];
  facts: Fact[];
  evidence?: Evidence[];
  media_links: Array<{ id: string; asset_id: string; filename: string; fact_statement?: string; score: number; confirmed: boolean; rejected: boolean }>;
  reports: Report[];
  latest_version?: number;
  jobs: Job[];
  cloud_audit: Array<{ id: string; operation: string; payload_summary: string; provider: string; status: string; created_at: string }>;
  export_audits: Array<{ id: string; format: "docx" | "pdf"; status: "PASSED" | "FAILED"; metrics?: { pages?: number }; errors?: string[]; created_at: string }>;
};

const sections: Record<string, string> = {
  summary: "会议摘要",
  article: "技术整理文稿",
  topics: "议题与背景",
  decisions: "关键决定",
  actions: "行动项",
  risks: "风险与异议",
  questions: "未决问题",
  media: "关联图片",
  review: "待核实项",
};

const stageLabels: Record<string, string> = {
  INGEST: "材料校验",
  AUDIO_PREP: "音频预处理",
  DIARIZE: "说话人分离",
  ASR: "语音识别",
  ALIGN: "时间对齐",
  IMAGE_PREP: "图片增强",
  OCR_VLM: "图片理解",
  FACT_EXTRACT: "事实抽取",
  LINK: "跨模态关联",
  REPORT: "生成报告",
  VERIFY: "证据审计",
  DONE: "处理完成",
};

const kindLabels: Record<string, string> = {
  topic: "议题",
  decision: "决定",
  action: "行动",
  risk: "风险",
  question: "问题",
  metric: "指标",
  term: "术语",
};

const reportDetails: Record<ReportDetail, { label: string; description: string }> = {
  BRIEF: { label: "精简纪要", description: "结论、行动项与少量关键材料" },
  STANDARD: { label: "标准报告", description: "议题脉络、证据边界与图片索引" },
  DETAILED: { label: "技术整理文稿", description: "知识化章节、解释正文、表格、图片与证据附录" },
};

function formatTime(ms: number) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(total / 60).toString().padStart(2, "0");
  const seconds = (total % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function apiToken() {
  return typeof window === "undefined" ? "" : window.sessionStorage.getItem("meeting-token") || "";
}

async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  const token = apiToken();
  if (token) headers.set("X-Meeting-Token", token);
  if (options.body && !(options.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(payload.detail || "请求失败");
  }
  return response.json();
}

function Waveform({ asset, seek, onSeek }: { asset?: Asset; seek?: { assetId: string; startMs: number; request: number }; onSeek?: (seconds: number) => void }) {
  const container = useRef<HTMLDivElement>(null);
  const instance = useRef<WaveSurfer | null>(null);
  const [audioUrl, setAudioUrl] = useState<string>();
  const requestedSeek = useRef(seek);
  useEffect(() => {
    requestedSeek.current = seek;
    if (seek && seek.assetId === asset?.id && instance.current?.getDuration()) instance.current.setTime(seek.startMs / 1000);
  }, [seek, asset?.id]);

  useEffect(() => {
    if (!asset || !container.current) return;
    let active = true;
    let objectUrl = "";
    const headers = new Headers();
    if (apiToken()) headers.set("X-Meeting-Token", apiToken());
    fetch(`/api/assets/${asset.id}`, { headers })
      .then((response) => {
        if (!response.ok) throw new Error("音频加载失败");
        return response.blob();
      })
      .then((blob) => {
        if (!active || !container.current) return;
        objectUrl = URL.createObjectURL(blob);
        setAudioUrl(objectUrl);
        instance.current?.destroy();
        const wave = WaveSurfer.create({
          container: container.current,
          url: objectUrl,
          height: 56,
          waveColor: "#a8b8ca",
          progressColor: "#146b71",
          cursorColor: "#e47145",
          barWidth: 2,
          barGap: 2,
          barRadius: 2,
          normalize: true,
        });
        wave.on("interaction", (time) => onSeek?.(time));
        wave.on("ready", () => {
          const target = requestedSeek.current;
          if (target?.assetId === asset.id) wave.setTime(target.startMs / 1000);
        });
        container.current.onclick = () => wave.playPause();
        instance.current = wave;
      })
      .catch(() => undefined);
    return () => {
      active = false;
      instance.current?.destroy();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [asset, onSeek]);

  if (!asset) return <div className="wave-empty">上传音频后将在这里显示波形</div>;
  return <><div className="waveform" ref={container} aria-label="会议音频波形，点击播放或暂停" />{audioUrl && <audio className="native-audio sr-only" preload="metadata" src={audioUrl} />}</>;
}

function AuthenticatedImage({ asset }: { asset: Asset }) {
  const [source, setSource] = useState<string>();
  useEffect(() => {
    let active = true;
    let objectUrl = "";
    const headers = new Headers();
    if (apiToken()) headers.set("X-Meeting-Token", apiToken());
    fetch(`/api/assets/${asset.id}`, { headers })
      .then((response) => {
        if (!response.ok) throw new Error("图片加载失败");
        return response.blob();
      })
      .then((blob) => {
        if (!active) return;
        objectUrl = URL.createObjectURL(blob);
        setSource(objectUrl);
      })
      .catch(() => undefined);
    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [asset.id]);
  return source ? <img src={source} alt={asset.filename} /> : <div className="image-loading">正在读取原图…</div>;
}

function MarkdownPreview({ text, onEvidence }: { text: string; onEvidence: (id: string) => void }) {
  function inline(value: string) {
    return value.split(/(\[证据:[0-9a-fA-F]{8,32}\]|\*\*[^*]+\*\*|`[^`]+`)/g).map((part, index) => {
      const reference = /^\[证据:([0-9a-fA-F]{8,32})\]$/.exec(part);
      if (reference) return <button key={index} className="citation-link" onClick={() => onEvidence(reference[1])}>来源 {reference[1].slice(0, 8)}</button>;
      if (part.startsWith("**") && part.endsWith("**")) return <strong key={index}>{part.slice(2, -2)}</strong>;
      if (part.startsWith("`") && part.endsWith("`")) return <code key={index}>{part.slice(1, -1)}</code>;
      return part;
    });
  }
  return <div className="report-prose" aria-label="报告阅读预览">{markdownBlocks(text).map((block, index) => {
    if (block.kind === "heading") return createElement(`h${block.level}`, { key: index }, inline(block.text));
    if (block.kind === "table") return <div className="preview-table" key={index}><table><thead><tr>{block.headers.map((cell, position) => <th key={position}>{inline(cell)}</th>)}</tr></thead><tbody>{block.rows.map((row, position) => <tr key={position}>{row.map((cell, column) => <td key={column}>{inline(cell)}</td>)}</tr>)}</tbody></table></div>;
    if (block.kind === "list") return createElement(block.ordered ? "ol" : "ul", { key: index }, block.items.map((item, position) => <li key={position}>{inline(item)}</li>));
    if (block.kind === "quote") return <blockquote key={index}>{inline(block.text)}</blockquote>;
    if (block.kind === "code") return <pre key={index}><code>{block.text}</code></pre>;
    return <p key={index}>{inline(block.text)}</p>;
  })}</div>;
}

function ReportEditor({ report, onSave, onEvidence, onDirty }: { report?: Report; onSave: (id: string, text: string) => Promise<void>; onEvidence: (id: string) => void; onDirty: (dirty: boolean) => void }) {
  const [draft, setDraft] = useState(report?.content_markdown || "");
  const [saved, setSaved] = useState(report?.content_markdown || "");
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState("");
  const dirty = draft !== saved;
  useEffect(() => {
    onDirty(dirty);
    const warn = (event: BeforeUnloadEvent) => { if (dirty) { event.preventDefault(); event.returnValue = ""; } };
    window.addEventListener("beforeunload", warn);
    return () => { window.removeEventListener("beforeunload", warn); onDirty(false); };
  }, [dirty, onDirty]);
  async function save() {
    if (!report) return;
    const snapshot = draft;
    setSaving(true); setSaveError("");
    try { await onSave(report.id, snapshot); setSaved(snapshot); }
    catch (error) { setSaveError((error as Error).message); }
    finally { setSaving(false); }
  }
  return (
    <div className="editor-shell">
      <div className="editor-toolbar">
        <button className={!editing ? "active" : ""} onClick={() => setEditing(false)}>阅读预览</button>
        <button className={editing ? "active" : ""} disabled={!report} onClick={() => setEditing(true)}>编辑正文</button>
        <span role="status">{dirty ? "有未保存修改" : "已保存"}</span>
        <button className="save-report" disabled={!report || saving || !dirty} onClick={save}>{saving ? "正在保存…" : "保存本章"}</button>
      </div>
      {saveError && <p role="alert" className="editor-notice">保存失败：{saveError}。修改仍保留在编辑框中。</p>}
      {editing ? <><p className="editor-notice">标题使用 #，表格使用 |。请保留对应句后的 [证据:…] 来源标记；保存保留原始结构与引用。</p><textarea className="markdown-editor" aria-label="报告 Markdown 正文" value={draft} spellCheck={false} onChange={(event) => setDraft(event.target.value)} /></> : <MarkdownPreview text={draft || "报告尚未生成。"} onEvidence={onEvidence} />}
    </div>
  );
}

export default function MeetingApp() {
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [selectedId, setSelectedId] = useState<string>();
  const [detail, setDetail] = useState<Detail>();
  const [screen, setScreen] = useState<"upload" | "workspace" | "report">("upload");
  const [title, setTitle] = useState("");
  const [mode, setMode] = useState<Meeting["mode"]>("LOCAL_ONLY");
  const [reportDetail, setReportDetail] = useState<ReportDetail>("STANDARD");
  const [glossary, setGlossary] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [job, setJob] = useState<Job>();
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [activeSection, setActiveSection] = useState("summary");
  const [activeFactKind, setActiveFactKind] = useState("all");
  const [editorDirty, setEditorDirty] = useState(false);
  const [selectedEvidence, setSelectedEvidence] = useState<Evidence>();
  const [evidenceSeek, setEvidenceSeek] = useState<{ assetId: string; startMs: number; request: number }>();
  const [tokenOpen, setTokenOpen] = useState(false);
  const [tokenDraft, setTokenDraft] = useState("");
  const [health, setHealth] = useState<{ status: string; mem_available_mb?: number; asr_configured?: boolean; diarization_configured?: boolean; ocr_configured?: boolean; vlm_configured?: boolean; cloud_configured?: boolean }>();
  const fileInput = useRef<HTMLInputElement>(null);

  const refreshMeetings = useCallback(async () => {
    try {
      const data = await api<Meeting[]>("/api/meetings");
      setMeetings(data);
      setError("");
    } catch (err) {
      setError((err as Error).message);
    }
  }, []);

  const loadDetail = useCallback(async (id: string) => {
    const data = await api<Detail>(`/api/meetings/${id}`);
    setDetail(data);
    setReportDetail(data.meeting.report_detail || "STANDARD");
    const latestJob = data.jobs[0];
    if (latestJob && ["QUEUED", "RUNNING"].includes(latestJob.status)) setJob(latestJob);
  }, []);

  useEffect(() => {
    let active = true;
    api<Meeting[]>("/api/meetings")
      .then((data) => { if (active) setMeetings(data); })
      .catch((err) => { if (active) setError(err.message); });
    api<NonNullable<typeof health>>("/api/health")
      .then((data) => { if (active) setHealth(data); })
      .catch(() => undefined);
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    let active = true;
    api<Detail>(`/api/meetings/${selectedId}`)
      .then((data) => {
        if (!active) return;
        setDetail(data);
        setReportDetail(data.meeting.report_detail || "STANDARD");
        const latestJob = data.jobs[0];
        if (latestJob && ["QUEUED", "RUNNING"].includes(latestJob.status)) setJob(latestJob);
      })
      .catch((err) => { if (active) setError(err.message); });
    return () => { active = false; };
  }, [selectedId]);

  const jobId = job?.id;
  const jobStatus = job?.status;
  useEffect(() => {
    if (!jobId || !jobStatus || !["QUEUED", "RUNNING"].includes(jobStatus)) return;
    const timer = window.setInterval(async () => {
      try {
        const current = await api<Job>(`/api/jobs/${jobId}`);
        setJob(current);
        if (["DONE", "FAILED"].includes(current.status)) {
          window.clearInterval(timer);
          if (selectedId) await loadDetail(selectedId);
          await refreshMeetings();
          setScreen(current.status === "DONE" ? "workspace" : "upload");
          setMessage(current.status === "DONE" ? "会议材料处理完成，可以开始校对。" : "");
          if (current.error) setError(current.error);
        }
      } catch (err) {
        setError((err as Error).message);
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [jobId, jobStatus, selectedId, loadDetail, refreshMeetings]);

  const audio = detail?.assets.find((asset) => asset.kind === "audio");
  const activeReport = detail?.reports.find((report) => report.section === activeSection);
  const currentEvidence = chapterEvidence(activeReport, detail?.evidence || []);
  const evidenceAsset = detail?.assets.find((asset) => asset.id === selectedEvidence?.asset_id);
  const evidenceSegment = detail?.segments.find((segment) => segment.id === selectedEvidence?.segment_id);
  const filteredFacts = useMemo(
    () => detail?.facts.filter((fact) => activeFactKind === "all" || fact.kind === activeFactKind) || [],
    [detail?.facts, activeFactKind],
  );

  async function createAndUpload() {
    if (!title.trim()) return setError("请填写会议名称");
    if (!files.length) return setError("请至少选择一个音频、图片或文字转写文件");
    setBusy(true);
    setError("");
    try {
      const meeting = await api<Meeting>("/api/meetings", {
        method: "POST",
        body: JSON.stringify({ title: title.trim(), mode, report_detail: reportDetail, glossary }),
      });
      const form = new FormData();
      files.forEach((file) => form.append("files", file));
      await api(`/api/meetings/${meeting.id}/assets`, { method: "POST", body: form });
      const started = await api<Job>(`/api/meetings/${meeting.id}/process`, { method: "POST" });
      setSelectedId(meeting.id);
      setJob(started);
      setMessage("材料已上传，K3 正在按阶段处理。");
      await refreshMeetings();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function updateSegment(segment: Segment, patch: Record<string, unknown>) {
    await api(`/api/transcript-segments/${segment.id}`, { method: "PATCH", body: JSON.stringify(patch) });
    if (selectedId) await loadDetail(selectedId);
  }

  async function updateFact(fact: Fact, statement: string, review_state: "accepted" | "rejected" | "pending" = "pending") {
    try {
      await api(`/api/facts/${fact.id}`, {
        method: "PATCH",
        body: JSON.stringify({ statement, user_locked: true, review_state }),
      });
      if (selectedId) await loadDetail(selectedId);
      setMessage(review_state === "accepted" ? "事实已人工采用；相关报告仍需重新核验。" : review_state === "rejected" ? "事实已拒绝采用。" : "修改已保存，事实等待重新确认。");
    } catch (error) { setError((error as Error).message); }
  }

  async function saveReport(id: string, text: string) {
    await api(`/api/reports/${id}`, {
      method: "PATCH",
      body: JSON.stringify(reportSavePayload(text)),
    });
    if (selectedId) await loadDetail(selectedId);
    setMessage("章节结构与证据引用已保存。人工编辑不等于已完成内容核验。");
  }

  function openEvidence(reference: string) {
    const match = chapterEvidence({ content_markdown: "", evidence_ids: [reference] }, detail?.evidence || [])[0];
    if (!match?.evidence) { setError(match?.issue || "找不到该证据"); return; }
    const evidence = match.evidence;
    const segment = detail?.segments.find((item) => item.id === evidence.segment_id);
    setSelectedEvidence(evidence);
    if (evidence.asset_id) setEvidenceSeek({ assetId: evidence.asset_id, startMs: evidence.locator?.start_ms ?? segment?.start_ms ?? 0, request: Date.now() });
  }

  function leaveEditor(action: () => void) {
    if (editorDirty && !window.confirm("本章还有未保存修改。确定放弃修改并离开吗？")) return;
    setEditorDirty(false);
    action();
  }

  async function updateMediaLink(id: string, confirmed: boolean, rejected: boolean) {
    await api(`/api/media-links/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ confirmed, rejected }),
    });
    if (selectedId) await loadDetail(selectedId);
  }

  async function updateReportDetail(value: ReportDetail) {
    setReportDetail(value);
    if (value === "DETAILED") setActiveSection("article");
    if (!detail) return;
    try {
      await api(`/api/meetings/${detail.meeting.id}`, {
        method: "PATCH",
        body: JSON.stringify({ report_detail: value }),
      });
      await loadDetail(detail.meeting.id);
      setMessage(`报告深度已切换为“${reportDetails[value].label}”。`);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function regenerate() {
    if (!selectedId) return;
    setBusy(true);
    try {
      await api(`/api/meetings/${selectedId}/reports/${activeSection}/regenerate`, { method: "POST" });
      await loadDetail(selectedId);
      setMessage(`${sections[activeSection]}已生成新版本。`);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function downloadReport(format: "docx" | "pdf") {
    if (!detail) return;
    setError("");
    try {
      const headers = new Headers();
      if (apiToken()) headers.set("X-Meeting-Token", apiToken());
      const response = await fetch(`/api/meetings/${detail.meeting.id}/export?format=${format}&detail=${reportDetail}`, { headers });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({ detail: "导出失败" }));
        throw new Error(payload.detail || "导出失败");
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${detail.meeting.title}-${reportDetails[reportDetail].label}.${format}`;
      anchor.click();
      URL.revokeObjectURL(url);
      await loadDetail(detail.meeting.id);
      setMessage(`${reportDetails[reportDetail].label} ${format.toUpperCase()} 已开始下载。导出检查不代表技术内容已通过人工语义核验。`);
    } catch (err) {
      setError((err as Error).message);
    }
  }

  function selectMeeting(id: string) {
    leaveEditor(() => { setSelectedId(id); setScreen("workspace"); setError(""); setSelectedEvidence(undefined); });
  }

  function newMeeting() {
    leaveEditor(() => {
      setSelectedId(undefined); setDetail(undefined); setFiles([]); setTitle("");
      setGlossary(""); setReportDetail("STANDARD"); setScreen("upload"); setSelectedEvidence(undefined);
    });
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand" onClick={newMeeting} role="button" tabIndex={0}>
          <span className="brand-mark">K3</span>
          <div><strong>会议智记</strong><small>Evidence-first Meeting Assistant</small></div>
        </div>
        <nav className="main-nav" aria-label="主要页面">
          <button className={screen === "upload" ? "active" : ""} onClick={() => leaveEditor(() => setScreen("upload"))}>材料</button>
          <button className={screen === "workspace" ? "active" : ""} disabled={!detail} onClick={() => leaveEditor(() => setScreen("workspace"))}>校对工作台</button>
          <button className={screen === "report" ? "active" : ""} disabled={!detail?.reports.length} onClick={() => setScreen("report")}>会议报告</button>
        </nav>
        <div className="system-state">
          <span className={`status-dot ${health?.status === "ok" ? "online" : ""}`} />
          <div><strong>K3 本地节点</strong><small>{health?.status === "ok" ? (health.mem_available_mb ? `可用内存 ${health.mem_available_mb} MB` : "本地服务正常") : "正在连接"}</small></div>
          <button className="icon-button" aria-label="配置访问令牌" onClick={() => { setTokenDraft(apiToken()); setTokenOpen(true); }}>•••</button>
        </div>
      </header>

      <aside className="sidebar">
        <button className="new-meeting" onClick={newMeeting}><span>＋</span> 新建会议</button>
        <div className="sidebar-label">最近会议</div>
        <div className="meeting-list">
          {meetings.map((meeting) => (
            <button key={meeting.id} className={`meeting-item ${selectedId === meeting.id ? "selected" : ""}`} onClick={() => selectMeeting(meeting.id)}>
              <span className="meeting-icon">会</span>
              <span><strong>{meeting.title}</strong><small>{new Date(meeting.created_at).toLocaleDateString("zh-CN")} · {meeting.asset_count || 0} 个材料</small></span>
              <i className={`meeting-status ${meeting.status.toLowerCase()}`} />
            </button>
          ))}
          {!meetings.length && <p className="empty-list">尚无会议记录</p>}
        </div>
        <div className="capability-card">
          <div className="sidebar-label">能力状态</div>
          <div><span className={health?.asr_configured ? "ok" : "warn"} />语音识别 <b>{health?.asr_configured ? "就绪" : "待配置"}</b></div>
          <div><span className={health?.diarization_configured ? "ok" : "warn"} />说话人分离 <b>{health?.diarization_configured ? "就绪" : "待配置"}</b></div>
          <div><span className={health?.ocr_configured ? "ok" : "warn"} />图片 OCR <b>{health?.ocr_configured ? "就绪" : "待配置"}</b></div>
          <div><span className={health?.vlm_configured ? "ok" : "warn"} />图片理解 <b>{health?.vlm_configured ? "就绪" : "待配置"}</b></div>
          <div><span className="warn" />云端候选复核 <b>{health?.cloud_configured ? "待完善" : "待配置"}</b></div>
        </div>
      </aside>

      <main className="content">
        {(message || error) && (
          <div className={`toast ${error ? "error" : "success"}`}>
            <span>{error || message}</span><button onClick={() => { setError(""); setMessage(""); }}>×</button>
          </div>
        )}
        {job && ["QUEUED", "RUNNING"].includes(job.status) && (
          <div className="job-banner">
            <div className="job-spinner" />
            <div><strong>{stageLabels[job.current_stage] || job.current_stage}</strong><small>K3 正在逐阶段处理，模型不会并发占用内存。</small></div>
            <div className="job-progress"><span style={{ width: `${Math.round(job.progress * 100)}%` }} /></div>
            <b>{Math.round(job.progress * 100)}%</b>
          </div>
        )}

        {screen === "upload" && (
          <section className="upload-screen">
            <div className="page-heading"><div><span className="eyebrow">NEW MEETING</span><h1>把会议材料变成可追溯的结论</h1><p>音频、PPT 截图和术语表将按阶段在 K3 上处理。任何决定、行动项和数字都保留证据引用。</p></div></div>
            <div className="upload-grid">
              <div className="form-card">
                <label>会议名称<input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="例如：K3 会议助手原型评审" /></label>
                <div className="field-label">处理模式</div>
                <div className="mode-switch">
                  <button className={mode === "LOCAL_ONLY" ? "selected" : ""} onClick={() => setMode("LOCAL_ONLY")}><strong>纯本地</strong><small>材料不离开 K3</small></button>
                  <button className={mode === "HYBRID_REVIEW" ? "selected" : ""} onClick={() => setMode("HYBRID_REVIEW")}><strong>端云复核</strong><small>云端候选复核待配置 / 待完善</small></button>
                </div>
                <div className="field-label">报告深度</div>
                <div className="detail-switch">
                  {(Object.entries(reportDetails) as Array<[ReportDetail, (typeof reportDetails)[ReportDetail]]>).map(([key, option]) => (
                    <button key={key} className={reportDetail === key ? "selected" : ""} onClick={() => setReportDetail(key)}>
                      <strong>{option.label}</strong><small>{option.description}</small>
                    </button>
                  ))}
                </div>
                <label>术语表与参会人<textarea value={glossary} onChange={(event) => setGlossary(event.target.value)} placeholder="每行一个：SpacemiT、RISC-V、张工、项目代号…" /></label>
              </div>
              <div className="drop-card" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); setFiles((current) => [...current, ...Array.from(event.dataTransfer.files)]); }}>
                <input ref={fileInput} type="file" multiple accept="audio/*,image/*,.txt,.md,.srt,.vtt" hidden onChange={(event) => setFiles(Array.from(event.target.files || []))} />
                <button className="drop-zone" onClick={() => fileInput.current?.click()}>
                  <span className="upload-glyph">↑</span><strong>拖入音频、图片或文字转写</strong><small>支持 MP3 / WAV / M4A / PNG / JPG / TXT，单场会议不超过 60 分钟</small><em>选择文件</em>
                </button>
                <div className="file-list">
                  {files.map((file, index) => <div key={`${file.name}-${index}`}><span>{file.type.startsWith("image") ? "图" : file.type.startsWith("audio") ? "音" : "文"}</span><p><strong>{file.name}</strong><small>{formatBytes(file.size)}</small></p><button onClick={() => setFiles((current) => current.filter((_, item) => item !== index))}>×</button></div>)}
                </div>
                <button className="primary-action" disabled={busy || !files.length || !title.trim()} onClick={createAndUpload}>{busy ? "正在上传…" : "上传并开始处理"}<span>→</span></button>
              </div>
            </div>
          </section>
        )}

        {screen === "workspace" && detail && (
          <section className="workspace-screen">
            <div className="page-heading compact"><div><span className="eyebrow">REVIEW WORKSPACE</span><h1>{detail.meeting.title}</h1><p>{detail.segments.length} 条转写 · {detail.facts.length} 条事实 · 报告版本 {detail.latest_version || 0}</p></div><button className="outline-action" onClick={() => setScreen("report")} disabled={!detail.reports.length}>查看报告 →</button></div>
            <div className="audio-card"><div className="audio-meta"><span className="audio-play">▶</span><div><strong>{audio?.filename || "尚未上传音频"}</strong><small>点击波形播放 / 暂停</small></div></div><Waveform asset={audio} /></div>
            <div className="workspace-grid">
              <div className="transcript-panel panel">
                <div className="panel-header"><div><h2>逐句转写</h2><span>{detail.segments.filter((item) => item.needs_review).length} 条待核实</span></div></div>
                <div className="segment-list">
                  {detail.segments.map((segment) => (
                    <article key={segment.id} className={segment.needs_review ? "needs-review" : ""}>
                      <time>{formatTime(segment.start_ms)}</time>
                      <div className="segment-body"><div className="speaker-row"><input defaultValue={segment.speaker_name || segment.speaker_label || "Speaker 1"} onBlur={(event) => updateSegment(segment, { speaker_name: event.target.value })} /><span className={`confidence ${segment.confidence < 0.55 ? "low" : ""}`}>{Math.round(segment.confidence * 100)}%</span></div><textarea defaultValue={segment.user_text || segment.corrected_text || segment.raw_text} onBlur={(event) => { if (event.target.value !== (segment.user_text || segment.corrected_text || segment.raw_text)) updateSegment(segment, { user_text: event.target.value, needs_review: false }); }} /></div>
                    </article>
                  ))}
                  {!detail.segments.length && <div className="panel-empty">处理完成后，带时间戳的转写会出现在这里。</div>}
                </div>
              </div>
              <div className="facts-panel panel">
                <div className="panel-header"><div><h2>会议事实库</h2><span>修改后自动锁定</span></div></div>
                <div className="fact-filters"><button className={activeFactKind === "all" ? "active" : ""} onClick={() => setActiveFactKind("all")}>全部</button>{["decision", "action", "risk", "question"].map((kind) => <button key={kind} className={activeFactKind === kind ? "active" : ""} onClick={() => setActiveFactKind(kind)}>{kindLabels[kind]}</button>)}</div>
                <div className="fact-list">
                  {filteredFacts.map((fact) => <article key={`${fact.id}:${fact.statement}`} className={`fact-card ${fact.kind}`}>
                    <div><span>{kindLabels[fact.kind] || fact.kind}</span>{fact.user_locked && <i>已锁定修改</i>}<b>{Math.round(fact.confidence * 100)}%</b></div>
                    <textarea aria-label={`${kindLabels[fact.kind] || fact.kind}事实正文`} defaultValue={fact.statement} onBlur={(event) => {
                      if ((event.relatedTarget as HTMLElement | null)?.dataset.reviewAction) return;
                      if (event.target.value !== fact.statement) void updateFact(fact, event.target.value);
                    }} />
                    {(fact.owner || fact.due_at) && <small>{fact.owner && `负责人：${fact.owner}`}{fact.owner && fact.due_at && " · "}{fact.due_at && `截止：${fact.due_at}`}</small>}
                    <footer><span>{fact.review_state === "accepted" ? "人工已采用" : fact.review_state === "rejected" ? "已拒绝采用" : "待人工确认"}</span><div className="fact-evidence-links">{fact.evidence_ids.map((id) => <button key={id} onClick={() => openEvidence(id)}>查看证据 {id.slice(0, 8)}</button>)}{!fact.evidence_ids.length && "无证据 · 待核实"}</div></footer>
                    <div className="fact-review-actions">{([['accepted', '核对后采用'], ['rejected', '拒绝采用'], ['pending', '恢复待核实']] as const).map(([state, label]) => <button key={state} data-review-action="true" disabled={state === "accepted" && !fact.evidence_ids.length} onClick={(event) => {
                      const statement = event.currentTarget.closest("article")?.querySelector("textarea")?.value ?? fact.statement;
                      void updateFact(fact, statement, state);
                    }}>{label}</button>)}</div>
                  </article>)}
                  {!filteredFacts.length && <div className="panel-empty">尚无此类事实。处理完成后可在这里校对并锁定。</div>}
                </div>
              </div>
            </div>
          </section>
        )}

        {screen === "report" && detail && (
          <section className="report-screen">
            <div className="page-heading compact"><div><span className="eyebrow">EVIDENCE REPORT · V{detail.latest_version || 0}</span><h1>{detail.meeting.title}</h1><p>请逐项核对原始材料。草稿、人工编辑与内容审定是不同状态。</p>{detail.export_audits?.[0] && <small className={`export-quality ${detail.export_audits[0].status.toLowerCase()}`}>最近导出：{detail.export_audits[0].status === "PASSED" ? "文件检查通过，内容仍需核验" : `未通过（${detail.export_audits[0].errors?.[0] || "请检查材料"}）`}{detail.export_audits[0].metrics?.pages ? ` · ${detail.export_audits[0].metrics.pages} 页` : ""}</small>}</div><div className="report-export-panel"><div className="report-detail-picker" role="group" aria-label="报告深度">{(Object.entries(reportDetails) as Array<[ReportDetail, (typeof reportDetails)[ReportDetail]]>).map(([key, option]) => <button key={key} disabled={editorDirty} className={reportDetail === key ? "selected" : ""} onClick={() => updateReportDetail(key)}><strong>{option.label}</strong><small>{option.description}</small></button>)}</div><div className="export-actions"><button disabled={editorDirty} onClick={() => downloadReport("docx")}>导出 Word</button><button disabled={editorDirty} onClick={() => downloadReport("pdf")}>导出 PDF</button></div></div></div>
            <div className="report-layout">
              <aside className="report-toc">{Object.entries(sections).map(([key, label]) => <button key={key} className={activeSection === key ? "active" : ""} onClick={() => leaveEditor(() => setActiveSection(key))}><span>{label}</span><small>{detail.reports.find((item) => item.section === key)?.evidence_ids?.length || 0}</small></button>)}</aside>
              <div className="report-main"><div className="report-main-header"><div><span>{sections[activeSection]}</span><small>{activeReport?.editorial_state === "user_edited" || activeReport?.editorial_state === "human_edited" ? "人工修订稿" : "自动生成草稿"} · {activeReport?.needs_review === false ? "请核对内容后使用" : "需要内容核验"}</small></div><button disabled={busy || editorDirty} onClick={regenerate}>↻ 仅重生成本章</button></div><ReportEditor key={activeReport?.id || activeSection} report={activeReport} onSave={saveReport} onEvidence={openEvidence} onDirty={setEditorDirty} />{activeSection === "media" && <div className="report-media-grid">{detail.assets.filter((asset) => asset.kind === "image").map((asset) => { const link = detail.media_links.find((item) => item.asset_id === asset.id); return <article key={asset.id} className={link?.rejected ? "rejected" : link?.confirmed ? "confirmed" : "pending"}><AuthenticatedImage asset={asset} /><div><strong>{asset.filename}</strong><p>{link?.fact_statement || "尚未建立文字事实关联"}</p><small>{link ? `${link.confirmed ? "已确认" : link.rejected ? "已取消" : "待确认"} · 关联分 ${Math.round(link.score * 100)}%` : "待关联"}</small>{link && <span><button onClick={() => updateMediaLink(link.id, true, false)}>确认关联</button><button onClick={() => updateMediaLink(link.id, false, true)}>取消关联</button></span>}</div></article>; })}</div>}</div>
              <aside className="evidence-rail"><h3>本章证据</h3>{currentEvidence.map(({ reference, evidence, issue }) => <button key={reference} disabled={!evidence} onClick={() => openEvidence(reference)}><time>{evidence?.kind === "audio" ? formatTime(evidence.locator?.start_ms || 0) : evidence?.kind === "image" ? "原始图片" : "文字材料"} · {reference.slice(0, 8)}</time><p>{issue || evidence?.text}</p></button>)}{!currentEvidence.length && <p>本章尚无证据引用。</p>}<div className="evidence-note">仅显示本章引用的来源。点击查看原图、原文或定位对应录音；引用存在不代表内容已核实。</div></aside>
            </div>
          </section>
        )}
      </main>

      {selectedEvidence && <div className="modal-backdrop"><section className="evidence-modal" role="dialog" aria-modal="true" aria-label="原始证据核对"><header><h2>原始证据核对</h2><button autoFocus onClick={() => setSelectedEvidence(undefined)}>关闭</button></header><p>{evidenceAsset?.filename || "文字材料"} · 来源 {selectedEvidence.id.slice(0, 8)}</p>{evidenceAsset?.kind === "image" && <AuthenticatedImage asset={evidenceAsset} />}{evidenceAsset?.kind === "audio" && <><p>定位：{formatTime(selectedEvidence.locator?.start_ms ?? evidenceSegment?.start_ms ?? 0)} — {formatTime(selectedEvidence.locator?.end_ms ?? evidenceSegment?.end_ms ?? 0)}</p><Waveform asset={evidenceAsset} seek={evidenceSeek} /></>}<h3>保存的来源文字</h3><pre>{selectedEvidence.text}</pre>{evidenceSegment && <><h3>原始识别与修订</h3><p>原始：{evidenceSegment.raw_text}</p><p>修订：{evidenceSegment.user_text ?? evidenceSegment.corrected_text ?? "暂无修订"}</p></>}<p className="evidence-note">识别文字可能有误。请对照原图或录音核对主体、数字、单位和适用条件，再决定是否采用。</p></section></div>}
      {tokenOpen && <div className="modal-backdrop"><div className="token-modal"><h2>访问令牌</h2><p>仅在 K3 服务启用了共享访问令牌时填写。令牌只保存在当前浏览器会话。</p><input type="password" value={tokenDraft} onChange={(event) => setTokenDraft(event.target.value)} placeholder="X-Meeting-Token" /><div><button onClick={() => setTokenOpen(false)}>取消</button><button onClick={() => leaveEditor(() => { window.sessionStorage.setItem("meeting-token", tokenDraft); setTokenOpen(false); window.location.reload(); })}>保存并重新连接</button></div></div></div>}
    </div>
  );
}
