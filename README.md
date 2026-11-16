# K3 会议智记

运行在进迭时空 K3 Pico-ITX 上的单用户、会后异步会议辅助工具。系统把音频转写、图片理解、事实抽取和报告生成拆成可恢复阶段，并要求决定、行动项、数字和日期引用原始证据。

面向产品评审、方案介绍和成果展示的说明，请阅读 [《K3 会议智记：产品与技术介绍》](docs/PROJECT_OVERVIEW_ZH.md)。

> 当前状态：源码发布候选，尚未完成产品级验收。新板与 SSD 部署仍待验证；真实会议准确率、性能和全自动报告语义质量的限制见文末。发布步骤见 [发布检查清单](docs/RELEASE_CHECKLIST.md)，安全边界见 [SECURITY.md](SECURITY.md)。

源码采用 [Apache-2.0](LICENSE) 许可证。模型权重、厂商运行时与第三方依赖分别遵守其原有许可；私有会议材料不随源码发布。

## 已实现能力

- `LOCAL_ONLY` 与 `HYBRID_REVIEW` 两种会议模式。
- 音频、图片、TXT/Markdown/SRT/VTT 上传，SHA-256 校验与文件元数据存储。
- SQLite 任务状态机：`INGEST → AUDIO_PREP → DIARIZE → ASR → ALIGN → IMAGE_PREP → OCR_VLM → FACT_EXTRACT → LINK → REPORT → VERIFY`。
- FFmpeg 16kHz 单声道标准化，以及确定性的图片自动对比度和锐化。
- 可配置 ASR、CAM++、OCR、VLM 命令适配器；缺少模型时明确降级，不生成虚假识别结果。
- K3 `llama-server` 模型生命周期管理、内存安全阈值和 OpenAI 兼容云端复核适配器。
- 原始转写、自动校正、用户修订三层数据；说话人批量改名。
- 飞书 TXT 导出稿解析，保留说话人、分钟/小时级精确时间轴和长说话轮次。
- 证据、结构化事实、图片关联、用户锁定事实和分章节版本化报告。
- 三档报告深度：精简纪要、标准报告、技术整理文稿。技术整理文稿先生成全材料目录，再按章节独立生成
  解释性正文、要点和可比较表格，不使用滚动摘要拼接长文。
- 技术正文导出使用读者友好的 `S01` 来源编号，并在附录映射到 PPT 原图或音频时间段；浏览器采用
  Markdown 源文编辑与阅读预览，保存时保留原始标题、表格及证据标记。
- 富材料会议的详细导出设有硬门禁：正文不少于 3000 字符、至少 5 个技术章节和 5 个小节，并覆盖
  至少 90% 的会议图片；不达标时拒绝交付而不是降级为摘要。
- 服务重启后从中断阶段恢复；ASR 占位、图片证据与人工确认关联可重复处理而不累积。
- React 校对工作台、WaveSurfer 波形、Markdown 报告编辑、Word/PDF 导出；DOCX 中的证据引用为
  OOXML 原生脚注，图片使用跨 Word/LibreOffice 更稳定的内联布局。
- 开发环境可选访问令牌；生产 systemd 服务强制要求令牌，支持日志轮换。

## 本地开发

```bash
npm install
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'

# 终端 1：API
MEETING_DATA_DIR=.data MEETING_STATIC_DIR=backend/static \
  .venv/bin/uvicorn meeting_assistant.main:app --app-dir backend --reload

# 终端 2：React 开发服务器
npm run dev:k3
```

浏览器访问 `http://localhost:4173`。生产静态资源由以下命令生成：

```bash
npm run build:k3
```

## 模型适配器

模型命令通过环境变量配置。命令必须把 JSON 打到标准输出，或写入 `{output}` 指定文件。

```bash
ASR_COMMAND='/opt/models/run_asr --input {input} --output {output}'
DIARIZATION_COMMAND='/opt/models/run_diarization --input {input} --output {output}'
OCR_COMMAND='/opt/models/run_ocr --input {input} --output {output}'
VLM_COMMAND='/opt/models/run_vlm --input {input} --ocr {ocr} --output {output}'
```

ASR JSON 至少应包含：

```json
{
  "segments": [
    {"start": 0.0, "end": 4.2, "text": "会议开始。", "speaker": "Speaker 1", "confidence": 0.92}
  ]
}
```

本地文本模型使用 `LOCAL_LLM_MODEL_PATH` 指向 GGUF。系统会按需启动监听 `127.0.0.1:8081` 的 `llama-server`，完成事实抽取后立即卸载模型。

云端适配层使用 `.env.example` 中的 `CLOUD_*` 配置，兼容 `/chat/completions` 和 `/audio/transcriptions` 接口。整场音频/整图的自动兜底外发已关闭；`HYBRID_REVIEW` 不代表已经完成按疑难片段的自动外发工作流。云端结果只能作为候选，不得覆盖本地或用户结果。

## K3 部署

模型不包含在源码仓库。可下载到已挂载的 SSD（示例 `/data`，先确认挂载和容量）：

```bash
python3 scripts/restore_models.py --root /data/meeting-assistant/models
python3 scripts/restore_vendor_models.py --root /data/meeting-assistant/models
```

第一个脚本固定 Qwen3-4B 官方仓库版本并验证文件哈希；官方站点不可达时可显式添加
`--endpoint https://hf-mirror.com`。第二个脚本恢复进迭时空 ASR q40、VLM、OCR、CAM++ 和 Silero VAD。
官方提供 MD5 的文件按固定值核对，并记录本地 SHA-256；未提供上游哈希的模型只记录本地摘要，
不宣称完成独立来源认证。下载完成不等于通过推理验收，须核对配置路径并执行模型基准。

已有板端原生运行时的恢复部署，可执行：

```bash
python3 scripts/adapt_campplus.py \
  /data/meeting-assistant/models/speaker/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx \
  /data/meeting-assistant/models/speaker/campplus-native-compatible.onnx
sudo python3 deploy/configure-ssd-models.py --root /data/meeting-assistant/models
```

CAM++ 适配只把输入名从 `x` 改为 `feature`，保留原文件并验证三组数值输出等价。
配置脚本备份原配置、保留访问令牌与会议数据路径，新增 SSD 挂载依赖；检查完成后再
`sudo systemctl daemon-reload && sudo systemctl restart meeting-assistant`。
这些脚本不安装厂商原生运行时：`spacemit_vad`、`spacemit_vision` 和
`vendor/onnxruntime/build/bin/extract_speaker_embedding` 仍需按厂商文档构建/安装，不能把干净系统当成已就绪。
如果升级系统到 OpenCV 5 后旧 OCR 扩展缺少 `libopencv_dnn.so.414`，可将官方 4.14 deb
单独解包到项目运行时目录，并通过 `configure-ssd-models.py --opencv-lib <兼容库目录>` 指定；
不要以错误的 so 文件软链接冒充 ABI 兼容，也不要因此降级整机的 OpenCV。

支持自定义 SSH 端口（使用 SSH agent 或交互认证，不把密码写入配置仓库）：

```bash
K3_HOST=root@your-host K3_PORT=60022 bash scripts/deploy-k3.sh
```

脚本构建前端、上传独立候选目录、保留旧代码、重启并鉴权检查。首次接入 SSD 时先按[检查清单](docs/RELEASE_CHECKLIST.md)核对挂载和备份，不自动格式化磁盘。

先在开发机执行前端构建，再把 `backend/`、`deploy/`、`pyproject.toml` 和 `.env.example` 同步到 `/opt/meeting-assistant`。板上运行：

```bash
chmod +x /opt/meeting-assistant/deploy/install-k3.sh
/opt/meeting-assistant/deploy/install-k3.sh
```

安装脚本会生成 `/etc/meeting-assistant.env`、创建 `/var/lib/meeting-assistant`、启用 systemd 服务并输出一次 Web 访问令牌。服务地址为 `http://<K3-IP>:8000`。

模型基准：

```bash
python3 scripts/benchmark_model.py /path/to/model.gguf --output benchmark.json

# 进入应用集成前，连续验证 20 次真实结构化请求，同时记录加载时间和峰值内存
python3 scripts/benchmark_model.py /path/to/model.gguf \
  --server-samples 20 --output benchmark-full.json
```

`--server-samples` 会在独立的 `127.0.0.1:18081` 临时启动模型，统计 JSON 合法率、
事实结构合法率、证据/类型召回、虚构证据 ID、数字一致性、请求延迟、加载时间、模型进程树峰值 RSS
和整机最低可用内存，完成后自动卸载。可通过 `--prompts-jsonl cases.jsonl` 使用真实脱敏样本；
每行包含 `system`、`prompt`，可选 `expected`（证据 ID 到允许事实类型数组的映射）字段。

## 测试

```bash
.venv/bin/pytest
npm test
npm run lint
npm run build
```

测试覆盖完整降级流水线、证据引用、技术文稿 HTML 清理、用户锁定事实、API 上传/处理/编辑、访问令牌和 DOCX 导出；
DOCX 测试还会检查读者引用、内部标记清除、精确表格几何和图片媒体包。真实 ASR/VLM 的准确率与性能验收
需要在 K3 上配置相应模型后使用业务数据集执行。

## 真实素材验证记录

- 91 分 08 秒飞书中文会议稿：解析为 117 个说话轮次、8 位说话人和 19 个事实窗口；Qwen3-0.6B 全流程约 6 分钟。
- 52 分 24.6 秒中英混合 AAC：K3 成功转换为 16kHz 单声道 PCM；Silero VAD 得到 49 个语音区间、146 个 ASR 切片。
- 31 张会场 PPT 照片：原图与确定性增强图均已保存；重复运行保持 31 条图片证据，不再累积副本。
- 在 `IMAGE_PREP` 阶段强制重启 systemd 服务后，任务从该阶段自动恢复并完成。
- 未配置 ASR/OCR/VLM 时，报告只生成“待核实项”，不会把占位提示提升为正式会议事实。
- Qwen3-0.6B Q4_0 严格 Schema 连续 20 次均生成合法结构，但合成事实类型召回为 50%；
  在真实会议稿的 4 个明确事实样本上仅 2/4 完整通过。
- Qwen3-4B Q4_K_M 严格 Schema 连续 20 次均生成合法结构，合成事实类型召回为 87.5%；
  在同一批真实会议稿样本上 4/4 通过。加载约 19 秒，峰值 RSS 约 5.8GiB，作为事实与报告默认模型。
- Qwen3-ASR-0.6B q40 完整转写 2945.6 秒有效语音耗时 431.5 秒（RTF 0.143），146 个切片中
  144 个获得文本，另外 2 个是低概率短 VAD 误触发；峰值 RSS 3.54GiB。1.7B 的 q4km 与 q40
  虽能通过官方短样本，但在真实中英混合片段上出现空文本和重复解码，均未进入默认链路。
- VibeVoice-ASR 社区 Q4_K 在匹配的 CrispASR 0.8.19 上完成 K3 原生构建并通过 5/5 个同源真实片段：
  能正确识别 `agent`、`flow`、`RISC-V`，且术语上下文可把 `HFS100/HFS200` 修正为
  `HAPS-100/HAPS-200`。但 81.5 秒音频在 4 线程下耗时 882 秒（RTF 10.82），8 个 X100 核心
  处理 15.1 秒困难片段仍需 98.8 秒（RTF 6.53）；峰值 RSS 5.16GiB。上下文还曾诱发无证据术语，
  因而不替换 0.6B 主链，只能作为少量疑难片段的候选复核，并必须保留原结果和人工确认。
- CAM++ 对 52 分钟会议的 1988 个全局声纹窗口完成聚类，耗时约 818 秒（RTF 0.260），峰值 RSS
  268MiB；在官方双人 RTTM 样本上 DER 为 2.74%。真实会议自动得到 6 个簇，因缺少人工 RTTM，
  该场只能验证性能与全局一致性，不能冒充真实 DER 验收集。
- PP-OCR 原图关键行召回为 36.84%，确定性增强图降至 31.58%，因此原图固定作为文字证据输入。
  FastVLM 严格 JSON 率为 35%，Qwen3.5-2B 配合 OCR 与重试达到 95%；2B 输出仍经过数字、图表和
  不可读区域的确定性门禁。Qwen3.5-4B 的 K3 Q4_1 包加载峰值为 6392–6444MiB，超过 6.2GiB
  产品上限，即使整机未 OOM 也不作为可部署模型。

- 同一场 52 分钟会议在最终质量配置下，音频预处理、全局说话人聚类、ASR、31 张图片 OCR/VLM
  和 4B 事实抽取的无重复阶段时间合计约 64 分钟，超过 40 分钟目标；未通过速度验收，未通过增加
  Swap、减少图片或更换低质量模型掩盖。
- 自动详细文稿曾生成 8 章、31 小节、16,552 字符并覆盖 31/31 张图片，但人工语义审查发现术语误译、
  句子截断和图片含义误判，因此没有把“结构通过”冒充“可发布”。当前正式交付采用用户提供并人工编辑锁定的
  技术文稿作为正文，模型结果只保留为候选材料。
- 最终确定性审计覆盖 144 条非降级转写、175 条证据、31/31 张图片、9/9 个报告章节和 9 条正式事实；
  正式事实证据覆盖率 100%，无启发式正式事实、未确认图片正式事实或媒体关系冲突。技术正文有
  11 处数字未出现在 OCR/VLM 文本层，均保留原图引用并标记为发布前人工对图复核，不冒充自动语义验证。
- 正式技术文稿为 10 章、32 小节、22,243 字符，正文引用覆盖 31/31 张原图；来源索引只显示清晰的
  `S 编号 → 图号 → 正文小节`，低质量 OCR 原文保留在证据库，不进入人类交付文档。
- 板端 DOCX/PDF 最终实测导出为 26 页 A4，包含 6 张与对应正文小节相邻的重点原图、31 张完整缩略图索引、正式事实附录和
  可回溯来源索引。板端 LibreOffice PDF 经全 26 页逐页检查，未发现中文缺字、裁切、重叠、孤立标题
  或图片/说明分离；PDF 字体均已嵌入。

## 尚未通过的正式验收项

- 现有素材不足 10 场带人工真值、每场不超过 60 分钟的会议，无法诚实给出总体 CER、真实会议 DER、
  实体/决定/行动项召回率。
- 当前 K3 软件栈没有可直接部署的 Qwen3-ForcedAligner-0.6B ONNX/GGUF 包，应用仍使用 VAD/ASR
  句段时间戳；强制对齐模型验收未通过。
- 52 分钟真实会议的高质量本地处理约 64 分钟，未达到 40 分钟目标。
- K3 本地 4B 模型的无人值守详细正文尚未通过人工语义验收；当前达到产品交付标准的是“模型生成候选 +
  人工校订锁定 + 确定性证据/排版门禁”的人机协作路径，不宣称全自动发布已验收。

照片存在透视、会场背景和观众遮挡。接入 OCR/VLM 时应对整图与屏幕候选裁剪分别推理并按置信度择优，生成式增强不得作为文字证据。
