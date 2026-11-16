# 部署、模型恢复与测试

本页保留项目的开发与运维说明。项目定位和当前限制见 [README](../README.md)，发布验收状态见[检查清单](RELEASE_CHECKLIST.md)。以下部署流程仍依赖厂商原生运行时，不是面向全新系统的一键安装承诺。

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

脚本构建前端、上传独立候选目录、保留旧代码、重启并鉴权检查。首次接入 SSD 时先按[检查清单](RELEASE_CHECKLIST.md)核对挂载和备份，不自动格式化磁盘。

手动部署时，先在开发机执行前端构建，再把 `backend/`（包含构建后的静态资源）、`deploy/`、`scripts/`、`pyproject.toml` 和 `.env.example` 同步到 `/opt/meeting-assistant`。板上运行：

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
