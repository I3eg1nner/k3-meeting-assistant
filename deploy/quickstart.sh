#!/usr/bin/env bash
# Application installer. Never upgrades vendor libraries or downloads models.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
check=false
yes=false
deps=false
for arg in "$@"; do
  case "$arg" in
    --check) check=true ;;
    --yes) yes=true ;;
    --install-deps) deps=true ;;
    --help|-h)
      echo '用法: sudo bash install.sh [--yes] [--install-deps] | bash install.sh --check'
      echo '保留已有配置、模型和会议数据；--install-deps 仅补装缺少的应用依赖。'
      echo '不安装或升级厂商模型运行时，不下载模型，不修改磁盘或 Swap。'
      exit 0 ;;
    *) echo "未知参数: $arg" >&2; exit 64 ;;
  esac
done
command -v python3 >/dev/null || { echo '需要系统 Python 3.11+' >&2; exit 1; }
if [[ "$check" == true ]]; then
  exec python3 "$root/deploy/preflight.py" --root "$root"
fi
[[ "$(id -u)" == 0 ]] || { echo '请使用 sudo 或 root 运行；--check 无需 root。' >&2; exit 1; }
[[ -f "$root/backend/static/index.html" ]] || {
  echo '缺少编译后的前端，请使用 .run 安装包，或先在开发机执行 npm run build:k3。' >&2
  exit 1
}
python3 "$root/deploy/preflight.py" --root "$root" --platform-only
# Prevent simultaneous installations, including two self-extracting packages.
exec 9>/run/lock/meeting-assistant-install.lock
flock -n 9 || { echo '另一个安装任务正在运行。' >&2; exit 1; }
python3 "$root/deploy/preflight.py" --root "$root" --idle-only
if [[ "$yes" != true ]]; then
  [[ -t 0 ]] || { echo '非交互安装请添加 --yes。' >&2; exit 1; }
  echo '将更新 /opt/meeting-assistant 并短暂重启服务；保留配置、模型及会议数据。'
  read -r -p '继续安装？[y/N] ' answer
  [[ "$answer" == y || "$answer" == Y ]] || exit 0
fi
if [[ "$deps" == true ]]; then
  missing=()
  for package in python3-fastapi python3-uvicorn python3-multipart python3-docx python3-pil ffmpeg libreoffice-writer poppler-utils fonts-noto-cjk; do
    if [[ "$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true)" != 'install ok installed' ]]; then
      missing+=("$package")
    fi
  done
  if (( ${#missing[@]} )); then
    apt-get update
    apt-get install -y --no-upgrade "${missing[@]}"
  fi
fi
python3 "$root/deploy/preflight.py" --root "$root"
candidate="$(mktemp -d /opt/meeting-assistant.pending-XXXXXXXX)"
chmod 0755 "$candidate"
cleanup() { [[ ! -d "$candidate" ]] || rm -rf -- "$candidate"; }
trap cleanup EXIT
for item in backend deploy scripts install.sh pyproject.toml .env.example; do
  cp -a "$root/$item" "$candidate/"
done
python3 "$root/deploy/preflight.py" --root "$root" --idle-only
MEETING_SKIP_SYSTEM_PACKAGES=1 bash "$candidate/deploy/activate-release.sh" "$candidate"
echo
echo '应用安装及鉴权检查通过。浏览器访问 http://<K3局域网IP>:8000'
echo '访问令牌保存在 /etc/meeting-assistant.env；安装不会重置已有令牌。'
echo '请仅通过可信局域网或 SSH 隧道访问，不要直接暴露公网。'
echo '模型可用性请查看下面诊断；此检查不等于模型质量验收。'
python3 /opt/meeting-assistant/deploy/preflight.py --root /opt/meeting-assistant --models-only
