#!/usr/bin/env bash
set -euo pipefail

host="${K3_HOST:?Set K3_HOST to user@host}"
port="${K3_PORT:-22}"
[[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) || { echo 'Invalid K3_PORT' >&2; exit 64; }
root_dir="$(cd "$(dirname "$0")/.." && pwd)"

cd "$root_dir"
npm run build:k3

release_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
candidate="/opt/meeting-assistant.pending-${release_id}"
ssh -p "$port" "$host" "mkdir -m 0755 '$candidate'"
rsync -az \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  backend deploy scripts pyproject.toml .env.example \
  -e "ssh -p $port" "$host:$candidate/"
ssh -p "$port" "$host" "bash '$candidate/deploy/activate-release.sh' '$candidate'"
