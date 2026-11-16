#!/usr/bin/env bash
# Called only after a complete upload; preserves previous code for rollback.
set -euo pipefail
candidate="${1:?release directory required}"
case "$candidate" in
  /opt/meeting-assistant.pending-*) ;;
  *) echo 'Invalid release directory' >&2; exit 64 ;;
esac
[[ -f "$candidate/deploy/install-k3.sh" && -f "$candidate/backend/meeting_assistant/main.py" ]]
[[ ! -L "$candidate" && ! -L /opt/meeting-assistant ]]
release_id="${candidate##*.pending-}"
backup="/opt/meeting-assistant.previous-${release_id}"
failed="/opt/meeting-assistant.failed-${release_id}"
[[ ! -e "$backup" && ! -e "$failed" ]]
had_previous=false
was_active=false
systemctl is-active --quiet meeting-assistant.service && was_active=true
systemctl stop meeting-assistant.service || {
  if [[ "$was_active" == true ]]; then exit 1; fi
}
if [[ -d /opt/meeting-assistant ]]; then
  mv /opt/meeting-assistant "$backup"
  had_previous=true
fi
rollback() {
  status=$?
  trap - ERR
  systemctl stop meeting-assistant.service || true
  if [[ -d /opt/meeting-assistant ]]; then mv /opt/meeting-assistant "$failed"; fi
  if [[ "$had_previous" == true ]]; then
    mv "$backup" /opt/meeting-assistant
    install -m 0644 /opt/meeting-assistant/deploy/meeting-assistant.service /etc/systemd/system/meeting-assistant.service
    systemctl daemon-reload
    if [[ "$was_active" == true ]]; then
      systemctl restart meeting-assistant.service
      /usr/bin/python3 "$failed/deploy/check-service.py" --health
    fi
  fi
  echo "Release failed; previous code restored. Failed release: $failed" >&2
  exit "$status"
}
trap rollback ERR
mv "$candidate" /opt/meeting-assistant
# Board-native binaries are built locally and deliberately excluded from the
# public source archive. Preserve them across a source-only application update.
if [[ "$had_previous" == true && -d "$backup/vendor" && ! -e /opt/meeting-assistant/vendor ]]; then
  cp -a "$backup/vendor" /opt/meeting-assistant/vendor
fi
bash /opt/meeting-assistant/deploy/install-k3.sh
trap - ERR
echo "Release active: ${release_id}; previous code retained at ${backup}"
