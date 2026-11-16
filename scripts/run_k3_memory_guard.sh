#!/usr/bin/env bash

# Run one heavyweight inference process while enforcing the K3 product memory
# limits. The guarded command is terminated if either threshold is crossed.

set -uo pipefail

max_rss_mb="${MAX_RSS_MB:-6349}"
min_available_mb="${MIN_AVAILABLE_MB:-512}"
monitor_interval="${MONITOR_INTERVAL_SECONDS:-0.2}"
metrics_path="${METRICS_PATH:-}"

if [[ "${1:-}" != "--" || $# -lt 2 ]]; then
    echo "usage: METRICS_PATH=metrics.json $0 -- command [args...]" >&2
    exit 64
fi
if [[ ! -r /proc/meminfo ]]; then
    echo "error: the memory guard requires Linux /proc" >&2
    exit 69
fi
shift

started_ns="$(date +%s%N)"
peak_rss_kb=0
lowest_available_kb=9223372036854775807
guard_triggered=false
child_pid=""

terminate_child() {
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM -- "-$child_pid" 2>/dev/null || kill -TERM "$child_pid" 2>/dev/null || true
        sleep 1
        kill -KILL -- "-$child_pid" 2>/dev/null || kill -KILL "$child_pid" 2>/dev/null || true
    fi
}

trap 'terminate_child; exit 130' HUP INT TERM

if command -v setsid >/dev/null 2>&1; then
    setsid "$@" &
else
    "$@" &
fi
child_pid=$!

while kill -0 "$child_pid" 2>/dev/null; do
    rss_kb="$(awk '/^VmRSS:/ { print $2; exit }' "/proc/$child_pid/status" 2>/dev/null || true)"
    available_kb="$(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo)"
    rss_kb="${rss_kb:-0}"

    if (( rss_kb > peak_rss_kb )); then
        peak_rss_kb=$rss_kb
    fi
    if (( available_kb < lowest_available_kb )); then
        lowest_available_kb=$available_kb
    fi

    if (( rss_kb > max_rss_mb * 1024 || available_kb < min_available_mb * 1024 )); then
        guard_triggered=true
        terminate_child
        break
    fi
    sleep "$monitor_interval"
done

wait "$child_pid"
exit_code=$?
trap - HUP INT TERM
finished_ns="$(date +%s%N)"
elapsed_ms=$(( (finished_ns - started_ns) / 1000000 ))

if [[ -n "$metrics_path" ]]; then
    printf '{"exit_code":%d,"elapsed_ms":%d,"peak_rss_mb":%d,"min_available_mb":%d,"guard_triggered":%s}\n' \
        "$exit_code" "$elapsed_ms" "$((peak_rss_kb / 1024))" \
        "$((lowest_available_kb / 1024))" "$guard_triggered" > "$metrics_path"
fi

exit "$exit_code"
