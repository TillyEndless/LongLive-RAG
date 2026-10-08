#!/usr/bin/env bash
set -u
ROOT=/data/zxl/LongLive-RAG-group11_15_h200
SESSION=group12_15_canonical10_rr
RUNNER=$ROOT/scripts/run_group12_15_canonical10_rr.py
STATE=$ROOT/results/group12_15_persistent_campaign/canonical10_rr/runner_state.json
LOG=$ROOT/results/group12_15_persistent_campaign/canonical10_rr/watchdog.log
mkdir -p "$(dirname "$LOG")"
while true; do
  if [ -f "$STATE" ] && grep -q '"stage": "COMPLETE"' "$STATE"; then
    echo "$(date -Is) COMPLETE" >> "$LOG"
    exit 0
  fi
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "$(date -Is) runner missing; restarting" >> "$LOG"
    tmux new-session -d -s "$SESSION" "cd $ROOT && exec python3 -u $RUNNER"
  fi
  sleep 900
done
