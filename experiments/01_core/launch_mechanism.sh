#!/bin/bash
# launch_mechanism.sh — nohup launcher for run_mechanism_mllm.sh
#   - runs in background, immune to SSH disconnect
#   - logs: results/01_core/mllm_image_level/run.log (append)
#   - pid file: results/01_core/mllm_image_level/run.pid (for monitoring/kill)
#   - resumable: re-run this script after an interruption; already-done images
#     are skipped via jsonl+checkpoint (no wasted work)
set -u

BASE=/home/broadsense/works/lizhuo/AutoResearchClaw/VisualAnchor
OUT=$BASE/results/01_core/mllm_image_level
LOG=$OUT/run.log
PIDF=$OUT/run.pid
mkdir -p "$OUT"

if [ -f "$PIDF" ]; then
  old=$(cat "$PIDF")
  if kill -0 "$old" 2>/dev/null; then
    echo "runner already running (pid $old). kill it first if you want to restart:"
    echo "  kill -TERM -$old ; pkill -f mllm_image_level_scb5"
    exit 1
  fi
  rm -f "$PIDF"
fi

cd "$BASE"
nohup bash experiments/01_core/run_mechanism_mllm.sh >/dev/null 2>&1 &
pid=$!
echo $pid > "$PIDF"
echo "launched runner pid=$pid (log=$LOG, pidfile=$PIDF)"
