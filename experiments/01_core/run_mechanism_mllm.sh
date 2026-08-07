#!/bin/bash
# run_mechanism_mllm.sh
# Run 6 canonical MLLMs on TeacherBehavior (per-image predictions as jsonl for
# the confusion-matrix transitivity analysis).
#
# Strategy (matches the paper's own evidence scale, n=100/class):
#   - --per-class-limit 100: each model only runs images still needed to reach
#     100 successful predictions per class (stratified, deterministic fname
#     order, 25% buffer for timeouts). No wasted work.
#   - Sequential (avoids GPU contention); resumable: state lives in the jsonl
#     (ground truth) + checkpoint cache; interrupted runs simply restart and
#     skip everything already done.
#   - --clean before each model: dedupe jsonl (best-result per image), rebuild
#     checkpoint, exit 0 if the per-class target is already met (skip).
#
# Usage: launch via launch_mechanism.sh (nohup + log + pid file)
set -u

BASE=/home/broadsense/works/lizhuo/AutoResearchClaw/VisualAnchor
DATA=/home/broadsense/works/lizhuo/AutoResearchClaw/datasets_scb
OUT=$BASE/results/01_core/mllm_image_level
LOG=$OUT/run.log
PER_CLASS=100
mkdir -p "$OUT"

cd "$BASE"

MODELS=(
  "qwen3.5:27b"
  "qwen3.6:27b"
  "qwen3.5:35b-a3b"
  "qwen3.6:35b-a3b"
  "gemma4:31b"
  "gemma4:26b"
)

echo "=== $(date) RUN START (per-class=$PER_CLASS) ===" | tee -a "$LOG"

for m in "${MODELS[@]}"; do
  tag="${m//[:.]/_}"
  jsonl="$OUT/predictions_${tag}_TeacherBehavior.jsonl"

  echo "=== $(date) CLEAN $m (dedupe jsonl + status) ===" | tee -a "$LOG"
  python3 experiments/01_core/mllm_image_level_scb5.py \
    --model "$m" --datasets TeacherBehavior \
    --data-root "$DATA" --out "$OUT" \
    --clean --per-class-limit "$PER_CLASS" >> "$LOG" 2>&1
  rc_clean=$?
  if [ "$rc_clean" -eq 0 ]; then
    echo "=== $(date) SKIP $m (per-class target already met) ===" | tee -a "$LOG"
    continue
  fi

  echo "=== $(date) START $m (tag=$tag) ===" | tee -a "$LOG"
  rc=1
  for attempt in 1 2 3 4 5; do
    echo "=== $(date) ATTEMPT $attempt/5 $m ===" | tee -a "$LOG"
    python3 experiments/01_core/mllm_image_level_scb5.py \
      --model "$m" --datasets TeacherBehavior --workers 4 \
      --data-root "$DATA" --out "$OUT" \
      --per-class-limit "$PER_CLASS" >> "$LOG" 2>&1
    rc=$?
    n=$(wc -l < "$jsonl" 2>/dev/null || echo 0)
    echo "=== $(date) EXIT $m attempt=$attempt (exit=$rc, jsonl=$n lines) ===" | tee -a "$LOG"
    [ "$rc" -eq 0 ] && break
    sleep 10
  done
  echo "=== $(date) DONE $m (exit=$rc, jsonl=$n lines) ===" | tee -a "$LOG"
done

echo "=== $(date) ALL DONE ===" | tee -a "$LOG"
