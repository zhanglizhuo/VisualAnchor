"""
realized_routing_aggregate.py

Aggregates the realized-routing shard outputs (run on V100) into
results/05_applications/hybrid/realized_routing.json.

Key facts:
- Realized pipeline accuracy (TeacherBehavior, tau=45, Qwen3.5-27B via
  Ollama, category-name prompt, bbox crops) vs the expected-value estimate.
- v1 (index-reply protocol) failed on this backend (teacher->'0' collapse,
  out-of-range '8' replies); v2 (category-name, as validated by
  prompt_optimization_v3.py on the same backend) is the reported protocol.
- Aggregation MUST filter by the "arm" field: 156 files appear in both arms
  (calib is a stratified sample of the full val split, overlapping routed).
"""
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from datetime import date

PROJ = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJ / "results"
MANIFEST = Path("/tmp/opencode/realized_routing_manifest.json")
SHARDS = Path("/tmp/opencode")  # shard_{0..3}.jsonl pulled from V100
OUT = RESULTS / "05_applications" / "hybrid" / "realized_routing.json"

manifest = json.load(open(MANIFEST))
CLASSES = manifest["classes"]
mllm = json.load(open(RESULTS / "01_core" / "paper_data" / "mllm_raw.json"))
models = [k for k in mllm["TeacherBehavior"].keys() if not k.startswith("_")]
RAW_KEYS = ["guide", "answer", "On-stage interaction", "blackboard-writing",
            "teacher", "stand", "screen", "blackBoard"]
canon_mean = {c: float(np.mean([mllm["TeacherBehavior"][m][k] for m in models]))
              for c, k in zip(CLASSES, RAW_KEYS)}
canon_q35 = {c: mllm["TeacherBehavior"]["Qwen3.5-27B"][k] for c, k in zip(CLASSES, RAW_KEYS)}

recs = []
for sid in range(4):
    with open(SHARDS / f"shard_{sid}.jsonl") as f:
        recs += [json.loads(ln) for ln in f]
assert len(recs) == 2072

routed = [r for r in recs if r["arm"] == "routed"]
calib = [r for r in recs if r["arm"] == "calib"]

pred = json.load(open(RESULTS / "01_core" / "clip_per_image" / "per_image_predictions.json"))
anchor = json.load(open(RESULTS / "01_core" / "anchor_score_scb5" / "anchor_scores.json"))
tb_anchor = {k: v["acc"] for k, v in anchor["TeacherBehavior"]["per_class_acc"].items()}
items = pred["TeacherBehavior"]["predictions"]
N = len(items)

n_kept = sum(1 for it in items if tb_anchor[it["pred_class_name"]] >= 45)
kept_correct = sum(1 for it in items if tb_anchor[it["pred_class_name"]] >= 45 and it["correct"])
mllm_correct = sum(1 for r in routed if r["correct"] is True)
realized = (kept_correct + mllm_correct) / N * 100

def per_class_breakdown(arm_recs, canon_ref):
    by_cls = defaultdict(lambda: [0, 0])
    for r in arm_recs:
        by_cls[r["true"]][1] += 1
        if r["correct"]:
            by_cls[r["true"]][0] += 1
    out = {}
    for c in sorted(by_cls):
        n_ok, n_t = by_cls[c]
        out[CLASSES[c]] = {
            "realized": round(100 * n_ok / n_t, 2),
            "canonical": round(canon_ref[CLASSES[c]], 2),
            "diff_pp": round(100 * n_ok / n_t - canon_ref[CLASSES[c]], 2),
            "n": n_t,
        }
    return out

routed_bd = per_class_breakdown(routed, canon_q35)
calib_bd = per_class_breakdown(calib, canon_q35)
calib_shifts_q35 = [v["diff_pp"] for v in calib_bd.values()]
calib_shifts_mean = [100 * 0 + v["realized"] - canon_mean[k] for k, v in calib_bd.items()]

out = {
    "description": (
        "R1 realized routing validation, TeacherBehavior tau=45. The MLLM branch "
        "(1,832 routed images) was actually executed: Qwen3.5-27B served via Ollama "
        "(q4 quantization), category-name classification prompt (as validated by "
        "prompt_optimization_v3.py on this backend), first-person-bbox crops with "
        "5% margin. v1 used the companion pipeline's index-reply prompt and failed "
        "on this backend (teacher->'0' collapse, out-of-range '8' replies); v2 is "
        "the reported protocol. calib = 30/class stratified full-val sample for "
        "backend-shift quantification (156 files overlap the routed arm; aggregate "
        "by arm field). kept branch uses cached per-image CLIP predictions."
    ),
    "computation_date": str(date.today()),
    "model": "Qwen3.5-27B (Ollama q4)",
    "protocol": "bbox crop (first person box + 5% margin), category-name reply",
    "realized_pipeline_acc": round(realized, 2),
    "expected_q35": 55.85,
    "expected_6model_mean": 55.66,
    "diff_vs_expected_q35_pp": round(realized - 55.85, 2),
    "diff_vs_expected_mean_pp": round(realized - 55.66, 2),
    "clip_branch": {"n": n_kept, "correct": kept_correct, "acc": round(100 * kept_correct / n_kept, 2)},
    "mllm_branch": {"n": len(routed), "correct": mllm_correct, "acc": round(100 * mllm_correct / len(routed), 2)},
    "calib_mean_shift_vs_canon_q35_pp": round(float(np.mean(calib_shifts_q35)), 2),
    "calib_mean_shift_vs_canon_mean_pp": round(float(np.mean(calib_shifts_mean)), 2),
    "routed_per_class": routed_bd,
    "calib_per_class": calib_bd,
}
json.dump(out, open(OUT, "w"), indent=1)
print(f"Saved {OUT}")
print(f"realized={realized:.2f}%  expected q35=55.85  diff={realized-55.85:+.2f} pp")
print(f"calib shift vs canon q35: {np.mean(calib_shifts_q35):+.2f} pp")
