"""
realized_routing_fullframe_aggregate.py

Aggregates the full-frame realized-routing shard outputs (run on V100) into
results/05_applications/hybrid/realized_routing_fullframe.json.

Full-frame variant = the deployment condition: the MLLM fallback receives the
entire image (no annotation-derived bbox crop), category-name prompt, CLIP
predicted-class routing at tau=45. Contrast with realized_routing.json, whose
fallback used the benchmark's canonical bbox-crop protocol.
"""
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJ / "results"
RAW = RESULTS / "05_applications" / "hybrid" / "realized_routing_fullframe_raw.jsonl"
CLIP_CACHE = RESULTS / "01_core" / "clip_per_image" / "per_image_predictions.json"
ANCHOR = RESULTS / "01_core" / "anchor_score_scb5" / "anchor_scores.json"
MLLM_RAW = RESULTS / "01_core" / "paper_data" / "mllm_raw.json"
HYBRID = RESULTS / "05_applications" / "hybrid" / "hybrid_deployable.json"
OUT = RESULTS / "05_applications" / "hybrid" / "realized_routing_fullframe.json"

CLASSES = ["guide", "answer", "on-stage interaction", "blackboard-writing",
           "teacher", "stand", "screen", "blackboard"]
RAW_KEYS = ["guide", "answer", "On-stage interaction", "blackboard-writing",
            "teacher", "stand", "screen", "blackBoard"]


def main():
    recs = [json.loads(ln) for ln in open(RAW)]
    assert len(recs) == 2072
    routed = [r for r in recs if r["arm"] == "routed"]
    calib = [r for r in recs if r["arm"] == "calib"]

    pred = json.load(open(CLIP_CACHE))
    anchor = json.load(open(ANCHOR))
    tb_anchor = {k: v["acc"] for k, v in anchor["TeacherBehavior"]["per_class_acc"].items()}
    items = pred["TeacherBehavior"]["predictions"]
    N = len(items)
    n_kept = sum(1 for it in items if tb_anchor[it["pred_class_name"]] >= 45)
    kept_c = sum(1 for it in items if tb_anchor[it["pred_class_name"]] >= 45 and it["correct"])
    ff_c = sum(1 for r in routed if r["correct"] is True)
    realized = (kept_c + ff_c) / N * 100

    hybrid = json.load(open(HYBRID))
    expected_q35 = hybrid["TeacherBehavior"]["deployable"]["Qwen3.5-27B"]["45"]["acc"]

    by_cls = defaultdict(lambda: [0, 0])
    for r in routed:
        by_cls[r["true"]][1] += 1
        if r["correct"]:
            by_cls[r["true"]][0] += 1
    mllm = json.load(open(MLLM_RAW))
    models = [k for k in mllm["TeacherBehavior"].keys() if not k.startswith("_")]
    per_class = {}
    for c in sorted(by_cls):
        n_ok, n_t = by_cls[c]
        canon = float(np.mean([mllm["TeacherBehavior"][m][RAW_KEYS[c]] for m in models]))
        per_class[CLASSES[c]] = {
            "fullframe_acc": round(100 * n_ok / n_t, 2),
            "canonical_bbox_acc": round(canon, 2),
            "n": n_t,
        }

    out = {
        "description": (
            "Realized routing under the deployment condition: TeacherBehavior tau=45, "
            "Qwen3.5-27B via Ollama (q4), full-frame images (no annotation-derived bbox "
            "crop), category-name prompt. The routing decision uses only CLIP predicted "
            "classes; no ground truth is used at decision time. Contrast with "
            "realized_routing.json (bbox-crop fallback, canonical protocol): the protocol "
            "difference costs 4.2 pp of pipeline accuracy, while the realized deployment "
            "gain over CLIP-only remains +17.5 pp."
        ),
        "computation_date": str(date.today()),
        "model": "Qwen3.5-27B (Ollama q4)",
        "protocol": "full-frame (no bbox), category-name reply",
        "realized_pipeline_acc": round(realized, 2),
        "expected_q35": round(expected_q35, 2),
        "bbox_realized_reference": 54.07,
        "clip_only": 32.38,
        "gain_over_clip_only_pp": round(realized - 32.38, 2),
        "protocol_cost_pp": round(54.07 - realized, 2),
        "clip_branch": {"n": n_kept, "correct": kept_c, "acc": round(100 * kept_c / n_kept, 2)},
        "mllm_branch": {"n": len(routed), "correct": ff_c, "acc": round(100 * ff_c / len(routed), 2)},
        "routed_per_class": per_class,
        "calib_n": len(calib),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"Saved {OUT}")
    print(f"realized full-frame = {realized:.2f}%  (+{realized-32.38:.1f} pp over CLIP-only; "
          f"bbox realized was 54.07%, protocol cost {54.07-realized:.1f} pp)")


if __name__ == "__main__":
    main()
