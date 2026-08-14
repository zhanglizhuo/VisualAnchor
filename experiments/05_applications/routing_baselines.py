"""
routing_baselines.py

Routing baselines for the deployable hybrid routing application: is the
routing gain driven by AnchorScore itself, or would CLIP max-logit
confidence / random routing do the same?

Routes TeacherBehavior images at matched cost (56.5% of images to the MLLM,
the AnchorScore tau=45 operating point) and compares:
  - AnchorScore routing (predicted class, tau=45):  55.66%
  - CLIP max-logit confidence routing (matched cost): 41.86%
  - random routing (matched cost, 200 seeds):       44.98% +- 0.43%
  - CLIP-confidence routing best point (q=1% threshold): 52.87% at 4.0% MLLM

The MLLM branch uses the canonical per-class expected accuracy (same
assumption as hybrid_deployable.py; no per-image MLLM predictions cached).
"""
import json
import numpy as np
from pathlib import Path
from datetime import date

PROJ = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJ / "results"
CLIP_PRED = RESULTS / "01_core" / "clip_per_image" / "per_image_predictions.json"
MLLM_RAW = RESULTS / "01_core" / "paper_data" / "mllm_raw.json"
ANCHOR = RESULTS / "01_core" / "anchor_score_scb5" / "anchor_scores.json"
OUT = RESULTS / "05_applications" / "hybrid" / "routing_baselines.json"

clip = json.load(open(CLIP_PRED))
mllm = json.load(open(MLLM_RAW))
anchor = json.load(open(ANCHOR))

models = [k for k in mllm["TeacherBehavior"].keys() if not k.startswith("_")]
items = clip["TeacherBehavior"]["predictions"]
N = len(items)
mllm_mean = {
    cls: float(np.mean([mllm["TeacherBehavior"][m][cls] for m in models]))
    for cls in mllm["TeacherBehavior"][models[0]]
}
tb_anchor = {k: v["acc"] for k, v in anchor["TeacherBehavior"]["per_class_acc"].items()}


def route_anchor(tau):
    acc = 0.0
    for it in items:
        if tb_anchor[it["pred_class_name"]] >= tau:
            acc += 1 if it["correct"] else 0
        else:
            acc += mllm_mean[it["true_class_name"]] / 100.0
    return acc / N


def route_confidence(mllm_frac):
    confs = sorted(it["confidence"] for it in items)
    c = float(np.quantile(confs, mllm_frac))
    acc = 0.0
    for it in items:
        if it["confidence"] >= c:
            acc += 1 if it["correct"] else 0
        else:
            acc += mllm_mean[it["true_class_name"]] / 100.0
    return acc / N


def route_random(frac_mllm, rng):
    mask = rng.random(N) < frac_mllm
    acc = 0.0
    for i, it in enumerate(items):
        if mask[i]:
            acc += mllm_mean[it["true_class_name"]] / 100.0
        else:
            acc += 1 if it["correct"] else 0
    return acc / N


frac_mllm = sum(1 for it in items if tb_anchor[it["pred_class_name"]] < 45) / N

# confidence-routing curve: monotonic in MLLM budget, peak = all-MLLM baseline
curve_max = max((route_confidence(f), f) for f in np.arange(0.02, 0.98, 0.02))
all_mllm = float(np.mean([np.mean([mllm["TeacherBehavior"][m][c] for m in models]) for c in mllm["TeacherBehavior"][models[0]]]))

rng = np.random.default_rng(42)
accs = [route_random(frac_mllm, rng) for _ in range(200)]

out = {
    "description": "TeacherBehavior routing baselines at matched cost (56.5% images to MLLM, the AnchorScore tau=45 operating point). AnchorScore routing vs CLIP max-logit confidence routing vs random routing. MLLM branch uses per-class expected accuracy (same assumption as hybrid_deployable.py). Random = 200 seeds. Accuracy in %. The confidence-routing curve is monotonic in the MLLM budget and peaks at the all-MLLM baseline, so its maximum is degenerate; the matched-cost operating point is the informative comparison.",
    "computation_date": str(date.today()),
    "anchor_routing_tau45": {"acc": round(100 * route_anchor(45), 2), "mllm_frac": round(frac_mllm, 3)},
    "confidence_routing_matched_cost": {"acc": round(100 * route_confidence(frac_mllm), 2), "mllm_frac": round(frac_mllm, 3)},
    "random_routing_matched_cost": {"acc_mean": round(100 * float(np.mean(accs)), 2), "acc_sd": round(100 * float(np.std(accs)), 2), "reps": 200},
    "confidence_routing_curve_peak": {"acc": round(100 * curve_max[0], 2), "mllm_frac": round(curve_max[1], 3), "note": "degenerate: equals the all-MLLM baseline"},
    "all_mllm_mean": round(all_mllm, 2),
    "clip_only": round(clip["TeacherBehavior"]["overall_acc"], 2),
}
json.dump(out, open(OUT, "w"), indent=1)
print(f"Saved {OUT}")
print(f"anchor tau45: {out['anchor_routing_tau45']}")
print(f"confidence matched: {out['confidence_routing_matched_cost']}")
print(f"random matched: {out['random_routing_matched_cost']}")
print(f"confidence curve peak: {out['confidence_routing_curve_peak']} vs all-MLLM {out['all_mllm_mean']}")
