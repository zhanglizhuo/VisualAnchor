"""Deployable (predicted-class) hybrid routing for SCB5.

The oracle simulation (hybrid_fixed_mllm.py) routes each image by its
ground-truth class. This script routes by CLIP's *predicted* class, which is
the decision a practitioner can actually make on an unlabeled image:

    if AnchorScore(CLIP(x)) >= tau: use CLIP(x)
    else:                           use a fixed MLLM on x

Accuracy decomposition (expected value):
    acc = (1/N) * sum_x [ routed_to_clip(x) * I[CLIP(x)=y(x)]
                          + routed_to_mllm(x) * MLLM_acc(y(x)) ]
where MLLM_acc(y) is the canonical per-class accuracy of the fixed MLLM
(same assumption class as the oracle simulation; the MLLM term uses the
true class only through the per-class expected rate).

Cost saving = fraction of images routed to CLIP.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"
CLIP_PRED = RESULTS / "01_core" / "clip_per_image" / "per_image_predictions.json"
MLLM_RAW = RESULTS / "01_core" / "paper_data" / "mllm_raw.json"
ANCHOR = RESULTS / "01_core" / "anchor_score_scb5" / "anchor_scores.json"
OUT = RESULTS / "05_applications" / "hybrid" / "hybrid_deployable.json"

clip = json.load(open(CLIP_PRED))
mllm = json.load(open(MLLM_RAW))
anchor = json.load(open(ANCHOR))

# Canonical MLLM models (order per mllm_raw)
models = [k for k in mllm["TeacherBehavior"].keys()]

def anchor_of(ds, cls):
    return anchor[ds]["per_class_acc"][cls]["acc"]

out = {}
for ds in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
    preds = clip[ds]["predictions"]
    N = len(preds)
    # per-image: CLIP correctness, predicted class anchor
    rows = []
    for p in preds:
        y = p["true_class_name"]
        c = p["pred_class_name"]
        correct = 1 if y == c else 0
        a = anchor_of(ds, c)
        rows.append((correct, a, y))
    # CLIP-only baseline (from per-image data)
    clip_only = sum(r[0] for r in rows) / N * 100
    # Oracle per-true-class routing (reproduces hybrid_fixed_mllm structure)
    gt_by_class = defaultdict(list)
    for correct, a, y in rows:
        gt_by_class[y].append(correct)
    ds_out = {"n_images": N, "clip_only_acc": round(clip_only, 2),
              "deployable": {}, "oracle": {}}
    for tau in range(0, 101, 5):
        # deployable: route by predicted class anchor
        for mk in models + ["mean"]:
            if mk == "mean":
                macc = lambda y: sum(mllm[ds][m][y] for m in models) / len(models)
            else:
                macc = lambda y: mllm[ds][mk][y]
            acc_sum = 0.0
            n_mllm = 0
            for correct, a, y in rows:
                if a >= tau:
                    acc_sum += correct
                else:
                    acc_sum += macc(y) / 100.0
                    n_mllm += 1
            acc = acc_sum / N * 100
            cost_saved = (1 - n_mllm / N) * 100
            ds_out["deployable"].setdefault(mk, {})[tau] = {
                "acc": round(acc, 2), "cost_saved_pct": round(cost_saved, 1)}
        # oracle: route by true class
        for mk in models + ["mean"]:
            if mk == "mean":
                macc = lambda y: sum(mllm[ds][m][y] for m in models) / len(models)
            else:
                macc = lambda y: mllm[ds][mk][y]
            acc_sum = 0.0
            n_mllm = 0
            for correct, a, y in rows:
                a_true = anchor_of(ds, y)
                if a_true >= tau:
                    acc_sum += correct
                else:
                    acc_sum += macc(y) / 100.0
                    n_mllm += 1
            acc = acc_sum / N * 100
            cost_saved = (1 - n_mllm / N) * 100
            ds_out["oracle"].setdefault(mk, {})[tau] = {
                "acc": round(acc, 2), "cost_saved_pct": round(cost_saved, 1)}
    out[ds] = ds_out

out["_meta"] = {
    "description": "Deployable predicted-class routing vs oracle true-class routing on SCB5. MLLM branch accuracy uses canonical per-class rates (expected-value estimator). CLIP predictions: clip_per_image run, same protocol as canonical AnchorScore (per-class within 0.5pp).",
    "source_files": [str(CLIP_PRED), str(MLLM_RAW), str(ANCHOR)],
    "assumption": "P(MLLM correct | routed) = per-class MLLM accuracy; expected-value estimate, no per-image MLLM predictions used",
    "computation_date": "2026-08-12",
}
json.dump(out, open(OUT, "w"), indent=2)
print("saved", OUT)

# Print highlights: best deployable vs oracle at tau=45 for TB
for ds in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
    d = out[ds]
    print(f"\n=== {ds} (N={d['n_images']}, CLIP-only {d['clip_only_acc']}%) ===")
    for tau in [40, 45, 50]:
        dep_mean = d["deployable"]["mean"][str(tau)]
        ora_mean = d["oracle"]["mean"][str(tau)]
        print(f"  tau={tau}: deployable(mean-MLLM) acc={dep_mean['acc']}% saved={dep_mean['cost_saved_pct']}% | oracle acc={ora_mean['acc']}% saved={ora_mean['cost_saved_pct']}%")
