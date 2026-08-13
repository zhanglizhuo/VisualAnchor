"""Deployable (predicted-class) hybrid routing for Stanford40.

Routes each of the 9,532 Stanford40 validation images by CLIP's *predicted*
class (no ground truth at routing time), mirroring the SCB5 analysis in
hybrid_deployable.py:

    if AnchorScore(CLIP(x)) >= tau: use CLIP(x)
    else:                           use a fixed MLLM on x

MLLM branch accuracy uses the canonical per-class rates from
ollama_results.json (50 images/class evaluated per MLLM), the same
expected-value assumption as the SCB5 deployable analysis.

Output: results/05_applications/hybrid_stanford40/hybrid_deployable.json
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"
CLIP_PRED = RESULTS / "02_robustness" / "stanford40" / "per_image_predictions.json"
ANCHOR = RESULTS / "02_robustness" / "stanford40" / "anchor_scores.json"
MLLM = RESULTS / "02_robustness" / "stanford40" / "ollama_results.json"
OUT_DIR = RESULTS / "05_applications" / "hybrid_stanford40"
OUT = OUT_DIR / "hybrid_deployable.json"

clip = json.load(open(CLIP_PRED))
anchor = json.load(open(ANCHOR))
mllm = json.load(open(MLLM))

anchor_pc = anchor["per_class_acc"]
models = list(mllm["models"].keys())

preds = clip["predictions"]
N = len(preds)
rows = []
for p in preds:
    y = p["true_class_name"]
    c = p["pred_class_name"]
    correct = 1 if y == c else 0
    a = anchor_pc[c]["acc"]
    rows.append((correct, a, y))
clip_only = sum(r[0] for r in rows) / N * 100

out = {"n_images": N, "clip_only_acc": round(clip_only, 2),
       "deployable": {}, "oracle": {}}

def mllm_rate(mk, y):
    if mk == "mean":
        return sum(mllm["models"][m]["per_class_acc"][y]["acc"] for m in models) / len(models)
    return mllm["models"][mk]["per_class_acc"][y]["acc"]

for tau in range(0, 101, 5):
    for mk in models + ["mean"]:
        acc_sum = 0.0
        n_mllm = 0
        for correct, a, y in rows:
            if a >= tau:
                acc_sum += correct
            else:
                acc_sum += mllm_rate(mk, y) / 100.0
                n_mllm += 1
        out["deployable"].setdefault(mk, {})[tau] = {
            "acc": round(acc_sum / N * 100, 2),
            "cost_saved_pct": round((1 - n_mllm / N) * 100, 1)}
    for mk in models + ["mean"]:
        acc_sum = 0.0
        n_mllm = 0
        for correct, a, y in rows:
            if anchor_pc[y]["acc"] >= tau:
                acc_sum += correct
            else:
                acc_sum += mllm_rate(mk, y) / 100.0
                n_mllm += 1
        out["oracle"].setdefault(mk, {})[tau] = {
            "acc": round(acc_sum / N * 100, 2),
            "cost_saved_pct": round((1 - n_mllm / N) * 100, 1)}

out["_meta"] = {
    "description": "Deployable predicted-class routing vs oracle true-class routing on Stanford40 (9,532 images). MLLM branch uses per-class rates from ollama_results.json (50 images/class per MLLM); expected-value estimator. CLIP predictions from stanford40/per_image_predictions.json (ViT-L-14 laion2b_s32b_b82k, 3-template averaged; overall 91.87% vs canonical 91.86%).",
    "source_files": [str(CLIP_PRED), str(ANCHOR), str(MLLM)],
    "assumption": "P(MLLM correct | routed) = per-class MLLM accuracy; expected-value estimate",
    "computation_date": "2026-08-12",
}
OUT_DIR.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT, "w"), indent=2)
print("saved", OUT)
print(f"N={N}, CLIP-only {clip_only:.2f}%")
print("\nbest deployable per model (>=10% saving):")
for mk in models + ["mean"]:
    cands = [(int(t), r["acc"], r["cost_saved_pct"]) for t, r in out["deployable"][mk].items()
             if r["cost_saved_pct"] >= 10.0]
    if cands:
        t_, a_, s_ = max(cands, key=lambda x: x[1])
        print(f"  {mk:14s} tau={t_} acc={a_:6.2f}% (+{a_-clip_only:+5.1f}pp) saved={s_:5.1f}%")
