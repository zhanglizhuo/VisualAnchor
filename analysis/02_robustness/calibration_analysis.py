"""
Calibration analysis: AnchorScore vs MLLM accuracy (47 classes, 5 bins).
Single source of truth = pooled_class_level_results.json (canonical).
Bin definition and ECE formula must match paper/generate_figures.py figA2 exactly.
"""
import json
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"

POOLED = RESULTS / "01_core" / "correlation" / "pooled_class_level_results.json"
OUT = RESULTS / "02_robustness" / "calibration" / "calibration_analysis.json"

with open(POOLED) as f:
    pooled = json.load(f)

data = pooled["data"]
anchor = np.array([e["anchor_score"] for e in data])
mllm = np.array([e["mllm_mean"] for e in data])
n_classes = len(anchor)

n_bins = 5
bin_edges = np.linspace(0, 100, n_bins + 1)
bin_details = []
ece = 0.0
for i in range(n_bins):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    mask = (anchor >= lo) & (anchor <= hi) if i == n_bins - 1 else (anchor >= lo) & (anchor < hi)
    n_bin = int(mask.sum())
    a_mean = float(anchor[mask].mean()) if n_bin > 0 else 0.0
    m_mean = float(mllm[mask].mean()) if n_bin > 0 else 0.0
    ece += abs(a_mean - m_mean) * (n_bin / n_classes)
    bin_details.append({
        "bin": i + 1,
        "range": [round(float(lo), 1), round(float(hi), 1)],
        "n_classes": n_bin,
        "mean_anchor": round(a_mean, 2),
        "mean_mllm": round(m_mean, 2),
        "abs_error": round(abs(m_mean - a_mean), 2),
    })

ece /= 100.0
summary = {
    "description": "Calibration analysis: AnchorScore vs MLLM accuracy",
    "source": str(POOLED.relative_to(BASE)),
    "n_classes": n_classes,
    "n_bins": n_bins,
    "ece": round(ece, 3),
    "ece_pct": round(ece * 100, 1),
    "bin_details": bin_details,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved {OUT}")
print(f"ECE = {summary['ece']} ({summary['ece_pct']}%)")
for b in bin_details:
    print(f"  bin{b['bin']}: n={b['n_classes']:2d} anchor={b['mean_anchor']:6.2f} mllm={b['mean_mllm']:6.2f} abs_err={b['abs_error']:5.2f}")
