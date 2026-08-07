#!/usr/bin/env python3
"""
three_tier_stats.py
===================
Regenerates results/02_robustness/robustness/three_tier_47class.json
from the canonical pooled class-level data
(results/01_core/correlation/pooled_class_level_results.json).

Three-tier heuristic: AnchorScore <=10% (low), (10,40]% (mid), >40% (high).
Runs from a single source of truth so the evidence file can never drift
from the pooled data that generates fig6_calibration.
"""
import json
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"

with open(RESULTS / "01_core" / "correlation" / "pooled_class_level_results.json") as f:
    pooled = json.load(f)

data = pooled["data"]
tiers = {"low": [], "mid": [], "high": []}
for e in data:
    a = e["anchor_score"]
    if a <= 10:
        tiers["low"].append(e["mllm_mean"])
    elif a <= 40:
        tiers["mid"].append(e["mllm_mean"])
    else:
        tiers["high"].append(e["mllm_mean"])

names = {"low": "<=10%", "mid": "10-40%", "high": ">40%"}
summary = {
    "description": "Three-tier heuristic on pooled 47-class data",
    "tiers": {
        k: {
            "threshold": names[k],
            "n": len(v),
            "mean_mllm": round(float(np.mean(v)), 1) if v else 0.0,
            "range": [round(float(min(v)), 1), round(float(max(v)), 1)] if v else [0.0, 0.0],
        }
        for k, v in tiers.items()
    },
}

out_path = RESULTS / "02_robustness" / "robustness" / "three_tier_47class.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved {out_path}")
for k, v in summary["tiers"].items():
    print(f"  {k:4s} {v['threshold']:>8s}: n={v['n']:2d} mean={v['mean_mllm']:.1f} range={v['range']}")
