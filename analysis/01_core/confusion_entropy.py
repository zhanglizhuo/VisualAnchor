#!/usr/bin/env python3
"""
Confusion entropy analysis.
For each class, compute how scattered CLIP's predictions are (entropy of
top-1 predicted-class distribution per true class).
Then correlate with MLLM accuracy.

Stop-loss: only include in paper if p < 0.05 and ρ direction is consistent.

Definition:
  For true class c, let p(pred=k|true=c) = count of images in class c
  where CLIP predicted class k / total images in class c.
  Confusion entropy H_agg(c) = -sum_k p(pred=k|true=c) * log2(p(pred=k|true=c))

  High H_agg means CLIP's top-1 predictions are scattered across many classes
  (genuine confusion). Low H_agg but low accuracy means CLIP consistently
  mispredicts the same wrong class (systematic composition error).

Usage:
  python analysis/01_core/confusion_entropy.py
"""
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent.parent.parent
CLIP_FILE = BASE / "results" / "01_core" / "clip_per_image" / "per_image_predictions.json"
MLLM_FILE = BASE / "results" / "01_core" / "paper_data" / "mllm_full.json"
OUT_DIR = BASE / "results" / "01_core" / "confusion_entropy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CLIP_FILE) as f:
    clip_data = json.load(f)

with open(MLLM_FILE) as f:
    mllm_data = json.load(f)

CANONICAL_6 = [
    "Qwen3.5-27B", "Qwen3.6-27B", "Qwen3.5-35B",
    "Qwen3.6-35B", "Gemma4-31B", "Gemma4-26B",
]

CLASS_MAP = {
    "TeacherBehavior": [
        "guide", "answer", "On-stage interaction", "blackboard-writing",
        "teacher", "stand", "screen", "blackBoard",
    ],
    "HandriseReadWrite": ["hand-raising", "read", "write"],
    "BowTurnHead": ["BowHead", "TurnHead"],
}


def compute_entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


# ── Compute per-class confusion entropy from CLIP predictions ──
results = []
for dataset, classes in CLASS_MAP.items():
    cdata = clip_data.get(dataset)
    if cdata is None:
        continue
    preds = cdata.get("predictions", [])
    true_to_pred_counts = {cls: {} for cls in classes}
    confidences = {cls: [] for cls in classes}

    for p in preds:
        tc = p["true_class_name"]
        pc = p["pred_class_name"]
        true_to_pred_counts[tc][pc] = true_to_pred_counts[tc].get(pc, 0) + 1
        confidences[tc].append(p["confidence"])

    for cls in classes:
        counts = list(true_to_pred_counts[cls].values())
        entropy = compute_entropy(counts)
        mean_conf = float(np.mean(confidences[cls])) if confidences[cls] else 0.0
        n = sum(counts)
        self_count = true_to_pred_counts[cls].get(cls, 0)
        correct_rate = self_count / n if n > 0 else 0.0

        # Top confused class (where most errors go)
        sorted_preds = sorted(true_to_pred_counts[cls].items(), key=lambda x: -x[1])
        top_confusion = [p for p in sorted_preds if p[0] != cls]
        top_confused_class = top_confusion[0][0] if top_confusion else None
        top_confused_pct = top_confusion[0][1] / n * 100 if top_confusion else 0.0

        results.append({
            "class": cls,
            "dataset": dataset,
            "n": n,
            "anchor_rate": round(correct_rate * 100, 2),
            "confusion_entropy": round(entropy, 4),
            "mean_confidence": round(mean_conf, 4),
            "top_confused_class": top_confused_class,
            "top_confused_pct": round(top_confused_pct, 2),
        })

# ── Get MLLM per-class means ──
mllm_means = {}
for dataset, classes in CLASS_MAP.items():
    for cls in classes:
        accs = []
        for model in CANONICAL_6:
            v = mllm_data.get(dataset, {}).get(model, {}).get(cls)
            if v is not None:
                accs.append(v)
        mllm_means[(dataset, cls)] = float(np.mean(accs)) if accs else float("nan")

# ── Correlate ──
entropy_vals, mllm_vals, conf_vals = [], [], []
for r in results:
    key = (r["dataset"], r["class"])
    m = mllm_means.get(key)
    if m is not None and not math.isnan(m):
        entropy_vals.append(r["confusion_entropy"])
        conf_vals.append(r["mean_confidence"])
        mllm_vals.append(m)

entropy_vals = np.array(entropy_vals)
mllm_vals = np.array(mllm_vals)

rho_ent, p_ent = spearmanr(entropy_vals, mllm_vals)
rho_conf, p_conf = spearmanr(conf_vals, mllm_vals)

# ── Output ──
output = {
    "n_classes": len(entropy_vals),
    "n_mllms": len(CANONICAL_6),
    "entropy_vs_mllm": {"spearman_rho": round(rho_ent, 4), "p": round(p_ent, 6)},
    "mean_confidence_vs_mllm": {"spearman_rho": round(rho_conf, 4), "p": round(p_conf, 6)},
    "per_class": results,
}

# Save
OUT_FILE = OUT_DIR / "confusion_entropy_results.json"
with open(OUT_FILE, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved to {OUT_FILE}")

# ── Console report ──
print("=" * 60)
print("Confusion Entropy vs. MLLM Accuracy (canonical 6 MLLMs)")
print("=" * 60)
print(f"\nEntropy vs. MLLM accuracy:  ρ = {rho_ent:.4f}, p = {p_ent:.6f}, n = {len(entropy_vals)}")
print(f"Mean confidence vs. MLLM:  ρ = {rho_conf:.4f}, p = {p_conf:.6f}, n = {len(conf_vals)}")

print("\n--- Per-dataset ---")
for dataset in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
    e_vals = [r["confusion_entropy"] for r in results if r["dataset"] == dataset]
    m_vals = [mllm_means[(dataset, r["class"])] for r in results if r["dataset"] == dataset]
    if len(e_vals) >= 2:
        rho, p = spearmanr(e_vals, m_vals)
        print(f"{dataset:30s}  ρ = {rho:.4f}, p = {p:.4f}, n = {len(e_vals)}")

print("\n--- Per-class detail ---")
print(f"{'Class':35s} {'n':>5s} {'Acc%':>7s} {'Entropy':>8s} {'Conf':>8s} {'MLLM%':>8s}  {'Top confusion'}")
for r in sorted(results, key=lambda x: x["confusion_entropy"], reverse=True):
    key = (r["dataset"], r["class"])
    m = mllm_means.get(key, float("nan"))
    if r["top_confused_class"]:
        tc = f"{r['top_confused_class']} ({r['top_confused_pct']:.0f}%)"
    else:
        tc = "--"
    print(f"{r['class']:35s} {r['n']:5d} {r['anchor_rate']:>7} {r['confusion_entropy']:>8.4f} {r['mean_confidence']:>8.4f} {m:>8.2f}  {tc}")

# ── Decision ──
print("\n" + "=" * 60)
alpha = 0.05
if p_ent < alpha and rho_ent < 0:
    print(f"✅ SIGNIFICANT: ρ = {rho_ent:.4f}, p = {p_ent:.6f}")
    print(f"   Recommendation: ADD to paper as evidence refinement.")
elif p_ent < alpha and rho_ent > 0:
    print(f"⚠️ CONTRADICTORY: ρ = {rho_ent:.4f}, p = {p_ent:.6f}")
    print(f"   Recommendation: DO NOT ADD.")
else:
    print(f"❌ NOT SIGNIFICANT: ρ = {rho_ent:.4f}, p = {p_ent:.6f}")
    print(f"   Recommendation: ARCHIVE. Do not include.")
print("=" * 60)
