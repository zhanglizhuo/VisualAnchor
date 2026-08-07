"""
Robustness analysis for AnchorScore paper (Experiments #2-#5).
Uses VisualAnchor data files directly.

Experiments:
  #2 SCB5 subset class-level meta-analysis
  #3 Leave-2-classes-out stability
  #4 AUC permutation test + LOCO AUC
  #5 TeacherBehavior vs headline Fisher z-test
"""

import json
import math
import random
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"

# ── Load data ──────────────────────────────────────────────────────────────
with open(RESULTS / "01_core" / "anchor_score_scb5" / "anchor_scores.json") as f:
    anchor = json.load(f)

with open(RESULTS / "01_core" / "paper_data" / "mllm_full.json") as f:
    mllm_src = json.load(f)

MODELS_OF_INTEREST = ["Qwen3.5-27B", "Qwen3.6-27B", "Qwen3.5-35B", "Qwen3.6-35B",
                      "Gemma-3-27B", "Gemma4-31B", "Gemma4-26B"]

CLASS_MAP = {
    "TeacherBehavior": ["guide", "answer", "On-stage interaction",
                        "blackboard-writing", "teacher", "stand", "screen", "blackBoard"],
    "HandriseReadWrite": ["hand-raising", "read", "write"],
    "BowTurnHead": ["BowHead", "TurnHead"],
}

def get_mllm_mean(ds_name, cname):
    """Mean MLLM accuracy for a class across all models."""
    vals = []
    for m in MODELS_OF_INTEREST:
        if m in mllm_src.get(ds_name, {}):
            v = mllm_src[ds_name][m].get(cname)
            if v is not None and not math.isnan(v):
                vals.append(v)
    return float(np.mean(vals)) if vals else 0.0

# Build 13-class table
rows = []
for ds_name, cls_list in CLASS_MAP.items():
    for cname in cls_list:
        anchor_val = anchor[ds_name]["per_class_acc"][cname]["acc"]
        mllm_mean = get_mllm_mean(ds_name, cname)
        rows.append({"dataset": ds_name, "class": cname,
                     "anchor_score": anchor_val, "mllm_mean": mllm_mean})

N_CLASSES = len(rows)
print(f"Loaded {N_CLASSES} classes")

# Per-class data table
print(f"\n{'Dataset':20s} {'Class':25s} {'Anchor':>7s} {'MLLM Avg':>9s}")
print("-" * 62)
for r in rows:
    ds = r['dataset'] if r['class'] == CLASS_MAP[r['dataset']][0] else ''
    print(f"{ds:20s} {r['class']:25s} {r['anchor_score']:6.1f}% {r['mllm_mean']:7.1f}%")

# ── Helpers ────────────────────────────────────────────────────────────────

def spearman_ci(x, y, n_bootstrap=1999, alpha=0.05):
    """Spearman ρ with percentile bootstrap CI."""
    rho, p = stats.spearmanr(x, y)
    if np.isnan(rho):
        return float(rho), float(p), np.nan, np.nan

    x_a, y_a = np.array(x), np.array(y)
    n = len(x_a)
    boot_rhos = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, n)
        if len(np.unique(x_a[idx])) < 2 or len(np.unique(y_a[idx])) < 2:
            continue
        r, _ = stats.spearmanr(x_a[idx], y_a[idx])
        if not np.isnan(r):
            boot_rhos.append(r)

    ci_low = float(np.percentile(boot_rhos, 100 * alpha / 2)) if len(boot_rhos) > 100 else np.nan
    ci_high = float(np.percentile(boot_rhos, 100 * (1 - alpha / 2))) if len(boot_rhos) > 100 else np.nan
    return float(rho), float(p), ci_low, ci_high


def fisher_z(r):
    return np.arctanh(r)


def fisher_z_test(r1, r2, n1, n2):
    z1 = fisher_z(r1)
    z2 = fisher_z(r2)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


# ── Experiment #2: SCB5 Subset Meta-Analysis ───────────────────────────────
print(f"\n{'='*70}")
print("EXPERIMENT #2: SCB5 Subset Class-Level Meta-Analysis")
print(f"{'='*70}")

subsets = {
    "TeacherBehavior (n=8)": "TeacherBehavior",
    "HandriseReadWrite (n=3)": "HandriseReadWrite",
    "BowTurnHead (n=2)": "BowTurnHead",
}

subset_data = {}
all_rho = []
all_var = []
for label, ds_name in subsets.items():
    sub = [r for r in rows if r['dataset'] == ds_name]
    x = [r['anchor_score'] for r in sub]
    y = [r['mllm_mean'] for r in sub]
    rho, p, ci_low, ci_high = spearman_ci(x, y)
    subset_data[label] = {"n": len(sub), "rho": rho, "p": p, "ci_low": ci_low, "ci_high": ci_high}
    sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else " n.s."
    print(f"  {label:25s}: ρ={rho:.4f}, p={p:.4f}{sig}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]")
    all_rho.append(rho)
    all_var.append(1.0 / max(len(sub) - 3, 0.01))

# Fixed-effects meta-analysis
if len(all_rho) >= 2:
    z_vals = [fisher_z(r) for r in all_rho]
    weights = [1 / v for v in all_var]
    w_sum = sum(weights)
    z_pooled = sum(w * z for w, z in zip(weights, z_vals)) / w_sum
    r_pooled = float(np.tanh(z_pooled))
    se_pooled = np.sqrt(1 / w_sum)
    ci_low_p = float(np.tanh(z_pooled - 1.96 * se_pooled))
    ci_high_p = float(np.tanh(z_pooled + 1.96 * se_pooled))
    Q = sum(w * (z - z_pooled)**2 for w, z in zip(weights, z_vals))
    I2 = max(0, (Q - (len(all_rho) - 1)) / Q * 100) if Q > 0 else 0
    print(f"\n  Fixed-effects pooled ρ={r_pooled:.4f}, 95% CI [{ci_low_p:.3f}, {ci_high_p:.3f}]")
    print(f"  Heterogeneity: Q={Q:.2f}, I²={I2:.1f}%")

    # Fisher z-test: TeacherBehavior vs pooled
    tb = subset_data.get("TeacherBehavior (n=8)", {})
    if tb.get("n", 0) >= 4:
        z, p = fisher_z_test(tb['rho'], r_pooled, tb['n'], len(rows))
        print(f"  TeacherBehavior vs Pooled: z={z:.3f}, p={p:.4f}")


# ── Experiment #3: Leave-2-Classes-Out Stability ──────────────────────────
print(f"\n{'='*70}")
print("EXPERIMENT #3: Leave-2-Classes-Out Stability")
print(f"{'='*70}")

x_all = np.array([r['anchor_score'] for r in rows])
y_all = np.array([r['mllm_mean'] for r in rows])
rho_head, _ = stats.spearmanr(x_all, y_all)
print(f"  Headline ρ={rho_head:.4f} (n={N_CLASSES})")

rho_vals = []
for pair in combinations(range(N_CLASSES), 2):
    keep = [i for i in range(N_CLASSES) if i not in pair]
    r, _ = stats.spearmanr(x_all[keep], y_all[keep])
    if not np.isnan(r):
        rho_vals.append(r)
rho_vals = np.array(rho_vals)

print(f"  N leave-2-out combinations: {len(rho_vals)}")
print(f"  ρ range: [{np.min(rho_vals):.4f}, {np.max(rho_vals):.4f}]")
print(f"  Mean={np.mean(rho_vals):.4f}, SD={np.std(rho_vals):.4f}")
print(f"  P5={np.percentile(rho_vals, 5):.4f}, P25={np.percentile(rho_vals, 25):.4f}")
print(f"  Median={np.median(rho_vals):.4f}, P75={np.percentile(rho_vals, 75):.4f}, P95={np.percentile(rho_vals, 95):.4f}")
print(f"  Fraction ρ < 0.50: {np.mean(rho_vals < 0.50)*100:.1f}%")
print(f"  Fraction ρ > 0.70: {np.mean(rho_vals > 0.70)*100:.1f}%")
print(f"  All positive? {np.all(rho_vals > 0)}")


# ── Experiment #4: AUC Permutation Test ───────────────────────────────────
print(f"\n{'='*70}")
print("EXPERIMENT #4: AUC Permutation Test")
print(f"{'='*70}")

from sklearn.metrics import roc_auc_score

# TeacherBehavior: k=8, meaningful AUC
tb_idx = [i for i, r in enumerate(rows) if r['dataset'] == 'TeacherBehavior']
x_tb = x_all[tb_idx]
y_tb = y_all[tb_idx]
median_tb = np.median(y_tb)
y_bin = (y_tb >= median_tb).astype(int)
n_hard = np.sum(y_bin == 0)
n_easy = np.sum(y_bin == 1)
print(f"  TeacherBehavior (k=8): {n_hard} hard, {n_easy} easy (median MLLM={median_tb:.2f}%)")

if len(np.unique(y_bin)) >= 2:
    auc = roc_auc_score(y_bin, x_tb)
    print(f"  AUC = {auc:.4f}")

    # Permutation test
    n_perm = 10000
    perm_aucs = []
    for _ in range(n_perm):
        yp = np.random.permutation(y_bin)
        if len(np.unique(yp)) >= 2:
            perm_aucs.append(roc_auc_score(yp, x_tb))
    perm_aucs = np.array(perm_aucs)
    p_perm = np.mean(perm_aucs >= auc)
    print(f"  Permutation p = {p_perm:.4f} (n_perm={n_perm})")
    print(f"  Permutation null 95% range: [{np.percentile(perm_aucs, 2.5):.4f}, {np.percentile(perm_aucs, 97.5):.4f}]")

    # LOCO AUC
    loco_aucs = []
    for i in range(len(tb_idx)):
        keep = [j for j in range(len(tb_idx)) if j != i]
        x_loo = x_tb[keep]
        y_loo = y_tb[keep]
        median_loo = np.median(y_loo)
        y_bin_loo = (y_loo >= median_loo).astype(int)
        if len(np.unique(y_bin_loo)) >= 2:
            loco_aucs.append(roc_auc_score(y_bin_loo, x_loo))
    loco_aucs = np.array(loco_aucs)
    print(f"  LOCO AUC: mean={np.mean(loco_aucs):.4f}, SD={np.std(loco_aucs):.4f}, range=[{np.min(loco_aucs):.4f}, {np.max(loco_aucs):.4f}]")
else:
    print("  Cannot compute AUC: binary labels degenerate")


# ── Experiment #5: TeacherBehavior vs Headline Fisher z-test ──────────────
print(f"\n{'='*70}")
print("EXPERIMENT #5: TeacherBehavior vs Headline Fisher z-test")
print(f"{'='*70}")

# Use the canonical class-level estimates (6-MLLM means) from
# unified_results.json: headline ρ=0.769 (n=13), TeacherBehavior ρ=0.595 (n=8).
with open(RESULTS / "01_core" / "correlation" / "unified_results.json") as f:
    _unified = json.load(f)
_canon_cls = _unified["scb5_class_level"]
_canon_tb = _unified["teacher_behavior_only"]
rho_h = _canon_cls["spearman_rho"]
p_h = _canon_cls["spearman_p"]
rho_t = _canon_tb["rho"]
p_t = _canon_tb["p"]
n_h = _canon_cls["n"]
n_t = _canon_tb["n"]

print(f"  Headline (n={n_h}):         ρ={rho_h:.4f}, p={p_h:.6f}")
print(f"  TeacherBehavior (n={n_t}): ρ={rho_t:.4f}, p={p_t:.4f}")
print(f"  ρ² (headline) = {rho_h**2:.4f}, ρ² (TB) = {rho_t**2:.4f}")

z_val, p_val = fisher_z_test(rho_h, rho_t, n_h, n_t)
print(f"  Fisher z-test: z={z_val:.3f}, p={p_val:.4f}")
if p_val > 0.05:
    print(f"  → NOT significantly different (p>0.05)")
else:
    print(f"  → Significantly different (p<0.05)")


# ── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"""
Headline: ρ={rho_h:.3f} (n={n_h}, p={p_h:.4f})
TeacherBehavior-only: ρ={rho_t:.3f} (n={n_t}, p={p_t:.4f})

#2 Subset Meta: TB ρ={subset_data.get('TeacherBehavior (n=8)',{}).get('rho',0):.3f}, Pooled ρ={r_pooled:.3f} [I²={I2:.0f}%]
#3 Leave-2-Out: ρ∈[{np.min(rho_vals):.3f},{np.max(rho_vals):.3f}], SD={np.std(rho_vals):.3f}
#4 AUC: {auc:.3f} (perm p={p_perm:.4f}), LOCO [{np.min(loco_aucs):.3f},{np.max(loco_aucs):.3f}]
#5 Fisher z: z={z_val:.3f}, p={p_val:.4f}
""")

# Save
output = {
    "headline": {"n": n_h, "rho": rho_h, "p": p_h},
    "teacher_only": {"n": n_t, "rho": rho_t, "p": p_t},
    "exp2_subset_meta": {k.replace(" ", "_").lower(): v for k, v in subset_data.items()},
    "exp2_pooled": {"rho": r_pooled, "ci_low": ci_low_p, "ci_high": ci_high_p, "i2": I2},
    "exp3_leave2out": {
        "min": float(np.min(rho_vals)), "max": float(np.max(rho_vals)),
        "mean": float(np.mean(rho_vals)), "sd": float(np.std(rho_vals)),
        "pct_below_05": float(np.mean(rho_vals < 0.50)),
        "all_positive": bool(np.all(rho_vals > 0)),
    },
    "exp4_auc": {
        "auc": float(auc),
        "p_perm": float(p_perm),
        "loco_mean": float(np.mean(loco_aucs)),
        "loco_range": [float(np.min(loco_aucs)), float(np.max(loco_aucs))],
    },
    "exp5_fisher": {"z": z_val, "p": p_val},
    "per_class": [{"dataset": r["dataset"], "class": r["class"],
                   "anchor_score": r["anchor_score"], "mllm_mean": r["mllm_mean"]} for r in rows],
}

out_path = RESULTS / "02_robustness" / "robustness" / "robustness_analysis.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved to {out_path}")
