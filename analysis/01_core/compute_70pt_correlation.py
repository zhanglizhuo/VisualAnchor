#!/usr/bin/env python3
"""
Generate canonical result files for the VisualAnchor paper.
Reads MLLM per-class accuracy from results/01_core/paper_data/mllm_raw.json
and results/01_core/llava_scb5/ (LLaVA). Outputs:
  results/01_core/correlation/paper_70pt_results.json
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE / "results"
ANCHOR_FILE = RESULTS_DIR / "01_core/anchor_score_scb5" / "anchor_scores.json"
MLLM_RAW = RESULTS_DIR / "01_core" / "paper_data" / "mllm_raw.json"
OUT_FILE = RESULTS_DIR / "01_core" / "correlation" / "paper_70pt_results.json"

CLASS_MAP = {
    "TeacherBehavior": ["guide", "answer", "On-stage interaction", "blackboard-writing", "teacher", "stand", "screen", "blackBoard"],
    "HandriseReadWrite": ["hand-raising", "read", "write"],
    "BowTurnHead": ["BowHead", "TurnHead"],
}

# Read AnchorScore
with open(ANCHOR_FILE) as f:
    anchor = json.load(f)

# Read 6-model data from JSON (paper scope: exactly 6 MLLMs, 70 class-model pairs)
with open(MLLM_RAW) as f:
    raw = json.load(f)
MLLM_DATA = {ds: {k: v for k, v in models.items() if not k.startswith("_")}
             for ds, models in raw.items() if not ds.startswith("_")}

# ============================================================
# 1. Collect all (anchor, mllm) pairs
# ============================================================
all_anchor, all_mllm = [], []
per_dataset = {}
for ds_name in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
    anchor_acc = anchor[ds_name]["per_class_acc"]
    classes = CLASS_MAP[ds_name]
    ds_a, ds_m = [], []
    for cname in classes:
        a_val = anchor_acc[cname]["acc"]
        for mname, mdata in MLLM_DATA.get(ds_name, {}).items():
            if cname in mdata:
                ds_a.append(a_val)
                ds_m.append(mdata[cname])
                all_anchor.append(a_val)
                all_mllm.append(mdata[cname])
    per_dataset[ds_name] = {"anchor": ds_a, "mllm": ds_m, "n": len(ds_a)}

n_models = max(len(v) for v in MLLM_DATA.values())
print(f"Total pairs: {len(all_anchor)} ({n_models} MLLMs)")
for ds_name in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
    print(f"  {ds_name}: {per_dataset[ds_name]['n']}")

# ============================================================
# 2. Global correlation
# ============================================================
aa, mm = np.array(all_anchor), np.array(all_mllm)
sp = spearmanr(aa, mm)
pr = pearsonr(aa, mm)
global_corr = {
    "n": len(aa),
    "spearman_rho": round(sp.statistic, 4),
    "spearman_p": sp.pvalue,
    "pearson_r": round(pr.statistic, 4),
    "pearson_p": pr.pvalue,
    "n_models": n_models,
}
print(f"\nGlobal ({len(aa)} pts): ρ={global_corr['spearman_rho']:.4f}, p={global_corr['spearman_p']:.2e}")
print(f"  Pearson r={global_corr['pearson_r']:.4f}, p={global_corr['pearson_p']:.2e}")

# ============================================================
# 3. Per-dataset correlation
# ============================================================
per_ds_corr = {}
for ds_name in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
    ds_a = np.array(per_dataset[ds_name]["anchor"])
    ds_m = np.array(per_dataset[ds_name]["mllm"])
    sp_ds = spearmanr(ds_a, ds_m)
    per_ds_corr[ds_name] = {
        "n": len(ds_a),
        "n_classes": len(CLASS_MAP[ds_name]),
        "spearman_rho": round(sp_ds.statistic, 4),
        "spearman_p": sp_ds.pvalue,
    }
    print(f"{ds_name}: n={per_ds_corr[ds_name]['n']}, ρ={per_ds_corr[ds_name]['spearman_rho']:.4f}, p={per_ds_corr[ds_name]['spearman_p']:.2e}")

# ============================================================
# 4. Class-level correlation (average across MLLMs)
# ============================================================
class_anchor, class_mllm = [], []
for ds_name in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
    anchor_acc = anchor[ds_name]["per_class_acc"]
    classes = CLASS_MAP[ds_name]
    for cname in classes:
        mllm_vals = [mdata[cname] for mname, mdata in MLLM_DATA.get(ds_name, {}).items() if cname in mdata]
        if mllm_vals:
            class_anchor.append(anchor_acc[cname]["acc"])
            class_mllm.append(np.mean(mllm_vals))

class_sp = spearmanr(np.array(class_anchor), np.array(class_mllm))
class_level = {
    "n": len(class_anchor),
    "spearman_rho": round(class_sp.statistic, 4),
    "spearman_p": class_sp.pvalue,
}
print(f"\nClass-level ({len(class_anchor)} pts): ρ={class_level['spearman_rho']:.4f}, p={class_level['spearman_p']:.3f}")

# ============================================================
# 5. Clustered bootstrap (class-level resampling)
# ============================================================
class_data = []
for ds_name in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
    anchor_acc = anchor[ds_name]["per_class_acc"]
    classes = CLASS_MAP[ds_name]
    for cname in classes:
        mllm_vals = [mdata[cname] for mname, mdata in MLLM_DATA.get(ds_name, {}).items() if cname in mdata]
        if mllm_vals:
            class_data.append({
                "anchor": anchor_acc[cname]["acc"],
                "mllm_list": mllm_vals,
                "dataset": ds_name,
                "class": cname,
            })

B = 100000
rng = np.random.default_rng(42)
bootstraps = []
n_clusters = len(class_data)
for _ in range(B):
    idx = rng.integers(0, n_clusters, size=n_clusters)
    bs_anchor, bs_mllm = [], []
    for i in idx:
        cd = class_data[i]
        bs_anchor.append(cd["anchor"])
        bs_mllm.append(rng.choice(cd["mllm_list"]))
    r = spearmanr(np.array(bs_anchor), np.array(bs_mllm)).statistic
    bootstraps.append(r)

bootstraps = np.array(bootstraps)
ci_low = np.percentile(bootstraps, 2.5)
ci_high = np.percentile(bootstraps, 97.5)
pct_positive = np.mean(bootstraps > 0) * 100

bootstrap_result = {
    "B": B,
    "method": "class-level resampling, one MLLM per class per replicate",
    "rho_mean": round(float(bootstraps.mean()), 4),
    "rho_std": round(float(bootstraps.std()), 4),
    "ci_95": [round(ci_low, 4), round(ci_high, 4)],
    "pct_positive": round(pct_positive, 2),
}
print(f"\nBootstrap ({B} iters): mean ρ={bootstrap_result['rho_mean']:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}], {pct_positive:.1f}% positive")

# ============================================================
# 6. Ranking AUC
# ============================================================
median_mllm = np.median(class_mllm)
labels = np.array(class_mllm) < median_mllm
scores = -np.array(class_anchor)

n_pos = np.sum(labels)
n_neg = len(labels) - n_pos
auc = 0
for i in range(len(labels)):
    for j in range(len(labels)):
        if labels[i] and not labels[j] and scores[i] > scores[j]:
            auc += 1
        elif labels[i] and not labels[j] and scores[i] == scores[j]:
            auc += 0.5
auc /= (n_pos * n_neg) if n_pos * n_neg > 0 else 1.0

sorted_classes = sorted(zip(class_anchor, class_mllm), key=lambda x: x[0])
sorted_by_mllm = sorted(zip(class_anchor, class_mllm), key=lambda x: x[1])
for k in [3, 4, 5, 6]:
    anchor_bottom_k = set(c[0] for c in sorted_classes[:k])
    mllm_bottom_k = set(c[0] for c in sorted_by_mllm[:k])
    overlap = len(anchor_bottom_k & mllm_bottom_k)
    print(f"Top-{k} overlap: {overlap}/{k}")

ranking = {
    "auc": round(auc, 4),
    "threshold": "median MLLM accuracy",
    "n_classes": len(class_anchor),
}
print(f"\nRanking AUC: {ranking['auc']:.4f}")

# ============================================================
# 7. Leave-one-class-out
# ============================================================
loco_results = {}
for i in range(len(class_anchor)):
    loo_anchor = [class_anchor[j] for j in range(len(class_anchor)) if j != i]
    loo_mllm = [class_mllm[j] for j in range(len(class_mllm)) if j != i]
    r, p = spearmanr(np.array(loo_anchor), np.array(loo_mllm))
    loco_results[f"leave_out_{i}"] = {"rho": round(r, 4), "p": p}
loco_rhos = [v["rho"] for v in loco_results.values()]
loco = {"rho_range": [round(min(loco_rhos), 4), round(max(loco_rhos), 4)], "rho_mean": round(float(np.mean(loco_rhos)), 4)}
print(f"LOCO: ρ range [{loco['rho_range'][0]}, {loco['rho_range'][1]}], mean={loco['rho_mean']:.4f}")

# ============================================================
# 7.5 Leave-one-MLLM-out (point-level global)
# ============================================================
lomo_results = {}
all_models = sorted({m for ds in MLLM_DATA.values() for m in ds})
print(f"\nLOMO (leave-one-MLLM-out, point-level, {len(all_models)} models):")
for held_out in all_models:
    lo_a, lo_m = [], []
    for ds_name in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
        anchor_acc = anchor[ds_name]["per_class_acc"]
        for cname in CLASS_MAP[ds_name]:
            a_val = anchor_acc[cname]["acc"]
            for mname, mdata in MLLM_DATA.get(ds_name, {}).items():
                if mname == held_out:
                    continue
                if cname in mdata:
                    lo_a.append(a_val)
                    lo_m.append(mdata[cname])
    if len(lo_a) >= 3:
        r, p = spearmanr(np.array(lo_a), np.array(lo_m))
        lomo_results[f"exclude_{held_out}"] = {"n": len(lo_a), "rho": round(r, 4), "p": p}
        print(f"  Exclude {held_out:20s}  n={len(lo_a):2d}  ρ={r:.4f}  p={p:.3f}")
lomo_rhos = [v["rho"] for v in lomo_results.values()]
lomo = {"rho_range": [round(min(lomo_rhos), 4), round(max(lomo_rhos), 4)], "rho_mean": round(float(np.mean(lomo_rhos)), 4)}
print(f"  LOMO: ρ range [{lomo['rho_range'][0]}, {lomo['rho_range'][1]}], mean={lomo['rho_mean']:.4f}")

# ============================================================
# 8. Leave-one-dataset-out
# ============================================================
lodo_results = {}
for holdout in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
    lo_anchor, lo_mllm = [], []
    for ds_name in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
        if ds_name == holdout:
            continue
        anchor_acc = anchor[ds_name]["per_class_acc"]
        classes = CLASS_MAP[ds_name]
        for cname in classes:
            mllm_vals = [mdata[cname] for mname, mdata in MLLM_DATA.get(ds_name, {}).items() if cname in mdata]
            if mllm_vals:
                lo_anchor.append(anchor_acc[cname]["acc"])
                lo_mllm.append(np.mean(mllm_vals))
    if len(lo_anchor) >= 3:
        r, p = spearmanr(np.array(lo_anchor), np.array(lo_mllm))
        lodo_results[f"exclude_{holdout}"] = {"n": len(lo_anchor), "rho": round(r, 4), "p": p}
        print(f"LODO exclude {holdout}: n={len(lo_anchor)}, ρ={r:.4f}, p={p:.3f}")

# ============================================================
# 9. Per-class TeacherBehavior (Table 4)
# ============================================================
per_class_tb = []
anchor_acc_tb = anchor["TeacherBehavior"]["per_class_acc"]
tb_models = MLLM_DATA.get("TeacherBehavior", {})
for cname in CLASS_MAP["TeacherBehavior"]:
    vals = [tb_models[m][cname] for m in tb_models if cname in tb_models[m]]
    per_class_tb.append({
        "class": cname,
        "anchor_score": anchor_acc_tb[cname]["acc"],
        "mllm_mean": round(float(np.mean(vals)), 1),
        "mllm_std": round(float(np.std(vals)), 1),
        "delta": round(float(np.mean(vals) - anchor_acc_tb[cname]["acc"]), 1),
    })
    print(f"  {cname:28s} anchor={anchor_acc_tb[cname]['acc']:5.1f}  MLLM avg={np.mean(vals):5.1f}  std={np.std(vals):4.1f}  Δ={np.mean(vals)-anchor_acc_tb[cname]['acc']:+5.1f}")

# ============================================================
# 10. Save everything
# ============================================================
model_names = sorted({m for ds in MLLM_DATA.values() for m in ds})
output = {
    "description": f"VisualAnchor paper results ({n_models} MLLMs, {len(all_anchor)} class-model pairs). Models: {', '.join(model_names)}.",
    "data_points": f"{len(all_anchor)} (TB: {per_dataset['TeacherBehavior']['n']}, HRW: {per_dataset['HandriseReadWrite']['n']}, BTH: {per_dataset['BowTurnHead']['n']})",
    "global_correlation": global_corr,
    "per_dataset": per_ds_corr,
    "class_level": class_level,
    "bootstrap": bootstrap_result,
    "ranking": ranking,
    "leave_one_class_out": loco,
    "leave_one_mllm_out": lomo,
    "leave_one_mllm_out_per_model": lomo_results,
    "leave_one_dataset_out": lodo_results,
    "per_class_teacher_behavior": per_class_tb,
}
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x, allow_nan=False)
print(f"\nSaved to {OUT_FILE}")
