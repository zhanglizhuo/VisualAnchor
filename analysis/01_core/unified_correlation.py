#!/usr/bin/env python3
"""
unified_correlation.py

Single source of truth for all correlation analyses between AnchorScore
(CLIP zero-shot accuracy) and MLLM annotation accuracy.

Produces:
  1. SCB5 pooled (6 MLLMs × 13 classes = 78 points)
  2. Class-level SCB5 (13 classes, mean across MLLMs)
  3. Pooled + cross-domain (47 classes; 6-MLLM mean on SCB5,
     4-MLLM mean on cross-domain)
  4. Leave-one-dataset-out (LODO) on 7 domains
  5. Leave-one-class-out (LOCO) on 13 SCB5 classes
  6. Leave-one-MLLM-out (LOMO) on SCB5
  7. Bootstrap 95% CI for class-level ρ
  8. Per-class MLLM mean / std / n
  9. Per-dataset breakdown
 10. Cross-domain AnchorScore summary
"""

import json
import random
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"
OUT_DIR = RESULTS / "01_core" / "correlation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Canonical MLLMs (6 Ollama models producing ρ=0.693, n=78) ──
CANONICAL_6 = [
    "Qwen3.5-27B",
    "Qwen3.6-27B",
    "Qwen3.5-35B",
    "Qwen3.6-35B",
    "Gemma4-31B",
    "Gemma4-26B",
]

CLASS_MAP = {
    "TeacherBehavior": [
        "guide", "answer", "On-stage interaction", "blackboard-writing",
        "teacher", "stand", "screen", "blackBoard",
    ],
    "HandriseReadWrite": ["hand-raising", "read", "write"],
    "BowTurnHead": ["BowHead", "TurnHead"],
}

SCB5_DATASETS = ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]


def load_mllm():
    with open(RESULTS / "01_core" / "paper_data" / "mllm_full.json") as f:
        return json.load(f)


def load_anchor_scb5():
    with open(RESULTS / "01_core" / "anchor_score_scb5" / "anchor_scores.json") as f:
        return json.load(f)


def load_anchor_cross():
    with open(RESULTS / "02_robustness" / "anchor_score_cross_domain" / "anchor_scores.json") as f:
        return json.load(f)


def spearman_with_nan(x, y):
    """Spearman correlation, returns (rho, p) or (nan, nan) if <3 points."""
    if len(x) < 3:
        return np.nan, np.nan
    r, p = spearmanr(x, y)
    return r, p


def bootstrap_ci(x, y, n_bootstrap=10000, ci=95):
    """Bootstrap 95% CI for Spearman ρ."""
    np.random.seed(42)
    random.seed(42)
    n = len(x)
    x_arr = np.array(x)
    y_arr = np.array(y)
    rhos = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, n)
        if len(np.unique(x_arr[idx])) < 2 or len(np.unique(y_arr[idx])) < 2:
            continue
        r, _ = spearmanr(x_arr[idx], y_arr[idx])
        rhos.append(r)
    rhos = sorted(rhos)
    alpha = (100 - ci) / 2
    lo = rhos[int(len(rhos) * alpha / 100)]
    hi = rhos[int(len(rhos) * (100 - alpha) / 100)]
    return round(lo, 3), round(hi, 3)


def get_scb5_points(mllm_src, models):
    """Build list of (dataset, class, anchor_score, mllm_acc) for SCB5."""
    anchor = load_anchor_scb5()
    points = []
    for ds_name in SCB5_DATASETS:
        per_class = anchor[ds_name]["per_class_acc"]
        for cname in CLASS_MAP[ds_name]:
            a_val = per_class[cname]["acc"]
            for mname in models:
                if mname not in mllm_src.get(ds_name, {}):
                    continue
                mdata = mllm_src[ds_name][mname]
                if cname not in mdata:
                    continue
                m_val = mdata[cname]
                points.append({
                    "dataset": ds_name,
                    "class": cname,
                    "anchor_score": a_val,
                    "mllm_acc": m_val,
                    "mllm_model": mname,
                })
    return points


def get_scb5_class_means(mllm_src, models):
    """Build list of (dataset, class, anchor_score, mllm_mean) for SCB5 class-level."""
    anchor = load_anchor_scb5()
    points = []
    for ds_name in SCB5_DATASETS:
        per_class = anchor[ds_name]["per_class_acc"]
        for cname in CLASS_MAP[ds_name]:
            a_val = per_class[cname]["acc"]
            mllm_vals = []
            for mname in models:
                if mname not in mllm_src.get(ds_name, {}):
                    continue
                mdata = mllm_src[ds_name][mname]
                if cname in mdata:
                    mllm_vals.append(mdata[cname])
            if not mllm_vals:
                continue
            mllm_mean = np.mean(mllm_vals)
            mllm_std = np.std(mllm_vals, ddof=1) if len(mllm_vals) > 1 else 0
            points.append({
                "dataset": ds_name,
                "class": cname,
                "anchor_score": a_val,
                "mllm_mean": round(mllm_mean, 2),
                "mllm_std": round(mllm_std, 2),
                "n_mllms": len(mllm_vals),
                "mllm_values": {m: mllm_src[ds_name].get(m, {}).get(cname, None)
                               for m in models if m in mllm_src.get(ds_name, {})},
            })
    return points


def get_cross_domain_points(mllm_src, models):
    """Build list of (dataset, class, anchor_score, mllm_mean) for cross-domain.

    Four MLLMs have cross-domain data: Qwen2-VL-7B, LLaVA-1.5-7B,
    LLaVA-NeXT-7B, Qwen2.5-VL-7B. mllm_mean is the mean across these.
    TissueMNIST "Collecting Duct" and "Connecting Tubule" share one
    AnchorScore entry ("Collecting Duct, Connecting Tubule"), so they
    form a single merged point (34 cross-domain classes total).
    AnchorScores read from anchor_scores.json with explicit key mappings.
    """
    cross_anchor = load_anchor_cross()

    # Canonical class names → JSON key mappings
    EUROSAT_CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
                       "Industrial", "Pasture", "PermanentCrop", "Residential",
                       "River", "SeaLake"]
    PATH_CLASSES = ["Adipose tissue", "Background", "Debris", "Lymphocytes",
                    "Mucus", "Smooth muscle", "Normal colon mucosa",
                    "Cancer-associated stroma", "Tumor epithelium"]
    BLOOD_CLASSES = ["Basophil", "Eosinophil", "Erythroblast",
                     "Immature granulocyte", "Lymphocyte", "Monocyte",
                     "Neutrophil", "Platelet"]
    TISSUE_CLASSES = ["Collecting Duct",
                      "Distal Convoluted Tubule", "Glomerular Endothelial Cells",
                      "Interstitial Endothelial Cells", "Leukocytes",
                      "Podocytes", "Proximal Tubule Segments"]

    # PathMNIST JSON keys differ from canonical names
    PATHMNIST_KEY_MAP = {
        "Adipose tissue": "adipose",
        "Background": "background",
        "Debris": "debris",
        "Lymphocytes": "lymphocytes",
        "Mucus": "mucus",
        "Smooth muscle": "smooth muscle",
        "Normal colon mucosa": "normal colon mucosa",
        "Cancer-associated stroma": "cancer-associated stroma",
        "Tumor epithelium": "colorectal adenocarcinoma epithelium",
    }
    BLOODMNIST_KEY_MAP = {
        "Basophil": "basophil",
        "Eosinophil": "eosinophil",
        "Erythroblast": "erythroblast",
        "Immature granulocyte": "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
        "Lymphocyte": "lymphocyte",
        "Monocyte": "monocyte",
        "Neutrophil": "neutrophil",
        "Platelet": "platelet",
    }
    TISSUEMNIST_KEY_MAP = {
        "Collecting Duct": "Collecting Duct, Connecting Tubule",
        "Distal Convoluted Tubule": "Distal Convoluted Tubule",
        "Glomerular Endothelial Cells": "Glomerular endothelial cells",
        "Interstitial Endothelial Cells": "Interstitial endothelial cells",
        "Leukocytes": "Leukocytes",
        "Podocytes": "Podocytes",
        "Proximal Tubule Segments": "Proximal Tubule Segments",
    }

    KEY_MAP = {
        "PathMNIST": PATHMNIST_KEY_MAP,
        "BloodMNIST": BLOODMNIST_KEY_MAP,
        "TissueMNIST": TISSUEMNIST_KEY_MAP,
    }

    DATASET_CLASSES = {
        "EuroSAT": EUROSAT_CLASSES,
        "PathMNIST": PATH_CLASSES,
        "BloodMNIST": BLOOD_CLASSES,
        "TissueMNIST": TISSUE_CLASSES,
    }

    # Cross-domain MLLM files: Qwen2-VL-7B, LLaVA-1.5-7B,
    # LLaVA-NeXT-7B, Qwen2.5-VL-7B (all store {dataset: {accuracy: {class: %}}})
    mllm_files = [
        "qwen2vl7b_results.json",
        "llava15b_results.json",
        "llavanext_results.json",
        "qwen25vl7b_results.json",
    ]
    mllm_data = {}
    for fname in mllm_files:
        fpath = RESULTS / "02_robustness" / "cross_domain_mllm" / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            mllm_data[fname] = json.load(f)

    def get_cross_anchor(ds_name, class_name):
        """Get AnchorScore from anchor_scores.json using key mapping."""
        if ds_name not in cross_anchor:
            return None
        pc = cross_anchor[ds_name]["per_class_acc"]
        key_map = KEY_MAP.get(ds_name, {})
        json_key = key_map.get(class_name, class_name)
        if json_key in pc:
            return pc[json_key]["acc"]
        return None

    points = []
    for ds_name, classes in DATASET_CLASSES.items():
        for cname in classes:
            a_val = get_cross_anchor(ds_name, cname)
            if a_val is None:
                continue
            mllm_vals = []
            for fname, mdata in mllm_data.items():
                acc = mdata.get(ds_name, {}).get("accuracy", {})
                if cname in acc:
                    mllm_vals.append(acc[cname])
            if not mllm_vals:
                continue
            mllm_mean = np.mean(mllm_vals)
            mllm_std = np.std(mllm_vals, ddof=1) if len(mllm_vals) > 1 else 0
            points.append({
                "dataset": ds_name,
                "class": cname,
                "anchor_score": a_val,
                "mllm_mean": round(mllm_mean, 2),
                "mllm_std": round(mllm_std, 2),
                "n_mllms": len(mllm_vals),
            })
    return points


def compute_correlation(label, points, prefix="  "):
    """Compute Spearman + Pearson for a list of point dicts."""
    if not points:
        print(f"{prefix}[SKIP] {label}: no data")
        return None
    a = [p["anchor_score"] for p in points]
    m = [p["mllm_acc"] if "mllm_acc" in p else p["mllm_mean"] for p in points]
    r_s, p_s = spearmanr(a, m)
    r_p, p_p = pearsonr(a, m)
    ci_lo, ci_hi = bootstrap_ci(a, m)
    print(f"{prefix}{label:40s} n={len(a):3d}  ρ={r_s:.4f}  p={p_s:.2e}  "
          f"r={r_p:.4f}  95%CI=[{ci_lo:.3f},{ci_hi:.3f}]")
    return {
        "n": len(a),
        "spearman_rho": round(r_s, 4),
        "spearman_p": round(p_s, 6),
        "pearson_r": round(r_p, 4),
        "pearson_p": round(p_p, 6),
        "bootstrap_ci_95": [ci_lo, ci_hi],
    }


def run_leave_one_dataset_out(all_class_points):
    """LODO on the pooled 47-class points."""
    domains = sorted(set(p["dataset"] for p in all_class_points))
    print(f"\n  LODO on {len(domains)} domains: {domains}")
    results = {}
    for holdout in domains:
        subset = [p for p in all_class_points if p["dataset"] != holdout]
        if len(subset) < 3:
            continue
        a = [p["anchor_score"] for p in subset]
        m = [p["mllm_mean"] for p in subset]
        r, p_val = spearmanr(a, m)
        results[f"exclude_{holdout}"] = {
            "n": len(subset),
            "rho": round(r, 3),
            "p": round(p_val, 4),
        }
        print(f"    Exclude {holdout:35s}  n={len(subset):2d}  ρ={r:.3f}  p={p_val:.4f}")
    rhos = [v["rho"] for v in results.values()]
    print(f"    LODO range: [{min(rhos):.3f}, {max(rhos):.3f}], mean={np.mean(rhos):.3f}")
    return results


def run_leave_one_class_out(scb5_class_points):
    """LOCO on SCB5 13 classes (leave one class out, recompute ρ)."""
    classes = sorted(set(p["class"] for p in scb5_class_points))
    print(f"\n  LOCO on {len(classes)} SCB5 classes")
    results = {}
    for holdout in classes:
        subset = [p for p in scb5_class_points if p["class"] != holdout]
        a = [p["anchor_score"] for p in subset]
        m = [p["mllm_mean"] for p in subset]
        r, p_val = spearmanr(a, m)
        results[f"exclude_{holdout}"] = {
            "n": len(subset),
            "rho": round(r, 3),
            "p": round(p_val, 4),
        }
        print(f"    Exclude {holdout:30s}  n={len(subset):2d}  ρ={r:.3f}  p={p_val:.4f}")
    rhos = [v["rho"] for v in results.values()]
    print(f"    LOCO range: [{min(rhos):.3f}, {max(rhos):.3f}], mean={np.mean(rhos):.3f}")
    return results


def run_leave_one_mllm_out_class_level(mllm_src, models):
    """LOMO on SCB5 class-level ρ (leave one MLLM out from the mean)."""
    print(f"\n  LOMO class-level on {len(models)} MLLMs")
    anchor = load_anchor_scb5()
    results = {}
    for holdout in models:
        subset = [m for m in models if m != holdout]
        a, m = [], []
        for ds_name in SCB5_DATASETS:
            per_class = anchor[ds_name]["per_class_acc"]
            for cname in CLASS_MAP[ds_name]:
                mllm_vals = [mllm_src[ds_name][mdl][cname] for mdl in subset
                             if mdl in mllm_src.get(ds_name, {}) and cname in mllm_src[ds_name][mdl]]
                if mllm_vals:
                    a.append(per_class[cname]["acc"])
                    m.append(np.mean(mllm_vals))
        r, p_val = spearmanr(a, m)
        results[f"exclude_{holdout}"] = {"n": len(a), "rho": round(r, 3), "p": round(p_val, 4)}
        print(f"    Exclude {holdout:20s}  n={len(a):2d}  ρ={r:.3f}  p={p_val:.4f}")
    rhos = [v["rho"] for v in results.values()]
    print(f"    LOMO class-level range: [{min(rhos):.3f}, {max(rhos):.3f}], mean={np.mean(rhos):.3f}")
    return results


def extended_model_set_analysis(mllm_src, base_models):
    """Class-level ρ with additional MLLMs added to the base set."""
    print(f"\n  EXTENDED MODEL SET ANALYSIS (base={len(base_models)} MLLMs)")
    anchor = load_anchor_scb5()
    extra_sets = {
        "base_plus_llava": base_models + ["LLaVA-1.5-7B"],
        "base_plus_qwen2vl": base_models + ["Qwen2-VL-7B"],
        "base_plus_llava_qwen2vl": base_models + ["LLaVA-1.5-7B", "Qwen2-VL-7B"],
        "base_only": base_models,
    }
    results = {}
    for name, models in extra_sets.items():
        a, m = [], []
        for ds_name in SCB5_DATASETS:
            per_class = anchor[ds_name]["per_class_acc"]
            for cname in CLASS_MAP[ds_name]:
                mllm_vals = [mllm_src[ds_name][mdl][cname] for mdl in models
                             if mdl in mllm_src.get(ds_name, {}) and cname in mllm_src[ds_name][mdl]]
                if mllm_vals:
                    a.append(per_class[cname]["acc"])
                    m.append(np.mean(mllm_vals))
        r, p_val = spearmanr(a, m)
        results[name] = {"n": len(a), "n_mllms": len(models), "rho": round(r, 3), "p": round(p_val, 4)}
        print(f"    {name:30s}  n_mllms={len(models):2d}  n_classes={len(a):2d}  ρ={r:.3f}  p={p_val:.4f}")
    return results


def teacher_behavior_only_class_level(mllm_src, models):
    """Class-level ρ for TeacherBehavior alone (8 classes)."""
    anchor = load_anchor_scb5()
    a, m = [], []
    for cname in CLASS_MAP["TeacherBehavior"]:
        mllm_vals = [mllm_src["TeacherBehavior"][mdl][cname] for mdl in models
                     if mdl in mllm_src.get("TeacherBehavior", {}) and cname in mllm_src["TeacherBehavior"][mdl]]
        if mllm_vals:
            a.append(anchor["TeacherBehavior"]["per_class_acc"][cname]["acc"])
            m.append(np.mean(mllm_vals))
    r, p_val = spearmanr(a, m)
    result = {"n": len(a), "rho": round(r, 3), "p": round(p_val, 4)}
    print(f"\n  TeacherBehavior alone: n_classes={len(a)}  ρ={r:.3f}  p={p_val:.4f}")
    return result


def run_leave_one_mllm_out(scb5_points):
    """LOMO on SCB5 pooled 78-point data (leave one MLLM out)."""
    models = sorted(set(p["mllm_model"] for p in scb5_points))
    print(f"\n  LOMO on {len(models)} MLLMs")
    results = {}
    for holdout in models:
        subset = [p for p in scb5_points if p["mllm_model"] != holdout]
        a = [p["anchor_score"] for p in subset]
        m = [p["mllm_acc"] for p in subset]
        r, p_val = spearmanr(a, m)
        results[f"exclude_{holdout}"] = {
            "n": len(subset),
            "rho": round(r, 3),
            "p": round(p_val, 4),
        }
        print(f"    Exclude {holdout:20s}  n={len(subset):2d}  ρ={r:.3f}  p={p_val:.4f}")
    rhos = [v["rho"] for v in results.values()]
    print(f"    LOMO range: [{min(rhos):.3f}, {max(rhos):.3f}], mean={np.mean(rhos):.3f}")
    return results


def per_model_breakdown(scb5_points):
    """Spearman ρ per MLLM."""
    models = sorted(set(p["mllm_model"] for p in scb5_points))
    results = {}
    for mname in models:
        subset = [p for p in scb5_points if p["mllm_model"] == mname]
        a = [p["anchor_score"] for p in subset]
        m = [p["mllm_acc"] for p in subset]
        r, p_val = spearmanr(a, m)
        results[mname] = {
            "n": len(subset),
            "spearman_rho": round(r, 3),
            "spearman_p": round(p_val, 4),
        }
    return results


def per_dataset_breakdown(scb5_points):
    """Spearman ρ per SCB5 dataset."""
    results = {}
    for ds_name in SCB5_DATASETS:
        subset = [p for p in scb5_points if p["dataset"] == ds_name]
        if not subset:
            continue
        a = [p["anchor_score"] for p in subset]
        m = [p["mllm_acc"] for p in subset]
        r, p_val = spearmanr(a, m)
        results[ds_name] = {
            "n": len(subset),
            "spearman_rho": round(r, 3),
            "spearman_p": round(p_val, 4),
        }
    return results


def load_multi_backbone():
    """Load existing multi-backbone results for reference."""
    path = RESULTS / "02_robustness" / "multi_backbone" / "backbone_correlation_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_cross_domain_summary():
    """Cross-domain AnchorScore summary."""
    anchor = load_anchor_scb5()
    cross = load_anchor_cross()

    CROSS_DOMAIN_STANDARD = {}
    for ds_name, ds_data in cross.items():
        n_cls = len(ds_data.get("per_class_acc", {}))
        CROSS_DOMAIN_STANDARD[f"{ds_name} ({n_cls}cls)"] = {
            "AnchorScore": ds_data.get("overall_acc", 0),
            "n": ds_data.get("total", 0),
        }

    scb5_total_correct = sum(anchor[d]["correct"] for d in SCB5_DATASETS)
    scb5_total = sum(anchor[d]["total"] for d in SCB5_DATASETS)
    CROSS_DOMAIN_STANDARD["SCB5 (weighted avg)"] = {
        "AnchorScore": round(scb5_total_correct / scb5_total * 100, 1),
        "n": scb5_total,
    }
    return CROSS_DOMAIN_STANDARD


def main():
    mllm_src = load_mllm()
    models = CANONICAL_6
    print(f"Canonical MLLMs ({len(models)}): {models}")
    print()

    # ── 1. SCB5 pooled (78 points: 6 MLLMs × 13 classes) ──
    scb5_points = get_scb5_points(mllm_src, models)
    print(f"SCB5 pooled points: {len(scb5_points)}")
    pooled = compute_correlation("SCB5 pooled (78pt)", scb5_points)

    # ── 2. SCB5 class-level (13 classes, mean across MLLMs) ──
    scb5_class_points = get_scb5_class_means(mllm_src, models)
    print(f"\nSCB5 class-level points: {len(scb5_class_points)}")
    class_level = compute_correlation("SCB5 class-level (13pt)", scb5_class_points)

    # ── 3. Cross-domain class-level ──
    cross_domain_points = get_cross_domain_points(mllm_src, models)
    print(f"\nCross-domain class-level points: {len(cross_domain_points)}")
    cross_result = compute_correlation("Cross-domain class-level", cross_domain_points)

    # ── 4. Pooled (SCB5 + cross-domain) ──
    all_class_points = scb5_class_points + cross_domain_points
    print(f"\nPooled class-level points: {len(all_class_points)}")
    pooled_class = compute_correlation("Pooled class-level (47pt)", all_class_points)

    # ── 5. LODO ──
    lodo = run_leave_one_dataset_out(all_class_points)

    # ── 6. LOCO ──
    loco = run_leave_one_class_out(scb5_class_points)

    # ── 7. LOMO (pooled) ──
    lomo_pooled = run_leave_one_mllm_out(scb5_points)

    # ── 8. LOMO (class-level) ──
    lomo_class = run_leave_one_mllm_out_class_level(mllm_src, models)

    # ── 9. Extended model set analysis ──
    extended_models = extended_model_set_analysis(mllm_src, models)

    # ── 10. TeacherBehavior-only class-level ──
    tb_only = teacher_behavior_only_class_level(mllm_src, models)

    # ── 11. Per-MLLM breakdown ──
    per_model = per_model_breakdown(scb5_points)

    # ── 12. Per-dataset breakdown ──
    per_dataset = per_dataset_breakdown(scb5_points)

    # ── 10. Per-class detail ──
    print(f"\n{'='*70}")
    print(f"  PER-CLASS DETAIL (sorted by AnchorScore)")
    print(f"{'='*70}")
    sorted_classes = sorted(scb5_class_points, key=lambda p: p["anchor_score"])
    print(f"  {'Class':28s} {'Dataset':18s} {'Anchor%':>8s} {'MLLM avg%':>10s} {'MLLM σ':>7s} {'n':>3s}")
    print(f"  {'-'*28} {'-'*18} {'-'*8} {'-'*10} {'-'*7} {'-'*3}")
    for p in sorted_classes:
        print(f"  {p['class']:28s} {p['dataset']:18s} {p['anchor_score']:>7.2f}% "
              f"{p['mllm_mean']:>9.2f}% {p['mllm_std']:>6.2f}% {p['n_mllms']:>3d}")

    # ── 11. Cross-domain summary ──
    cross_summary = load_cross_domain_summary()
    print(f"\n{'='*70}")
    print(f"  CROSS-DOMAIN ANCHORSCORE SUMMARY")
    print(f"{'='*70}")
    for ds_name, info in sorted(cross_summary.items()):
        print(f"  {ds_name:40s}  {info['AnchorScore']:5.1f}%  (n={info['n']})")

    # ── 12. Multi-backbone reference ──
    backbone_ref = load_multi_backbone()
    if backbone_ref:
        print(f"\n{'='*70}")
        print(f"  MULTI-BACKBONE REFERENCE (from backbone_correlation_results.json)")
        print(f"{'='*70}")
        for bk, bd in backbone_ref.items():
            print(f"  {bk:25s}  ρ={bd.get('spearman_rho', '?'):}")

    # ── Verify against paper's Table 1 ──
    print(f"\n{'='*70}")
    print(f"  PAPER TABLE 1 VERIFICATION (means across {len(models)} MLLMs)")
    print(f"{'='*70}")
    for p in sorted_classes:
        print(f"  {p['class']:28s}  Anchor={p['anchor_score']:>5.1f}%  "
              f"MLLM={p['mllm_mean']:>5.1f}%±{p['mllm_std']:.1f}")

    # ── Save ──
    output = {
        "meta": {
            "description": "Unified correlation analysis — single source of truth",
            "canonical_mllms": models,
            "n_mllms": len(models),
            "scb5_datasets": SCB5_DATASETS,
            "generated_by": "analysis/01_core/unified_correlation.py",
        },
        "scb5_pooled": pooled,
        "scb5_class_level": class_level,
        "cross_domain_class_level": cross_result,
        "pooled_class_level": pooled_class,
        "leave_one_dataset_out": lodo,
        "leave_one_class_out": loco,
        "leave_one_mllm_out_pooled": lomo_pooled,
        "leave_one_mllm_out_class_level": lomo_class,
        "extended_model_sets": extended_models,
        "teacher_behavior_only": tb_only,
        "per_model": per_model,
        "per_dataset": per_dataset,
        "per_class": sorted_classes,
        "cross_domain_summary": cross_summary,
    }
    out_path = OUT_DIR / "unified_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, allow_nan=False)
    print(f"\n{'='*70}")
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
