#!/usr/bin/env python3
"""
Compute pooled class-level correlation, LODO, and meta-analysis
using 4 MLLMs (Qwen2-VL-7B, Qwen2.5-VL-7B, LLaVA-1.5-7B, LLaVA-NeXT-Mistral-7B)
for cross-domain datasets and 6 MLLMs for SCB5.

Outputs:
  results/01_core/correlation/pooled_class_level_results.json  (updated, n=47)
  results/01_core/correlation/meta_analysis_results.json
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, chi2

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS = BASE / "results"


def fisher_z(r):
    return np.arctanh(r)


def fisher_z_inv(z):
    return np.tanh(z)


def meta_analysis(studies):
    """Random-effects meta-analysis returning dict of results."""
    k = len(studies)
    z_vals = np.array([fisher_z(s['r']) for s in studies])
    se_vals = np.array([1.0 / np.sqrt(s['n'] - 3) for s in studies])
    w_fixed = 1.0 / se_vals ** 2

    z_fixed = np.sum(w_fixed * z_vals) / np.sum(w_fixed)
    se_fixed = np.sqrt(1.0 / np.sum(w_fixed))
    r_fixed = fisher_z_inv(z_fixed)
    r_fixed_ci = (fisher_z_inv(z_fixed - 1.96 * se_fixed),
                  fisher_z_inv(z_fixed + 1.96 * se_fixed))

    Q = np.sum(w_fixed * (z_vals - z_fixed) ** 2)
    df = k - 1
    p_het = 1 - chi2.cdf(Q, df)
    I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0
    C = np.sum(w_fixed) - np.sum(w_fixed ** 2) / np.sum(w_fixed)
    tau2 = max(0, (Q - df) / C) if C > 0 else 0

    w_random = 1.0 / (se_vals ** 2 + tau2)
    z_random = np.sum(w_random * z_vals) / np.sum(w_random)
    se_random = np.sqrt(1.0 / np.sum(w_random))
    r_random = fisher_z_inv(z_random)
    r_random_ci = (fisher_z_inv(z_random - 1.96 * se_random),
                   fisher_z_inv(z_random + 1.96 * se_random))

    pred_sd = np.sqrt(se_random ** 2 + tau2)
    r_pred = (fisher_z_inv(z_random - 1.96 * pred_sd),
              fisher_z_inv(z_random + 1.96 * pred_sd))

    return {
        "n_studies": k,
        "studies": [{"name": s["name"], "r": s["r"], "n": s["n"]} for s in studies],
        "fixed_effect": {"rho": round(r_fixed, 3), "ci_95": [round(r_fixed_ci[0], 3), round(r_fixed_ci[1], 3)]},
        "heterogeneity": {"Q": round(Q, 2), "df": df, "p": round(p_het, 4), "I2": round(I2, 1), "tau2": round(tau2, 4)},
        "random_effects": {"rho": round(r_random, 3), "ci_95": [round(r_random_ci[0], 3), round(r_random_ci[1], 3)]},
        "prediction_interval": [round(r_pred[0], 3), round(r_pred[1], 3)],
    }


def main():
    # ================================================================
    # 1. SCB5 data (13 classes, 6 MLLMs)
    # ================================================================
    with open(RESULTS / "01_core" / "anchor_score_scb5" / "anchor_scores.json") as f:
        scb5_anchor = json.load(f)

    with open(RESULTS / "01_core" / "paper_data" / "mllm_full.json") as f:
        mllm_src = json.load(f)

    CORE_MLLMS = ["Qwen3.5-27B", "Qwen3.6-27B", "Qwen3.5-35B", "Qwen3.6-35B", "Gemma4-31B", "Gemma4-26B"]

    SCB5_MAP = {
        "TeacherBehavior": ["guide", "answer", "On-stage interaction", "blackboard-writing", "teacher", "stand", "screen", "blackBoard"],
        "HandriseReadWrite": ["hand-raising", "read", "write"],
        "BowTurnHead": ["BowHead", "TurnHead"],
    }

    scb5_data = []
    for ds_name, classes in SCB5_MAP.items():
        anchor_acc = scb5_anchor[ds_name]["per_class_acc"]
        for cname in classes:
            a_val = anchor_acc[cname]["acc"]
            mllm_vals = [mllm_src[ds_name][m][cname] for m in CORE_MLLMS
                         if m in mllm_src.get(ds_name, {}) and cname in mllm_src[ds_name][m]]
            mllm_mean = np.mean(mllm_vals) if mllm_vals else 0
            scb5_data.append((f"SCB5-{ds_name}", cname, a_val, mllm_mean))

    print(f"SCB5: {len(scb5_data)} classes")

    # ================================================================
    # 2. Cross-domain data (34 classes, 4 MLLMs)
    # ================================================================
    with open(RESULTS / "02_robustness" / "anchor_score_cross_domain" / "anchor_scores.json") as f:
        cross_anchor = json.load(f)

    MLLM_FILES = {
        "LLaVA-1.5-7B": RESULTS / "02_robustness" / "cross_domain_mllm" / "llava15b_results.json",
        "LLaVA-NeXT-Mistral-7B": RESULTS / "02_robustness" / "cross_domain_mllm" / "llavanext_results.json",
        "Qwen2.5-VL-7B": RESULTS / "02_robustness" / "cross_domain_mllm" / "qwen25vl7b_results.json",
        "Qwen2-VL-7B": RESULTS / "02_robustness" / "cross_domain_mllm" / "qwen2vl7b_results.json",
    }

    mllm_cross = {}
    for mname, mpath in MLLM_FILES.items():
        if mpath.exists():
            with open(mpath) as f:
                mllm_cross[mname] = json.load(f)

    # Class mapping: AnchorScore name → MLLM name
    CROSS_MAP = {
        "EuroSAT": {k: k for k in cross_anchor["EuroSAT"]["per_class_acc"]},
        "PathMNIST": {
            "adipose": "Adipose tissue",
            "background": "Background",
            "debris": "Debris",
            "lymphocytes": "Lymphocytes",
            "mucus": "Mucus",
            "smooth muscle": "Smooth muscle",
            "normal colon mucosa": "Normal colon mucosa",
            "cancer-associated stroma": "Cancer-associated stroma",
            "colorectal adenocarcinoma epithelium": "Tumor epithelium",
        },
        "BloodMNIST": {
            "basophil": "Basophil",
            "eosinophil": "Eosinophil",
            "erythroblast": "Erythroblast",
            "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)": "Immature granulocyte",
            "lymphocyte": "Lymphocyte",
            "monocyte": "Monocyte",
            "neutrophil": "Neutrophil",
            "platelet": "Platelet",
        },
        "TissueMNIST": {
            "Collecting Duct, Connecting Tubule": "Collecting Duct",
            "Distal Convoluted Tubule": "Distal Convoluted Tubule",
            "Glomerular endothelial cells": "Glomerular Endothelial Cells",
            "Interstitial endothelial cells": "Interstitial Endothelial Cells",
            "Leukocytes": "Leukocytes",
            "Podocytes": "Podocytes",
            "Proximal Tubule Segments": "Proximal Tubule Segments",
        },
    }

    cross_data = []
    for ds_name in ["EuroSAT", "PathMNIST", "BloodMNIST", "TissueMNIST"]:
        anchor_ds = cross_anchor[ds_name]["per_class_acc"]
        cmap = CROSS_MAP.get(ds_name, {})
        for a_cls, a_info in anchor_ds.items():
            m_cls = cmap.get(a_cls)
            if m_cls is None:
                continue  # skip classes with no MLLM match (e.g., "Thick Ascending Limb")
            mllm_vals = []
            for mname, mdata in mllm_cross.items():
                if ds_name not in mdata:
                    continue
                mllm_acc = mdata[ds_name].get("accuracy", {})
                if m_cls in mllm_acc:
                    mllm_vals.append(mllm_acc[m_cls])
            if not mllm_vals:
                continue
            mllm_mean = np.mean(mllm_vals)
            cross_data.append((ds_name, a_cls, a_info["acc"], mllm_mean))

    print(f"Cross-domain: {len(cross_data)} classes")
    for ds_name in ["EuroSAT", "PathMNIST", "BloodMNIST", "TissueMNIST"]:
        cnt = sum(1 for d in cross_data if d[0] == ds_name)
        print(f"  {ds_name}: {cnt}")

    # ================================================================
    # 3. Pooled class-level correlation
    # ================================================================
    all_data = scb5_data + cross_data
    print(f"\nTotal pooled: {len(all_data)} classes")

    all_anchor = [d[2] for d in all_data]
    all_mllm = [d[3] for d in all_data]

    rho, p = spearmanr(all_anchor, all_mllm)
    r, pr = pearsonr(all_anchor, all_mllm)
    print(f"\n=== Pooled class-level (n={len(all_data)}) ===")
    print(f"  Spearman rho={rho:.3f}, p={p:.6f}")
    print(f"  Pearson  r={r:.3f}, p={pr:.6f}")

    # ================================================================
    # 4. Per-domain breakdown
    # ================================================================
    domains = {}
    for d in all_data:
        domain = d[0]
        if domain not in domains:
            domains[domain] = []
        domains[domain].append((d[2], d[3]))

    per_domain = {}
    for domain, vals in sorted(domains.items()):
        a = [v[0] for v in vals]
        m = [v[1] for v in vals]
        rho_d, p_d = spearmanr(a, m)
        per_domain[domain] = {"n": len(vals), "spearman_rho": round(rho_d, 3),
                              "spearman_p": round(p_d, 4) if not np.isnan(p_d) else None}
        print(f"  {domain:35s} n={len(vals):2d}  rho={rho_d:.3f}  p={p_d:.4f}")

    # SCB5 only and cross-domain only
    scb5_a = [d[2] for d in scb5_data]
    scb5_m = [d[3] for d in scb5_data]
    rho_scb5, p_scb5 = spearmanr(scb5_a, scb5_m)
    cross_a = [d[2] for d in cross_data]
    cross_m = [d[3] for d in cross_data]
    rho_cross, p_cross = spearmanr(cross_a, cross_m)
    print(f"\n  SCB5 only:        n={len(scb5_data)}, rho={rho_scb5:.3f}, p={p_scb5:.4f}")
    print(f"  Cross-domain only: n={len(cross_data)}, rho={rho_cross:.3f}, p={p_cross:.4f}")

    # ================================================================
    # 5. Leave-one-dataset-out (7 domains: 3 SCB5 + 4 cross-domain)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  LEAVE-ONE-DATASET-OUT (7 domains, pooled {len(all_data)} classes)")
    print(f"{'='*70}")
    domains_list = sorted(domains.keys())
    lodo = {}
    for holdout in domains_list:
        lo_a, lo_m = [], []
        for d in all_data:
            if d[0] == holdout:
                continue
            lo_a.append(d[2])
            lo_m.append(d[3])
        if len(lo_a) >= 3:
            r_lodo, p_lodo = spearmanr(lo_a, lo_m)
            lodo[f"exclude_{holdout}"] = {"n": len(lo_a), "rho": round(r_lodo, 3), "p": round(p_lodo, 4)}
            print(f"  Exclude {holdout:35s}  n={len(lo_a):2d}  rho={r_lodo:.3f}  p={p_lodo:.4f}")
    lodo_rhos = [v["rho"] for v in lodo.values()]
    if lodo_rhos:
        print(f"\n  LODO range: [{min(lodo_rhos):.3f}, {max(lodo_rhos):.3f}]")
        print(f"  LODO mean:  {np.mean(lodo_rhos):.3f}")

    # ================================================================
    # 6. Meta-analysis (3 studies)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  META-ANALYSIS")
    print(f"{'='*70}")

    studies_data = [
        {"name": "SCB5", "r": rho_scb5, "n": len(scb5_data)},
        {"name": "SCB-LLM-202506", "r": 0.506, "n": 10},  # from ensemble_correlation.json
        {"name": "Cross-domain per-class", "r": rho_cross, "n": len(cross_data)},
    ]
    meta_result = meta_analysis(studies_data)

    print(f"  Individual studies:")
    for s in meta_result["studies"]:
        print(f"    {s['name']:25s}  r={s['r']:.3f}, n={s['n']:2d}")
    print(f"  Fixed-effect:    rho={meta_result['fixed_effect']['rho']}, CI={meta_result['fixed_effect']['ci_95']}")
    print(f"  Heterogeneity:   Q={meta_result['heterogeneity']['Q']}, I²={meta_result['heterogeneity']['I2']}%, p={meta_result['heterogeneity']['p']}")
    print(f"  Random-effects:  rho={meta_result['random_effects']['rho']}, CI={meta_result['random_effects']['ci_95']}")
    print(f"  Prediction int:  {meta_result['prediction_interval']}")

    # Classroom-only meta-analysis (SCB5 + SCB-LLM)
    classroom = [s for s in studies_data if s['name'] != 'Cross-domain per-class']
    meta_classroom = meta_analysis(classroom)
    print(f"\n  Classroom-only fixed-effect: rho={meta_classroom['fixed_effect']['rho']}, CI={meta_classroom['fixed_effect']['ci_95']}")

    # ================================================================
    # 7. Save results
    # ================================================================
    output_pooled = {
        "description": f"Pooled class-level analysis: SCB5 (13) + cross-domain ({len(cross_data)}) = {len(all_data)} classes. Cross-domain uses 4 MLLM avg. TissueMNIST drops Thick Ascending Limb (no MLLM match).",
        "computation_date": "2026-07-14",
        "n_mllms_cross_domain": 4,
        "cross_domain_mllms": list(MLLM_FILES.keys()),
        "pooled": {
            "n": len(all_data),
            "spearman_rho": round(rho, 3),
            "spearman_p": p,
            "pearson_r": round(r, 3),
            "pearson_p": pr,
        },
        "scb5_only": {"n": len(scb5_data), "spearman_rho": round(rho_scb5, 3), "spearman_p": p_scb5},
        "cross_domain_only": {"n": len(cross_data), "spearman_rho": round(rho_cross, 3), "spearman_p": p_cross},
        "per_domain": per_domain,
        "leave_one_dataset_out": lodo,
        "leave_one_dataset_out_range": [round(min(lodo_rhos), 3), round(max(lodo_rhos), 3)] if lodo_rhos else None,
        "leave_one_dataset_out_mean": round(np.mean(lodo_rhos), 3) if lodo_rhos else None,
        "data": [{"domain": d[0], "class": d[1], "anchor_score": d[2], "mllm_mean": round(d[3], 1)} for d in all_data],
    }

    out_dir = RESULTS / "01_core" / "correlation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pooled_class_level_results.json", "w") as f:
        json.dump(output_pooled, f, indent=2)
    print(f"\nSaved pooled results to {out_dir / 'pooled_class_level_results.json'}")

    output_meta = {
        "description": "Random-effects meta-analysis of AnchorScore-MLLM correlations across 3 independent studies.",
        "computation_date": "2026-07-14",
        "pooled_class_level_source": str(out_dir / "pooled_class_level_results.json"),
        "meta_analysis": meta_result,
        "classroom_only_fixed_effect": meta_classroom["fixed_effect"],
    }

    with open(out_dir / "meta_analysis_results.json", "w") as f:
        json.dump(output_meta, f, indent=2)
    print(f"Saved meta-analysis to {out_dir / 'meta_analysis_results.json'}")


if __name__ == "__main__":
    main()
