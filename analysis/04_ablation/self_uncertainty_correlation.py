#!/usr/bin/env python3
"""
self_uncertainty_correlation.py

Compare MLLM self-uncertainty vs AnchorScore as predictors of per-class
MLLM accuracy on SCB5.

Loads per-class confidence from results/04_ablation/self_uncertainty/ and compares
rho(confidence, accuracy) against rho(AnchorScore, accuracy).

Usage:
  python self_uncertainty_correlation.py
"""

import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
SU_DIR = RESULTS_DIR / "04_ablation" / "self_uncertainty"
ANCHOR_FILE = RESULTS_DIR / "01_core/anchor_score_scb5" / "anchor_scores.json"

CLASS_MAP = {
    "TeacherBehavior": [
        "guide", "answer", "On-stage interaction",
        "blackboard-writing", "teacher", "stand", "screen", "blackBoard",
    ],
    "HandriseReadWrite": ["hand-raising", "read", "write"],
    "BowTurnHead": ["BowHead", "TurnHead"],
}

MODEL_NAME_MAP = {
    "ollama_qwen3_5_27b": "Qwen3.5-27B",
    "ollama_qwen3_6_27b": "Qwen3.6-27B",
    "ollama_qwen3_6_35b-a3b": "Qwen3.6-35B",
    "ollama_gemma4_26b": "Gemma4-26B",
    "llava15_7b": "LLaVA-1.5-7B",
}


def load_self_uncertainty():
    """Load all self-uncertainty result files."""
    models = {}
    if not SU_DIR.exists():
        return models
    for f in sorted(SU_DIR.glob("*.json")):
        stem = f.stem
        if "correlation" in stem or "comparison" in stem:
            continue
        if stem.startswith("llava15_7b_2") or stem == "llava15_7b":
            name = "LLaVA-1.5-7B"
        else:
            name = MODEL_NAME_MAP.get(stem, stem)
        with open(f) as fh:
            data = json.load(fh)
        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            models[name] = data
    return models


def load_anchor_scores():
    with open(ANCHOR_FILE) as f:
        return json.load(f)


def load_mllm_accuracy():
    """Load per-class MLLM accuracy from mllm_raw.json."""
    with open(RESULTS_DIR / "01_core" / "paper_data" / "mllm_raw.json") as f:
        raw = json.load(f)
    mllm = {}
    for ds_name, models in raw.items():
        if ds_name.startswith("_"):
            continue
        for model, classes in models.items():
            if model.startswith("_"):
                continue
            mllm.setdefault(model, {})
            for cls, acc in classes.items():
                mllm[model].setdefault(cls, {})[ds_name] = acc
    return mllm


def correlate(x, y):
    if len(x) < 3:
        return None
    sp = spearmanr(x, y)
    rho = round(sp.statistic, 4) if not np.isnan(sp.statistic) else None
    p = round(sp.pvalue, 4) if not np.isnan(sp.pvalue) else None
    return {"rho": rho, "p": p, "n": len(x)}


def fmt_val(v):
    return f"{v:+.4f}" if v is not None else "N/A"


def fmt_p(v):
    return f"{v:.4f}" if v is not None else "N/A"


def main():
    su_models = load_self_uncertainty()
    anchor = load_anchor_scores()
    mllm_acc = load_mllm_accuracy()

    print(f"{'='*70}")
    print(f"  SELF-UNCERTAINTY vs ANCHORSCORE COMPARISON")
    print(f"{'='*70}")
    print(f"  Self-uncertainty models found: {list(su_models.keys())}")

    per_model_results = {}
    all_conf, all_acc, all_anchor = [], [], []

    for model_name, su_data in su_models.items():
        confs, accs, anchors = [], [], []
        for ds_name, classes in CLASS_MAP.items():
            if ds_name not in su_data:
                continue
            for cls in classes:
                if cls not in su_data[ds_name]:
                    continue
                su = su_data[ds_name][cls]
                confs.append(su["mean_confidence"])
                accs.append(su["acc"])
                a = anchor.get(ds_name, {}).get("per_class_acc", {}).get(cls, {}).get("acc", None)
                if a is not None:
                    anchors.append(a)

        if len(confs) < 3:
            continue

        conf_acc = correlate(confs, accs)
        anchor_acc = correlate(anchors, accs) if len(anchors) == len(accs) else None

        per_model_results[model_name] = {
            "confidence_vs_acc": conf_acc,
            "anchor_vs_acc": anchor_acc,
            "n_classes": len(confs),
        }

        print(f"\n  {model_name} (n={len(confs)} classes)")
        if conf_acc:
            print(f"    rho(confidence, acc)  = {fmt_val(conf_acc['rho'])}  p={fmt_p(conf_acc['p'])}")
        if anchor_acc:
            print(f"    rho(AnchorScore, acc) = {fmt_val(anchor_acc['rho'])}  p={fmt_p(anchor_acc['p'])}")

        all_conf.extend(confs)
        all_acc.extend(accs)
        if len(anchors) == len(accs):
            all_anchor.extend(anchors)

    print(f"\n{'='*70}")
    print(f"  POOLED (all models, all classes)")
    print(f"{'='*70}")
    pooled_conf = correlate(all_conf, all_acc)
    pooled_anchor = correlate(all_anchor, all_acc)
    if pooled_conf:
        print(f"  rho(confidence, acc)  = {fmt_val(pooled_conf['rho'])}  p={fmt_p(pooled_conf['p'])}  n={pooled_conf['n']}")
    if pooled_anchor:
        print(f"  rho(AnchorScore, acc) = {fmt_val(pooled_anchor['rho'])}  p={fmt_p(pooled_anchor['p'])}  n={pooled_anchor['n']}")

    class_level_conf, class_level_acc, class_level_anchor = [], [], []
    for ds_name, classes in CLASS_MAP.items():
        for cls in classes:
            confs_c, accs_c = [], []
            for model_name, su_data in su_models.items():
                if ds_name in su_data and cls in su_data[ds_name]:
                    su = su_data[ds_name][cls]
                    confs_c.append(su["mean_confidence"])
                    accs_c.append(su["acc"])
            if confs_c:
                class_level_conf.append(float(np.mean(confs_c)))
                class_level_acc.append(float(np.mean(accs_c)))
                a = anchor.get(ds_name, {}).get("per_class_acc", {}).get(cls, {}).get("acc")
                if a is not None:
                    class_level_anchor.append(a)

    print(f"\n{'='*70}")
    print(f"  CLASS-LEVEL (n={len(class_level_conf)} classes, averaged across models)")
    print(f"{'='*70}")
    cl_conf = correlate(class_level_conf, class_level_acc)
    cl_anchor = correlate(class_level_anchor, class_level_acc)
    if cl_conf:
        print(f"  rho(mean_confidence, mean_acc)  = {fmt_val(cl_conf['rho'])}  p={fmt_p(cl_conf['p'])}")
    if cl_anchor:
        print(f"  rho(AnchorScore, mean_acc)      = {fmt_val(cl_anchor['rho'])}  p={fmt_p(cl_anchor['p'])}")

    if cl_conf and cl_anchor:
        delta = (cl_anchor["rho"] - cl_conf["rho"]) if (cl_anchor["rho"] is not None and cl_conf["rho"] is not None) else None
        print(f"\n  Delta (AnchorScore advantage): {delta:+.4f}")
        if delta > 0.05:
            print(f"  => AnchorScore is meaningfully better than self-uncertainty")
        elif delta < -0.05:
            print(f"  => Self-uncertainty is meaningfully better than AnchorScore")
        else:
            print(f"  => Comparable predictive power")

    out = {
        "per_model": per_model_results,
        "pooled": {"confidence": pooled_conf, "anchor": pooled_anchor},
        "class_level": {"confidence": cl_conf, "anchor": cl_anchor},
    }
    out_path = SU_DIR / "correlation_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
