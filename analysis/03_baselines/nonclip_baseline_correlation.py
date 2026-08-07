#!/usr/bin/env python3
"""
nonclip_baseline_correlation.py

Compare DINOv2 kNN accuracy vs CLIP AnchorScore as predictors of
per-class MLLM accuracy on SCB5 (tests whether CLIP is special or any
classifier works).

Usage:
  python nonclip_baseline_correlation.py
"""

import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

CLASS_MAP = {
    "TeacherBehavior": [
        "guide", "answer", "On-stage interaction",
        "blackboard-writing", "teacher", "stand", "screen", "blackBoard",
    ],
    "HandriseReadWrite": ["hand-raising", "read", "write"],
    "BowTurnHead": ["BowHead", "TurnHead"],
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def correlate(x, y):
    if len(x) < 3:
        return None
    sp = spearmanr(x, y)
    return {"rho": round(sp.statistic, 4), "p": round(sp.pvalue, 4), "n": len(x)}


def main():
    clip_anchor = load_json(RESULTS_DIR / "01_core/anchor_score_scb5" / "anchor_scores.json")
    dinov2_path = RESULTS_DIR / "01_core/anchor_score_scb5" / "dinov2_anchor.json"

    if not dinov2_path.exists():
        print("ERROR: dinov2_anchor.json not found. Run experiments/dinov2_anchor_scb5.py first.")
        return

    dinov2_anchor = load_json(dinov2_path)

    with open(RESULTS_DIR / "01_core" / "paper_data" / "mllm_raw.json") as f:
        raw = load_json(str(RESULTS_DIR / "01_core" / "paper_data" / "mllm_raw.json"))

    mean_mllm = {}
    for ds_name, models in raw.items():
        if ds_name.startswith("_"):
            continue
        for model, classes in models.items():
            if model.startswith("_"):
                continue
            for cls, acc in classes.items():
                mean_mllm.setdefault(ds_name, {}).setdefault(cls, []).append(acc)
    for ds_name in mean_mllm:
        mean_mllm[ds_name] = {cls: float(np.mean(accs)) for cls, accs in mean_mllm[ds_name].items()}

    clip_vals, dinov2_vals, mllm_vals = [], [], []
    per_ds = {}

    for ds_name, classes in CLASS_MAP.items():
        ds_clip, ds_dino, ds_mllm = [], [], []
        for cls in classes:
            ca = clip_anchor.get(ds_name, {}).get("per_class_acc", {}).get(cls, {}).get("acc")
            da = dinov2_anchor.get(ds_name, {}).get("per_class_acc", {}).get(cls, {}).get("acc")
            ma = mean_mllm.get(ds_name, {}).get(cls)
            if ca is not None and da is not None and ma is not None:
                clip_vals.append(ca)
                dinov2_vals.append(da)
                mllm_vals.append(ma)
                ds_clip.append(ca)
                ds_dino.append(da)
                ds_mllm.append(ma)

        if len(ds_clip) >= 3:
            per_ds[ds_name] = {
                "clip_vs_mllm": correlate(ds_clip, ds_mllm),
                "dinov2_vs_mllm": correlate(ds_dino, ds_mllm),
                "n_classes": len(ds_clip),
            }

    print(f"{'='*70}")
    print(f"  NON-CLIP BASELINE: DINOv2 vs CLIP AnchorScore")
    print(f"{'='*70}")
    print(f"\n  {'Dataset':22s}  {'n':>3s}  {'rho(CLIP)':>12s}  {'rho(DINOv2)':>12s}  {'Delta':>8s}")
    for ds_name, r in per_ds.items():
        c = r["clip_vs_mllm"]
        d = r["dinov2_vs_mllm"]
        if c and d:
            delta = c["rho"] - d["rho"]
            print(f"  {ds_name:22s}  {r['n_classes']:>3d}  "
                  f"{c['rho']:>+10.4f}  {d['rho']:>+10.4f}  {delta:>+8.4f}")

    print(f"\n  CLASS-LEVEL (n={len(clip_vals)} classes):")
    cl_clip = correlate(clip_vals, mllm_vals)
    cl_dino = correlate(dinov2_vals, mllm_vals)
    if cl_clip:
        print(f"    rho(CLIP AnchorScore, MLLM acc) = {cl_clip['rho']:+.4f}  p={cl_clip['p']:.4f}")
    if cl_dino:
        print(f"    rho(DINOv2 kNN, MLLM acc)       = {cl_dino['rho']:+.4f}  p={cl_dino['p']:.4f}")

    if cl_clip and cl_dino:
        delta = cl_clip["rho"] - cl_dino["rho"]
        print(f"\n    Delta (CLIP advantage): {delta:+.4f}")
        if delta > 0.1:
            print(f"    => CLIP's vision-language alignment adds unique predictive signal")
        elif delta > 0:
            print(f"    => CLIP is slightly better, but DINOv2 also captures task difficulty")
        else:
            print(f"    => DINOv2 is comparable or better — correlation may reflect generic task difficulty")

    clip_dino_corr = correlate(clip_vals, dinov2_vals)
    if clip_dino_corr:
        print(f"\n    rho(CLIP, DINOv2) = {clip_dino_corr['rho']:+.4f}  "
              f"(how similar are their per-class rankings?)")

    print(f"\n  Per-class detail:")
    print(f"  {'Class':28s}  {'CLIP':>8s}  {'DINOv2':>8s}  {'MLLM':>8s}")
    for ds_name, classes in CLASS_MAP.items():
        for cls in classes:
            ca = clip_anchor.get(ds_name, {}).get("per_class_acc", {}).get(cls, {}).get("acc")
            da = dinov2_anchor.get(ds_name, {}).get("per_class_acc", {}).get(cls, {}).get("acc")
            ma = mean_mllm.get(ds_name, {}).get(cls)
            if ca is not None and da is not None and ma is not None:
                print(f"  {cls:28s}  {ca:>7.1f}%  {da:>7.1f}%  {ma:>7.1f}%")

    out = {
        "per_dataset": per_ds,
        "class_level": {
            "clip": cl_clip,
            "dinov2": cl_dino,
            "clip_vs_dinov2": clip_dino_corr,
        },
    }
    out_path = RESULTS_DIR / "03_baselines" / "baselines" / "nonclip_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
