import json, os
import numpy as np
from scipy.stats import spearmanr

ROOT = "/home/lizhuo/Work/Mayan/VisualAnchor"
OUT = os.path.join(ROOT, "results", "06_consensus_control")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)
B = 10000


def load_anchor():
    d = json.load(open(f"{ROOT}/results/01_core/anchor_score_scb5/anchor_scores.json"))
    out = {}
    for ds in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
        out.update({k: v["acc"] for k, v in d[ds]["per_class_acc"].items()})
    return out


def load_mllm_mean():
    d = json.load(open(f"{ROOT}/results/01_core/paper_data/mllm_raw.json"))
    agg, cnt = {}, {}
    for ds in d:
        if isinstance(d[ds], str):
            continue
        for m in d[ds]:
            for cls, acc in d[ds][m].items():
                agg.setdefault(cls, 0.0)
                cnt.setdefault(cls, 0)
                agg[cls] += acc
                cnt[cls] += 1
    return {c: agg[c] / cnt[c] for c in agg}


def per_class_from_backbone():
    bb = json.load(open(f"{ROOT}/results/02_robustness/multi_backbone/backbone_results.json"))
    out = {}
    for key, label in [("openai_l14", "OpenAI-L/14"), ("openai_b32", "OpenAI-B/32")]:
        per = {}
        for ds in bb[key]["results"]:
            for c, rec in bb[key]["results"][ds]["per_class_acc"].items():
                per[c] = rec["acc"]
        out[label] = per
    return out


def per_class_from_01_core(fname):
    d = json.load(open(f"{ROOT}/results/01_core/anchor_score_scb5/{fname}"))
    return {k: v for k, v in d["per_class_vl"].items()}


def rho(x, y):
    return float(spearmanr(x, y)[0])


def main():
    anchors = load_anchor()
    mllm = load_mllm_mean()
    classes = sorted(set(anchors.keys()) & set(mllm.keys()))
    x_laion = np.array([anchors[c] for c in classes])
    y_m = np.array([mllm[c] for c in classes])

    predictors = {"AnchorScore (LAION-L/14)": x_laion}
    for label, per in per_class_from_backbone().items():
        predictors[label] = np.array([per[c] for c in classes])
    for label, fname in [("SigLIP", "siglip_correlation.json"),
                          ("DINOv2", "dinov2_correlation.json"),
                          ("BLIP2", "blip2_correlation.json")]:
        per = per_class_from_01_core(fname)
        predictors[label] = np.array([per[c] for c in classes])
    rn = json.load(open(f"{ROOT}/results/03_baselines/resnet50_baseline/resnet50_results.json"))
    predictors["ResNet-50"] = np.array([rn["per_class_accuracy"][c] * 100.0 for c in classes])

    rows = []
    laion_rho = rho(x_laion, y_m)
    for name, xx in predictors.items():
        if name.startswith("AnchorScore"):
            r = laion_rho
            boots = []
            for _ in range(B):
                idx = RNG.choice(len(classes), size=len(classes), replace=True)
                if len(set(x_laion[idx])) < 3 or len(set(y_m[idx])) < 3:
                    continue
                boots.append(rho(x_laion[idx], y_m[idx]))
            ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            rows.append({"predictor": name, "rho": round(r, 4), "ci95_rho": [round(ci[0], 4), round(ci[1], 4)],
                         "delta_vs_laion": 0.0, "ci95_delta": [0.0, 0.0], "significant_5pct": True})
            continue
        r = rho(xx, y_m)
        boots = []
        for _ in range(B):
            idx = RNG.choice(len(classes), size=len(classes), replace=True)
            if len(set(xx[idx])) < 3 or len(set(x_laion[idx])) < 3 or len(set(y_m[idx])) < 3:
                continue
            boots.append(rho(x_laion[idx], y_m[idx]) - rho(xx[idx], y_m[idx]))
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        rows.append({"predictor": name, "rho": round(r, 4),
                     "delta_vs_laion": round(laion_rho - r, 4),
                     "ci95_delta": ci, "significant_5pct": not (ci[0] <= 0 <= ci[1])})

    result = {
        "description": "Exp1: delta-rho tests vs AnchorScore (LAION-L/14). Coupled class-level bootstrap B=10000, n=13 classes, target = 6-MLLM mean per-class accuracy.",
        "n_classes": len(classes),
        "laion_rho": round(laion_rho, 4),
        "rows": rows,
    }
    out_path = os.path.join(OUT, "exp1_delta_rho.json")
    json.dump(result, open(out_path, "w"), indent=2)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()