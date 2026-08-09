import json, os
import numpy as np

ROOT = "/home/lizhuo/Work/Mayan/VisualAnchor"
OUT = os.path.join(ROOT, "results", "06_consensus_control")
os.makedirs(OUT, exist_ok=True)


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


def main():
    anchors = load_anchor()
    mllm = load_mllm_mean()
    classes = sorted(set(anchors.keys()) & set(mllm.keys()))

    predictors = {"AnchorScore (LAION-L/14)": anchors}
    bb = json.load(open(f"{ROOT}/results/02_robustness/multi_backbone/backbone_results.json"))
    for key, label in [("openai_l14", "OpenAI-L/14"), ("openai_b32", "OpenAI-B/32")]:
        per = {}
        for ds in bb[key]["results"]:
            for c, rec in bb[key]["results"][ds]["per_class_acc"].items():
                per[c] = rec["acc"]
        predictors[label] = per
    for label, fname in [("SigLIP", "siglip_correlation.json"),
                          ("DINOv2", "dinov2_correlation.json"),
                          ("BLIP2", "blip2_correlation.json")]:
        d = json.load(open(f"{ROOT}/results/01_core/anchor_score_scb5/{fname}"))
        predictors[label] = d["per_class_vl"]
    rn = json.load(open(f"{ROOT}/results/03_baselines/resnet50_baseline/resnet50_results.json"))
    predictors["ResNet-50"] = rn["per_class_accuracy"]
    predictors["MLLM mean (target)"] = mllm

    floor = []
    pctg = {}
    for name, per in predictors.items():
        vals = np.array([per[c] for c in classes])
        floor.append({
            "predictor": name,
            "mean": round(float(vals.mean()), 2),
            "std": round(float(vals.std()), 2),
            "min": round(float(vals.min()), 2),
            "max": round(float(vals.max()), 2),
            "pct_classes_lt_15": round(float((vals < 15).mean() * 100), 1),
            "pct_classes_ge_85": round(float((vals >= 85).mean() * 100), 1),
            "range": round(float(vals.max() - vals.min()), 2),
        })
        pctg = None

    table = []
    for name, per in predictors.items():
        row = {"predictor": name}
        for c in classes:
            row[c] = round(per[c], 1)
        table.append(row)

    result = {
        "description": "Exp3: baseline per-class accuracy floor table (SCB5, 13 classes). Checks whether null correlations of weak baselines can be explained by variance compression.",
        "classes": classes,
        "summary": floor,
        "per_class_table": table,
    }
    out_path = os.path.join(OUT, "exp3_baseline_floor.json")
    json.dump(result, open(out_path, "w"), indent=2)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()