"""
eurosat_qwen35_aggregate.py

Aggregates the M5(b) EuroSAT 27B scale-check shard outputs into
results/02_robustness/cross_domain_mllm/eurosat_qwen35_results.json.

The shards (500 lines, 50/class x 10 classes) were produced on the V100 by
eurosat_qwen35_v100.py under the same protocol as the 7B cross-domain runs
(cross_domain_llava.py / cross_domain_qwen.py): 50 images/class, full-frame,
category-name reply, seed-42 per-class sampling.

Reports per-class accuracy, mean, and the within-dataset Spearman vs the
canonical EuroSAT AnchorScore, alongside the per-model 7B reference values.
"""
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from scipy.stats import spearmanr

PROJ = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJ / "results"
RAW = RESULTS / "02_robustness" / "cross_domain_mllm" / "eurosat_qwen35_raw.jsonl"
ANCHOR = RESULTS / "02_robustness" / "anchor_score_cross_domain" / "anchor_scores.json"
OUT = RESULTS / "02_robustness" / "cross_domain_mllm" / "eurosat_qwen35_results.json"

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
           "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

SEVEN_B_FILES = {
    "Qwen2-VL-7B": "qwen2vl7b_results.json",
    "Qwen2.5-VL-7B": "qwen25vl7b_results.json",
    "LLaVA-1.5-7B": "llava15b_results.json",
    "LLaVA-NeXT-7B": "llavanext_results.json",
}


def main():
    anchor = json.load(open(ANCHOR))
    anch = {k: v["acc"] for k, v in anchor["EuroSAT"]["per_class_acc"].items()}

    recs = [json.loads(ln) for ln in open(RAW)]
    assert len(recs) == 500
    by_cls = defaultdict(lambda: [0, 0])
    for r in recs:
        by_cls[r["true"]][1] += 1
        if r["correct"]:
            by_cls[r["true"]][0] += 1
    per_class = {
        cls: {"correct": by_cls[i][0], "total": by_cls[i][1],
              "acc": round(100 * by_cls[i][0] / by_cls[i][1], 1)}
        for i, cls in enumerate(CLASSES)
    }
    mean_acc = round(sum(v["acc"] for v in per_class.values()) / 10, 1)

    xs, ys = [], []
    for cls in CLASSES:
        xs.append(anch[cls])
        ys.append(per_class[cls]["acc"])
    r, p = spearmanr(xs, ys)

    sev7 = {}
    sev7_means = []
    for name, fn in SEVEN_B_FILES.items():
        d = json.load(open(RESULTS / "02_robustness" / "cross_domain_mllm" / fn))
        euro = d["EuroSAT"]
        vals = euro["accuracy"] if "accuracy" in euro else euro["per_class"]
        acc = {k: (v["acc"] if isinstance(v, dict) else v) for k, v in vals.items()}
        rr, _ = spearmanr([anch[c] for c in CLASSES], [acc[c] for c in CLASSES])
        sev7[name] = round(float(rr), 3)
        sev7_means.append(sum(acc[c] for c in CLASSES) / 10)

    out = {
        "description": "M5(b) scale-check: canonical 27B model (Qwen3.5-27B, Ollama q4) on EuroSAT under the same protocol as the 7B cross-domain runs (50 imgs/class, full-frame, category-name reply, seed-42 sampling). Between-domain ordering is robust to annotator scale (mean 41.8% within the 7B range 32.4-48.2%, still the highest domain); within-dataset signal is noisy at n=10 across all scales.",
        "computation_date": str(date.today()),
        "model": "Qwen3.5-27B (Ollama q4)",
        "protocol": "50 images/class, full-frame, category-name reply",
        "mean_accuracy_pct": mean_acc,
        "per_class": per_class,
        "spearman_rho_vs_anchor": round(float(r), 3),
        "spearman_p": round(float(p), 3),
        "n_classes": 10,
        "seven_b_rho_reference": sev7,
        "seven_b_mean_range": [round(min(sev7_means), 1), round(max(sev7_means), 1)],
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"Saved {OUT}: mean={mean_acc} rho={r:.3f} p={p:.3f}")


if __name__ == "__main__":
    main()
