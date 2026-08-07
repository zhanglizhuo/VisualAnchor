#!/usr/bin/env python3
"""
Merge PathMNIST/BloodMNIST per-class AnchorScore (class-index format) into
the cross_domain_anchor_per_class.json (class-name format) for a unified,
human-readable archive.

Also regenerates llava15_7b_summary.json from the per-dataset JSON files.

Idempotent: safe to re-run.
"""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"
CDM = RESULTS / "02_robustness" / "cross_domain_mllm"
ASC = RESULTS / "02_robustness" / "anchor_score_cross_domain" / "anchor_scores.json"
ACD = RESULTS / "02_robustness" / "anchor_score_cross_domain"

# Class name mappings (index -> name), matching MedMNIST convention
PATH_CLASSES = [
    "Adipose tissue", "Background", "Debris", "Lymphocytes", "Mucus",
    "Smooth muscle", "Normal colon mucosa", "Cancer-associated stroma",
    "Tumor epithelium",
]
BLOOD_CLASSES = [
    "Basophil", "Eosinophil", "Erythroblast", "Immature granulocyte",
    "Lymphocyte", "Monocyte", "Neutrophil", "Platelet",
]
NAME_MAP = {"PathMNIST": PATH_CLASSES, "BloodMNIST": BLOOD_CLASSES}


def merge_anchor():
    per_class_file = ACD / "cross_domain_anchor_per_class.json"
    with open(per_class_file) as f:
        merged = json.load(f)

    with open(ASC) as f:
        idx_data = json.load(f)

    for ds, classes in NAME_MAP.items():
        if ds not in idx_data:
            print(f"  WARN: {ds} missing from anchor_scores.json")
            continue
        per_class = idx_data[ds]["per_class_acc"]
        merged[ds] = {}
        for i, name in enumerate(classes):
            key = str(i)
            if key in per_class:
                merged[ds][name] = per_class[key]["acc"]
        print(f"  {ds}: overall={idx_data[ds]['overall_acc']:.2f}%, "
              f"{len(merged[ds])} classes merged")

    with open(per_class_file, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged -> {per_class_file}")


def regen_llava_summary():
    summary = {}
    for f in sorted(CDM.glob("llava15_7b_*.json")):
        if "summary" in f.name or "cross_domain" in f.name:
            continue
        with open(f) as fp:
            d = json.load(fp)
        tc = sum(v["correct"] for v in d.values())
        tt = sum(v["total"] for v in d.values())
        # dataset name from filename: llava15_7b_<dataset>_<YYYYMMDD>_<HHMMSS>.json
        parts = f.stem.replace("llava15_7b_", "").split("_")
        name = parts[0]  # dataset name is the first token
        summary[name] = {"correct": tc, "total": tt,
                         "acc": round(tc / tt * 100, 1)}
        print(f"  {name:15s} {tc}/{tt} = {tc/tt*100:.1f}%")

    out = CDM / "llava15_7b_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary -> {out}")


if __name__ == "__main__":
    print("=== Merging per-class AnchorScore ===")
    merge_anchor()
    print("\n=== Regenerating LLaVA summary ===")
    regen_llava_summary()
