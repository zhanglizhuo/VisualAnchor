#!/usr/bin/env python3
"""Fix Qwen2-VL-7B rows in mllm_full.json.

The stored Qwen2-VL-7B TeacherBehavior and HandriseReadWrite rows did not
match the raw inference outputs (phase1_annotations, index-only protocol,
full validation crops). Several values had been copied from other models
(teacher=54.89 == Qwen3.5-27B, blackboard-writing=98.56 == Qwen3.5-35B,
hand-raising=90.26/read=24.47/write=90.66 were LLaVA duplicate values).
The BowTurnHead row was already correct.

Recompute all three rows from the raw phase1 annotations and rewrite.
"""
import json
from collections import defaultdict
from pathlib import Path

PHASE1 = Path("/home/lizhuo/Work/Mayan/llm-annotation/results/phase1_annotations/full_20260418_0001")
TARGET = Path("/home/lizhuo/Work/Mayan/VisualAnchor/results/01_core/paper_data/mllm_full.json")

CLASS_KEYS = {
    "TeacherBehavior": {
        "guide": "guide", "answer": "answer", "on-stage interaction": "On-stage interaction",
        "blackboard-writing": "blackboard-writing", "teacher": "teacher", "stand": "stand",
        "screen": "screen", "blackboard": "blackBoard",
    },
    "HandriseReadWrite": {"hand-raise": "hand-raising", "read": "read", "write": "write"},
    "BowTurnHead": {"bow-head": "BowHead", "turn-head": "TurnHead"},
}


def per_class(path: Path) -> dict:
    rows = [json.loads(l) for l in open(path)]
    out = defaultdict(lambda: [0, 0])
    for r in rows:
        gt = r["gt_name"]
        out[gt][1] += 1
        if r["pred_qwen_name"] == gt:
            out[gt][0] += 1
    return {k: round(100.0 * c / n, 2) for k, (c, n) in out.items()}


def main() -> None:
    data = json.load(open(TARGET))
    for ds, mapping in CLASS_KEYS.items():
        raw = per_class(PHASE1 / f"{ds}_val_annotations.jsonl")
        row = {canon: raw[raw_key] for raw_key, canon in mapping.items()}
        data[ds]["Qwen2-VL-7B"] = row
        print(ds, row)
    data["_provenance"] = data["_provenance"] + " Qwen2-VL-7B recomputed from phase1_annotations full-val index-only raw outputs (2026-08-04)."
    with open(TARGET, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
