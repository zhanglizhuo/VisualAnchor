"""
realized_routing_manifest.py

Builds the realized-routing manifest for TeacherBehavior (tau=45) from the
canonical per-image CLIP prediction cache:
  - routed: images whose CLIP predicted class has AnchorScore < 45 (the
    exact subset the expected-value estimate sends to the MLLM branch)
  - calib: 30/class stratified sample of the full val split (seed 42) to
    quantify the serving-backend shift vs canonical per-class accuracies
  - kept: CLIP-accepted images (cached correctness is used directly)

true/clip_pred are indices into the canonical class list
[guide, answer, on-stage interaction, blackboard-writing, teacher, stand,
screen, blackboard] (llm-annotation cross_model_annotate.py order).
"""
import json
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJ / "results"
CLIP_CACHE = RESULTS / "01_core" / "clip_per_image" / "per_image_predictions.json"
ANCHOR = RESULTS / "01_core" / "anchor_score_scb5" / "anchor_scores.json"
OUT = RESULTS / "05_applications" / "hybrid" / "realized_routing_manifest.json"

CLASSES = ["guide", "answer", "on-stage interaction", "blackboard-writing",
           "teacher", "stand", "screen", "blackboard"]
NAME2IDX = {"guide": 0, "answer": 1, "On-stage interaction": 2, "blackboard-writing": 3,
            "teacher": 4, "stand": 5, "screen": 6, "blackBoard": 7, "blackboard": 7}


def main():
    pred = json.load(open(CLIP_CACHE))
    anchor = json.load(open(ANCHOR))
    tb_anchor = {k: v["acc"] for k, v in anchor["TeacherBehavior"]["per_class_acc"].items()}
    items = pred["TeacherBehavior"]["predictions"]
    N = len(items)

    routed = []
    for it in items:
        rec = {"f": it["path"].split("/")[-1], "true": NAME2IDX[it["true_class_name"]],
               "clip_pred": NAME2IDX[it["pred_class_name"]], "clip_correct": bool(it["correct"])}
        if tb_anchor[it["pred_class_name"]] < 45:
            routed.append(rec)

    rng = np.random.default_rng(42)
    by_cls = {}
    for it in items:
        by_cls.setdefault(NAME2IDX[it["true_class_name"]], []).append(it["path"].split("/")[-1])
    calib = []
    for c in sorted(by_cls):
        chosen = rng.permutation(len(by_cls[c]))[:30]
        calib += [{"f": by_cls[c][i], "true": c} for i in chosen]

    expected = json.load(open(RESULTS / "05_applications" / "hybrid" / "hybrid_deployable.json"))
    expected_value_acc = round(expected["TeacherBehavior"]["deployable"]["mean"]["45"]["acc"], 2)
    manifest = {
        "description": "Realized-routing manifest for TeacherBehavior (tau=45). routed = images whose CLIP predicted class has AnchorScore < 45; calib = 30/class stratified full-val sample (seed 42). true/clip_pred are indices into the canonical class list.",
        "classes": CLASSES,
        "n_val": N, "n_routed": len(routed), "n_calib": len(calib),
        "expected_value_acc": expected_value_acc,
        "routed": routed, "calib": calib,
    }
    json.dump(manifest, open(OUT, "w"))
    print(f"Saved {OUT}: routed={len(routed)} calib={len(calib)} (overlap files in both arms is expected; aggregate by arm)")


if __name__ == "__main__":
    main()
