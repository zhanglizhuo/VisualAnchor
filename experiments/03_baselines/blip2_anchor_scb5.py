"""
blip2_anchor_scb5.py

BLIP-2 baseline: compute zero-shot per-class accuracy on SCB5 using
BLIP-2 (Salesforce/blip2-opt-2.7b) via transformers.

Uses BLIP-2's image-text matching via the Q-Former:
  For each (image, class), we compute the cross-entropy loss of
  generating the class description as a continuation, and pick the
  class with the lowest loss (highest likelihood).

This tests whether the AnchorScore-MLLM correlation is specific
to CLIP's dual-encoder architecture or generalizes to other
vision-language models with fused encoders.

WARNING: BLIP-2 requires ~3.5GB of weights and is slow because it
evaluates each (image, class) pair individually. Expect ~1-2 hours
on a V100 for 5416 images x 13 classes.

Usage:
  python blip2_anchor_scb5.py [--batch_size 1] [--data-root /path/to/scb5]
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from PIL import Image

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import CLASS_NAME_MAP, read_label

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent.parent
SERVER_DATA = Path(os.environ.get("SCB5_DATA_ROOT", str(PROJ / "data" / "scb5")))

DATASET_CFG = {
    "TeacherBehavior": {
        "dir": "SCB5_TeacherBehavior",
        "subdir": "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2",
        "classes": [
            "guide", "answer", "On-stage interaction", "blackboard-writing",
            "teacher", "stand", "screen", "blackBoard",
        ],
    },
    "HandriseReadWrite": {
        "dir": "SCB5_HandriseReadWrite",
        "subdir": "SCB5-Handrise-Read-write-2024-9-17",
        "classes": ["hand-raising", "read", "write"],
    },
    "BowTurnHead": {
        "dir": "SCB_BowTurnHead",
        "subdir": None,
        "classes": ["BowHead", "TurnHead"],
    },
}

PROMPT_TEMPLATES = [
    "a photo of a person {} in classroom.",
    "a classroom scene showing {}.",
    "the action of {} in a school environment.",
]


def load_blip2(device):
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    model_name = "Salesforce/blip2-opt-2.7b"
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16,
    )
    model = model.to(device).eval()
    logger.info(f"Loaded BLIP-2 {model_name}")
    return model, processor


def build_class_prompts(class_names, templates):
    """Build {(class_name, template_idx): prompt_string} for loss evaluation."""
    prompts = {}
    for cls in class_names:
        friendly = CLASS_NAME_MAP.get(cls, cls)
        for t_idx, t in enumerate(templates):
            prompt = t.format(friendly)
            prompts[(cls, t_idx)] = prompt
    return prompts


@torch.no_grad()
def evaluate(model, processor, device, samples, class_names, batch_size=8):
    from transformers import BatchFeature
    n_classes = len(class_names)
    n_templates = len(PROMPT_TEMPLATES)

    class_prompts = build_class_prompts(class_names, PROMPT_TEMPLATES)

    class_correct = [0] * n_classes
    class_total = [0] * n_classes

    n_images = len(samples)
    logger.info(f"  Evaluating {n_images} images x {n_classes} classes x {n_templates} prompts")

    for img_idx, (img_path, actual_label) in enumerate(samples):
        if (img_idx + 1) % 50 == 0:
            logger.info(f"    [{img_idx + 1}/{n_images}]")

        raw_image = Image.open(img_path).convert("RGB")

        best_cls = -1
        best_loss = float("inf")

        for cid in range(n_classes):
            cls_name = class_names[cid]
            avg_loss = 0.0
            for t_idx in range(n_templates):
                prompt = class_prompts[(cls_name, t_idx)]
                inputs = processor(
                    images=raw_image,
                    text=prompt,
                    return_tensors="pt",
                ).to(device)

                labels = inputs["input_ids"].clone()
                labels[labels == processor.tokenizer.pad_token_id] = -100

                outputs = model(
                    pixel_values=inputs["pixel_values"].to(torch.float16),
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    labels=labels,
                )
                avg_loss += outputs.loss.item()
            avg_loss /= n_templates

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_cls = cid

        class_total[actual_label] += 1
        if best_cls == actual_label:
            class_correct[actual_label] += 1

    return {
        cls_name: {
            "n": class_total[cid],
            "acc": round(100.0 * class_correct[cid] / class_total[cid], 2)
            if class_total[cid] else 0.0,
        }
        for cid, cls_name in enumerate(class_names)
    }


def collect_samples(cfg, split):
    base = SERVER_DATA / cfg["dir"]
    sub = cfg["subdir"]
    img_dir = (base / sub / "images" / split if sub else
               base / "images" / split)
    lbl_dir = (base / sub / "labels" / split if sub else
               base / "labels" / split)
    samples = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        cid = read_label(lbl_path)
        if cid is None or cid >= len(cfg["classes"]):
            continue
        samples.append((str(img_path), cid))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    global SERVER_DATA
    if args.data_root:
        SERVER_DATA = Path(args.data_root)

    out_dir = Path(args.out or str(PROJ / "results" / "01_core/anchor_score_scb5"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, processor = load_blip2(device)

    results = {}
    for ds_name, cfg in DATASET_CFG.items():
        logger.info(f"\n{'='*60}\n{ds_name}")
        classes = cfg["classes"]

        val_samples = collect_samples(cfg, "val")
        logger.info(f"  val samples: {len(val_samples)}")

        if not val_samples:
            logger.warning(f"  Skipping {ds_name}: no data")
            continue

        per_class = evaluate(
            model, processor, device,
            val_samples, classes, args.batch_size,
        )

        tot_correct = sum(v["n"] * v["acc"] / 100 for v in per_class.values())
        tot_count = sum(v["n"] for v in per_class.values())
        overall = round(tot_correct / tot_count * 100, 2) if tot_count else 0.0

        for cls_name, info in per_class.items():
            logger.info(f"    {cls_name:28s}  n={info['n']:5d}  {info['acc']:6.2f}%")

        logger.info(f"  Overall BLIP-2: {overall}% ({int(tot_correct)}/{tot_count})")
        results[ds_name] = {
            "overall_acc": overall,
            "per_class_acc": per_class,
            "total": int(tot_count),
            "correct": int(tot_correct),
            "model": "blip2-opt-2.7b",
        }

    out_path = out_dir / "blip2_anchor.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")

    print(f"\n{'='*60}")
    print(f"  BLIP-2 ANCHORSCORE")
    print(f"{'='*60}")
    print(f"  {'Dataset':22s}  {'Overall':>8s}")
    for ds_name, r in results.items():
        print(f"  {ds_name:22s}  {r['overall_acc']:>7.2f}%")


if __name__ == "__main__":
    main()
