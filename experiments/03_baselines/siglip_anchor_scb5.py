"""
siglip_anchor_scb5.py

SigLIP baseline: compute zero-shot per-class accuracy on SCB5 using
SigLIP (ViT-SO400M-14-SigLIP-384) via open_clip.

Tests whether the AnchorScore-MLLM correlation is specific to CLIP's
contrastive loss or generalizes to other vision-language models
(SigLIP uses a sigmoid pairwise loss).

Output format matches anchor_scores.json for direct comparison.

First run requires internet to download weights. For offline use:
  1. On a machine with internet, run once: this caches to ~/.cache/huggingface/hub/
  2. Copy the cached directory to the offline machine
  Or use HF_ENDPOINT=https://hf-mirror.com for faster downloads in China.

Usage:
  python siglip_anchor_scb5.py [--batch_size 32] [--data-root /path/to/scb5]
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


def load_siglip(device):
    import open_clip
    model, _, transform = open_clip.create_model_and_transforms(
        "ViT-SO400M-14-SigLIP-384", pretrained="webli"
    )
    tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")
    model = model.to(device).eval()
    logger.info("Loaded SigLIP ViT-SO400M-14 (webli)")
    return model, tokenizer, transform


def build_prompts(class_names, templates):
    prompts = []
    for cls in class_names:
        friendly = CLASS_NAME_MAP.get(cls, cls)
        for t in templates:
            prompts.append(t.format(friendly))
    return prompts


@torch.no_grad()
def evaluate(model, tokenizer, transform, device, samples, class_names,
             batch_size=64):
    prompts = build_prompts(class_names, PROMPT_TEMPLATES)
    n_classes = len(class_names)
    n_templates = len(PROMPT_TEMPLATES)
    n_prompts = len(prompts)

    # Pre-compute ALL text features once
    text_tokens = tokenizer(prompts).to(device)
    all_text_feats = model.encode_text(text_tokens)
    all_text_feats = all_text_feats / all_text_feats.norm(dim=-1, keepdim=True)

    class_correct = [0] * n_classes
    class_total = [0] * n_classes

    n_total = len(samples)
    log_interval = max(batch_size, n_total // 20)
    for i in range(0, len(samples), batch_size):
        if (log_interval // batch_size == 0) or ((i // batch_size) % (log_interval // batch_size) == 0):
            logger.info(f"  processed {i}/{n_total} images")
        batch = samples[i: i + batch_size]
        images = torch.stack([
            transform(Image.open(p).convert("RGB")) for p, _ in batch
        ]).to(device)
        batch_labels = [c for _, c in batch]

        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # (batch_size, d)

        # Compute all similarities at once and reshape
        # sim: (batch_size, n_prompts)
        sim = image_features @ all_text_feats.T
        # Reshape to (batch_size, n_classes, n_templates)
        sim = sim.view(-1, n_classes, n_templates)
        # Average over templates: (batch_size, n_classes)
        avg_sim = sim.mean(dim=2)
        best_classes = avg_sim.argmax(dim=1)  # (batch_size,)

        for bi in range(len(batch)):
            actual_label = batch_labels[bi]
            predicted = int(best_classes[bi].item())
            class_total[actual_label] += 1
            if predicted == actual_label:
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
    parser.add_argument("--batch_size", type=int, default=64)
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

    model, tokenizer, transform = load_siglip(device)

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
            model, tokenizer, transform, device,
            val_samples, classes, args.batch_size,
        )

        tot_correct = sum(v["n"] * v["acc"] / 100 for v in per_class.values())
        tot_count = sum(v["n"] for v in per_class.values())
        overall = round(tot_correct / tot_count * 100, 2) if tot_count else 0.0

        for cls_name, info in per_class.items():
            logger.info(f"    {cls_name:28s}  n={info['n']:5d}  {info['acc']:6.2f}%")

        logger.info(f"  Overall SigLIP: {overall}% ({int(tot_correct)}/{tot_count})")
        results[ds_name] = {
            "overall_acc": overall,
            "per_class_acc": per_class,
            "total": int(tot_count),
            "correct": int(tot_correct),
            "model": "ViT-SO400M-14-SigLIP-384",
        }

    out_path = out_dir / "siglip_anchor.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")

    print(f"\n{'='*60}")
    print(f"  SIGLIP ANCHORSCORE")
    print(f"{'='*60}")
    print(f"  {'Dataset':22s}  {'Overall':>8s}")
    for ds_name, r in results.items():
        print(f"  {ds_name:22s}  {r['overall_acc']:>7.2f}%")


if __name__ == "__main__":
    main()
