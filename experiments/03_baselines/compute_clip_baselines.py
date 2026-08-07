"""
compute_clip_baselines.py

Compute CLIP-derived baseline predictors for MLLM accuracy:
  1. Confidence (mean top-1 softmax probability)
  2. Margin (mean top1 - top2 softmax probability)
  3. Entropy (mean softmax entropy, inverted so high=confident)
  4. Feature dispersion (mean pairwise cosine distance among embeddings)

All metrics are class-level averages over SCB5 validation images, then
correlated (Spearman) with per-class MLLM mean accuracy (n=13).

Usage:
  python experiments/03_baselines/compute_clip_baselines.py
  python experiments/03_baselines/compute_clip_baselines.py --batch_size 128 --out results/03_baselines/clip_baselines.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent.parent

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


def load_mllm_accuracy(path=None):
    """Load per-class MLLM mean accuracy from pooled_class_level_results.json."""
    if path is None:
        path = PROJ / "results" / "01_core" / "correlation" / "pooled_class_level_results.json"
    with open(path) as f:
        data = json.load(f)
    mllm = {}
    for entry in data["data"]:
        if entry["domain"].startswith("SCB5"):
            cls_name = entry["class"]
            mllm[cls_name] = entry["mllm_mean"]
    return mllm


@torch.no_grad()
def compute_baselines(model, tokenizer, transform, device, batch_size, model_dtype):
    """Compute per-class CLIP confidence, margin, entropy, and feature dispersion.

    Returns dict mapping class_name -> {confidence, margin, entropy_inv,
    dispersion, n_images, embeddings}

    Also returns the full embedding matrix for downstream correlation.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import CLASS_NAME_MAP, read_label

    # Build class list from all datasets
    all_classes = []
    class_to_dataset = {}
    class_to_idx = {}
    for ds_name, cfg in DATASET_CFG.items():
        for cls in cfg["classes"]:
            if cls not in class_to_idx:
                class_to_idx[cls] = len(all_classes)
                all_classes.append(cls)
                class_to_dataset[cls] = ds_name
    num_classes = len(all_classes)
    logger.info(f"Total SCB5 classes: {num_classes}")

    # Build class-name prompts (3-template average, same as anchor_scb5.py)
    prompt_templates = [
        "a photo of a person {} in classroom.",
        "a classroom scene showing {}.",
        "the action of {} in a school environment.",
    ]
    prompt_texts = []
    for cls in all_classes:
        friendly = CLASS_NAME_MAP.get(cls, cls)
        for t in prompt_templates:
            prompt_texts.append(t.format(friendly))

    text_tokens = tokenizer(prompt_texts).to(device)
    text_feats = model.encode_text(text_tokens)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.view(num_classes, -1, text_feats.shape[-1]).mean(dim=1)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    # Collect all image samples
    all_samples = []  # (img_path, class_id)
    SERVER_DATA = Path(os.environ.get("SCB5_DATA_ROOT", str(PROJ / "data" / "scb5")))
    for ds_name, cfg in DATASET_CFG.items():
        base = SERVER_DATA / cfg["dir"]
        sub = cfg["subdir"]
        img_dir = (base / sub / "images" if sub else base / "images") / "val"
        lbl_dir = (base / sub / "labels" if sub else base / "labels") / "val"
        if not img_dir.exists():
            logger.warning(f"Skipping {ds_name}: {img_dir} not found")
            continue
        for fname in sorted(img_dir.iterdir()):
            if fname.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = lbl_dir / (fname.stem + ".txt")
            if not lbl_path.exists():
                continue
            cid = read_label(lbl_path)
            if cid is None or cid >= len(cfg["classes"]):
                continue
            cls_name = cfg["classes"][cid]
            all_samples.append((str(fname), class_to_idx[cls_name]))

    logger.info(f"Total validation images: {len(all_samples)}")

    # Per-class accumulators
    num_classes = len(all_classes)
    cls_confidence = [[] for _ in range(num_classes)]
    cls_margin = [[] for _ in range(num_classes)]
    cls_entropy = [[] for _ in range(num_classes)]
    cls_embeddings = [[] for _ in range(num_classes)]

    for i in range(0, len(all_samples), batch_size):
        batch = all_samples[i : i + batch_size]
        images = torch.stack([
            transform(Image.open(p).convert("RGB")) for p, _ in batch
        ]).to(device).to(model_dtype)
        labels = torch.tensor([c for _, c in batch], device=device)

        img_feats = model.encode_image(images)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        logits = (100.0 * img_feats) @ text_feats.T
        probs = torch.softmax(logits, dim=-1)

        # Per-sample metrics
        top_probs, top_idxs = probs.topk(2, dim=-1)
        confidence = top_probs[:, 0]  # top-1 probability
        margin = top_probs[:, 0] - top_probs[:, 1]  # top1 - top2
        entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=-1)

        for j in range(len(batch)):
            cid = labels[j].item()
            cls_confidence[cid].append(confidence[j].item())
            cls_margin[cid].append(margin[j].item())
            cls_entropy[cid].append(entropy[j].item())
            cls_embeddings[cid].append(img_feats[j].cpu().numpy())

        if (i // batch_size) % 20 == 0 and i > 0:
            logger.info(f"  Processed {i}/{len(all_samples)} images")

    # Aggregate per class
    per_class = {}
    for cid, cls_name in enumerate(all_classes):
        conf = np.array(cls_confidence[cid])
        marg = np.array(cls_margin[cid])
        entr = np.array(cls_entropy[cid])
        emb = np.stack(cls_embeddings[cid]) if cls_embeddings[cid] else np.zeros((0, 768))

        n = len(conf)
        # Feature dispersion = mean pairwise cosine distance
        # Cosine distance = 1 - cosine_similarity, range [0, 2]
        if n >= 2 and emb.shape[0] >= 2:
            pairwise_dists = pdist(emb, metric="cosine")
            dispersion = float(np.mean(pairwise_dists))
        else:
            dispersion = 0.0

        per_class[cls_name] = {
            "n_images": n,
            "dataset": class_to_dataset[cls_name],
            "confidence_mean": float(np.mean(conf)),
            "confidence_std": float(np.std(conf)),
            "margin_mean": float(np.mean(marg)),
            "margin_std": float(np.std(marg)),
            "entropy_mean": float(np.mean(entr)),
            "entropy_std": float(np.std(entr)),
            "dispersion": round(dispersion, 6),
        }
        logger.info(
            f"  {cls_name:28s} n={n:4d}  "
            f"conf={np.mean(conf):.4f}  margin={np.mean(marg):.4f}  "
            f"entropy={np.mean(entr):.4f}  disp={dispersion:.4f}"
        )

    return per_class


def correlate_with_mllm(per_class, mllm_data):
    """Compute Spearman correlation between each baseline metric and MLLM accuracy.

    Returns dict of {metric_name: {rho, p, n}}.
    """
    # Align by class name
    classes = sorted(set(per_class.keys()) & set(mllm_data.keys()))
    logger.info(f"Correlating {len(classes)} classes with MLLM accuracy")

    mllm_vals = np.array([mllm_data[c] for c in classes])

    metrics = {
        "anchor_score": np.array([per_class[c].get("anchor_score", 0) for c in classes]),
        "confidence": np.array([per_class[c]["confidence_mean"] for c in classes]),
        "margin": np.array([per_class[c]["margin_mean"] for c in classes]),
        "entropy_inv": np.array([-per_class[c]["entropy_mean"] for c in classes]),
        "dispersion": np.array([per_class[c]["dispersion"] for c in classes]),
    }

    results = {}
    for metric_name, vals in metrics.items():
        rho, p = spearmanr(vals, mllm_vals)
        results[metric_name] = {
            "spearman_rho": round(rho, 4),
            "spearman_p": round(p, 4) if not np.isnan(p) else 1.0,
            "n": len(classes),
        }
        logger.info(f"  {metric_name:20s}  ρ={rho:.4f}  p={p:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--model", type=str, default="ViT-L-14",
                        help="CLIP model architecture")
    parser.add_argument("--pretrained", type=str, default="laion2b_s32b_b82k",
                        help="CLIP pretrained weights")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import load_clip_model
    data_root = args.data_root or os.environ.get("SCB5_DATA_ROOT") or str(PROJ / "data" / "scb5")
    out_path = Path(args.out or str(PROJ / "results" / "03_baselines" / "clip_baselines.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["SCB5_DATA_ROOT"] = data_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Output: {out_path}")

    model, tokenizer, transform = load_clip_model(args.model, args.pretrained)
    model = model.to(device).eval()
    model_dtype = next(model.parameters()).dtype
    if device.type == "cuda":
        model = model.half()
        model_dtype = next(model.parameters()).dtype
    logger.info(f"Model dtype: {model_dtype}")

    mllm_data = load_mllm_accuracy()
    logger.info(f"Loaded MLLM accuracy for {len(mllm_data)} classes")

    per_class = compute_baselines(model, tokenizer, transform, device, args.batch_size, model_dtype)

    # Optionally load anchor scores for comparison
    anchor_path = PROJ / "results" / "01_core" / "anchor_score_scb5" / "anchor_scores.json"
    if anchor_path.exists():
        with open(anchor_path) as f:
            anchor_data = json.load(f)
        for ds_name, ds_entry in anchor_data.items():
            for cls_name, cinfo in ds_entry.get("per_class_acc", {}).items():
                if cls_name in per_class:
                    per_class[cls_name]["anchor_score"] = cinfo["acc"]

    correlations = correlate_with_mllm(per_class, mllm_data)

    output = {
        "description": "CLIP-derived baseline predictors vs MLLM accuracy on SCB5 (n=13 classes). "
                       "Computed with ViT-L-14 LAION-2B, FP16, 3-prompt templates.",
        "model": f"{args.model} {args.pretrained}",
        "n_classes": len(per_class),
        "per_class": per_class,
        "correlations": correlations,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary: Spearman ρ vs MLLM accuracy (n=13)")
    print("=" * 60)
    print(f"{'Metric':<20s}  {'ρ':>7s}  {'p':>8s}")
    print("-" * 40)
    for metric_name, corr in correlations.items():
        print(f"{metric_name:<20s}  {corr['spearman_rho']:>7.4f}  {corr['spearman_p']:>8.4f}")
    print("-" * 40)


if __name__ == "__main__":
    main()
