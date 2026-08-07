#!/usr/bin/env python3
"""
clip_scb5_predictions.py

Compute per-image CLIP zero-shot predictions on SCB5 validation set.
Saves per-image predictions and confusion matrix for downstream experiments
(hybrid annotation, prompt optimization).

Usage:
  python experiments/01_core/clip_scb5_predictions.py
  python experiments/01_core/clip_scb5_predictions.py --data-root /path/to/scb5
  python experiments/01_core/clip_scb5_predictions.py --model-name ViT-L-14 --pretrained laion2b_s32b_b82k
"""

import os, json, sys, argparse, time
from pathlib import Path
from PIL import Image

import torch
import numpy as np

PROJ = Path(__file__).resolve().parent.parent.parent
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import CLASS_NAME_MAP, read_label, load_clip_model

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


def build_all_prompts(classes):
    """Build all prompt strings: list of (class_id, prompt_template_id, text)."""
    all_prompts = []
    for cls_id, cls_name in enumerate(classes):
        mapped = CLASS_NAME_MAP.get(cls_name, cls_name)
        for tpl_id, tpl in enumerate(PROMPT_TEMPLATES):
            all_prompts.append({
                "cls_id": cls_id,
                "tpl_id": tpl_id,
                "text": tpl.format(mapped),
            })
    return all_prompts


def load_clip_local(ckpt_path, model_name="ViT-L-14", device="cuda"):
    """Load CLIP from a local PyTorch checkpoint, bypassing huggingface_hub."""
    import open_clip

    # Create model architecture without pretrained weights
    model, _, transform = open_clip.create_model_and_transforms(
        model_name, pretrained=None
    )
    # Load weights from local checkpoint
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer, transform


@torch.no_grad()
def precompute_text_features(model, tokenizer, all_prompts, device, num_classes):
    """Encode all prompt texts and average features per class (consistent with anchor_scb5.py).

    Returns averaged + re-normalized text features of shape (num_classes, D).
    """
    texts = [p["text"] for p in all_prompts]
    tokens = tokenizer(texts).to(device)
    features = model.encode_text(tokens)
    features = features / features.norm(dim=-1, keepdim=True)
    # Average features across templates per class, then re-normalize
    num_templates = len(all_prompts) // num_classes
    features = features.view(num_classes, num_templates, features.shape[-1]).mean(dim=1)
    features = features / features.norm(dim=-1, keepdim=True)
    return features


@torch.no_grad()
def compute_per_image_predictions(model, transform, classes, text_features, all_prompts, img_dir, lbl_dir):
    """Run CLIP on all val images, return per-image predictions.

    text_features: (num_classes, D), already averaged across templates.
    """
    device = next(model.parameters()).device
    half = next(model.parameters()).dtype == torch.float16

    # Collect image paths and labels
    samples = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        lbl_path = lbl_dir / (os.path.splitext(fname)[0] + ".txt")
        if not lbl_path.exists():
            continue
        lbl = read_label(lbl_path)
        if lbl is not None and 0 <= lbl < len(classes):
            samples.append((str(img_dir / fname), lbl))

    results = []
    conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)

    for img_path, true_id in samples:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        if half:
            tensor = tensor.half()

        img_features = model.encode_image(tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        # Similarity with averaged text features (feature-average method)
        cls_sims = (img_features @ text_features.T).squeeze(0)

        pred_id = cls_sims.argmax(dim=-1).item()
        confidence = torch.softmax(cls_sims, dim=-1)[pred_id].item()

        results.append({
            "path": img_path,
            "true_class_id": true_id,
            "true_class_name": classes[true_id],
            "pred_class_id": pred_id,
            "pred_class_name": classes[pred_id],
            "confidence": round(confidence, 4),
            "correct": int(pred_id == true_id),
        })
        conf_matrix[true_id, pred_id] += 1

    return results, conf_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model-name", default="ViT-L-14")
    parser.add_argument("--pretrained", default="laion2b_s32b_b82k")
    parser.add_argument("--checkpoint", default=None,
                        help="Local checkpoint path (bypasses huggingface_hub download)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    data_root = args.data_root or os.environ.get("SCB5_DATA_ROOT") or str(PROJ / "data" / "scb5")
    SERVER_DATA = Path(data_root)
    print(f"Data root: {SERVER_DATA}")
    assert SERVER_DATA.exists(), f"Data root not found: {SERVER_DATA}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    if args.checkpoint:
        print(f"Loading from local checkpoint: {args.checkpoint}")
        model, tokenizer, transform = load_clip_local(args.checkpoint, args.model_name, device)
    else:
        model, tokenizer, transform = load_clip_model(args.model_name, args.pretrained)
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()
    print(f"Model: {args.model_name} ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    output_dir = Path(args.output_dir or PROJ / "results" / "01_core" / "clip_per_image")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = {}
    all_conf_matrices = {}

    for ds_name, cfg in DATASET_CFG.items():
        print(f"\n=== {ds_name} ===")
        base = SERVER_DATA / cfg["dir"]
        sub = cfg["subdir"]
        img_dir = (base / sub / "images" / "val" if sub else base / "images" / "val")
        lbl_dir = (base / sub / "labels" / "val" if sub else base / "labels" / "val")

        if not img_dir.exists():
            print(f"  Skipping {ds_name}: {img_dir} not found")
            continue

        classes = cfg["classes"]
        all_prompts = build_all_prompts(classes)
        text_features = precompute_text_features(model, tokenizer, all_prompts, device, len(classes))

        t0 = time.time()
        results, conf_matrix = compute_per_image_predictions(
            model, transform, classes, text_features, all_prompts, img_dir, lbl_dir
        )
        elapsed = time.time() - t0

        n_total = len(results)
        n_correct = sum(r["correct"] for r in results)
        acc = 100.0 * n_correct / n_total if n_total > 0 else 0
        # Per-class accuracy
        cls_correct = {c: 0 for c in classes}
        cls_total = {c: 0 for c in classes}
        for r in results:
            cn = r["true_class_name"]
            cls_total[cn] += 1
            cls_correct[cn] += r["correct"]
        cls_acc_str = ", ".join(f"{c}:{100*cls_correct[c]/max(1,cls_total[c]):.1f}%" for c in classes)
        print(f"  {ds_name}: {n_total} images, {acc:.2f}% acc, {elapsed:.0f}s")
        print(f"    {cls_acc_str}")

        all_predictions[ds_name] = {
            "overall_acc": round(acc, 2),
            "n_images": n_total,
            "predictions": results,
        }
        all_conf_matrices[ds_name] = {
            "classes": classes,
            "matrix": conf_matrix.tolist(),
        }

    # Save predictions
    pred_path = output_dir / "per_image_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(all_predictions, f, indent=2)
    print(f"\nPer-image predictions saved to {pred_path}")

    # Save confusion matrix
    cm_path = output_dir / "confusion_matrix.json"
    with open(cm_path, "w") as f:
        json.dump(all_conf_matrices, f, indent=2)
    print(f"Confusion matrix saved to {cm_path}")

    # Print confusion matrices
    print("\n\n=== Confusion Matrices ===")
    for ds_name, cm_data in all_conf_matrices.items():
        classes = cm_data["classes"]
        matrix = np.array(cm_data["matrix"])
        print(f"\n{ds_name}:")
        header_fmt = "{:>20} " + " ".join("{:>12}" for _ in classes)
        header_parts = ["True\\Pred"] + [c[:12] for c in classes]
        header = header_fmt.format(*header_parts)
        print(header)
        for i, cls in enumerate(classes):
            row = matrix[i]
            pct = 100.0 * row / row.sum() if row.sum() > 0 else row
            row_fmt = "{:>20} " + " ".join("{:>12}" for _ in classes)
            row_parts = [cls] + [int(pct[j]) for j in range(len(classes))]
            print(row_fmt.format(*row_parts))


if __name__ == "__main__":
    main()
