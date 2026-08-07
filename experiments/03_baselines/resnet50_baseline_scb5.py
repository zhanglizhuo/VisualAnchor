"""
resnet50_baseline_scb5.py

Control baseline: train a weakly supervised ResNet-50 on SCB5
training data, compute per-class accuracy on val, and test whether it
correlates with MLLM accuracy. This tests the "shared task difficulty"
account: if ANY supervised classifier trained on the same data produces
a similar correlation with MLLM accuracy, then the AnchorScore signal
reflects generic task difficulty rather than CLIP-specific alignment.

If the ResNet-50 correlation is comparable to AnchorScore (rho=0.769),
this supports the shared-task-difficulty interpretation. If it is small
(like DINOv2 kNN rho=+0.050), this confirms the signal is specific to
CLIP's vision-language alignment.

Usage:
  python experiments/03_baselines/resnet50_baseline_scb5.py --epochs 30 --lr 1e-3
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import read_label

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


def build_class_index(cfg):
    """Build global class_name -> class_id mapping across all datasets."""
    cls_to_id = {}
    for ds_name, ds_cfg in cfg.items():
        for c in ds_cfg["classes"]:
            if c not in cls_to_id:
                cls_to_id[c] = len(cls_to_id)
    return cls_to_id


def build_global_label_map(cfg, class_to_id):
    """Build per-dataset local cid → global cid mapping."""
    local_to_global = {}
    for ds_name, ds_cfg in cfg.items():
        for local_cid, class_name in enumerate(ds_cfg["classes"]):
            global_cid = class_to_id[class_name]
            local_to_global[(ds_name, local_cid)] = global_cid
    return local_to_global


def collect_samples(cfg, split, data_root, local_to_global):
    """Collect image paths + labels matching the structure used by SCB5 experiments."""
    all_images, all_labels = [], []
    for ds_name, ds_cfg in cfg.items():
        base = data_root / ds_cfg["dir"]
        sub = ds_cfg["subdir"]
        img_dir = (base / sub / "images" / split) if sub else (base / "images" / split)
        lbl_dir = (base / sub / "labels" / split) if sub else (base / "labels" / split)
        if not img_dir.exists():
            logger.warning(f"Image dir not found: {img_dir}")
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            cid = read_label(lbl_path)
            if cid is None or cid >= len(ds_cfg["classes"]):
                continue
            global_cid = local_to_global.get((ds_name, cid))
            if global_cid is None:
                continue
            all_images.append(str(img_path))
            all_labels.append(global_cid)
    logger.info(f"Loaded {len(all_images)} images from {split} split")
    return all_images, all_labels


class SCB5Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def per_class_accuracy(model, loader, num_classes, device, class_to_id=None):
    """Compute per-class accuracy on validation set."""
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        for i in range(labels.size(0)):
            cls = labels[i].item()
            class_total[cls] += 1
            if preds[i].item() == cls:
                class_correct[cls] += 1
    accuracies = {}
    for cls_name, cls_id in (class_to_id or {}).items():
        if class_total[cls_id] > 0:
            accuracies[cls_name] = round(100.0 * class_correct[cls_id] / class_total[cls_id], 1)
    return accuracies


def load_mllm_accuracy(results_dir):
    """Load canonical 6-MLLM per-class mean from pooled_class_level_results.json."""
    path = Path(results_dir) / "01_core" / "correlation" / "pooled_class_level_results.json"
    with open(path) as f:
        data = json.load(f)
    return {e["class"]: e["mllm_mean"] for e in data["data"] if e["domain"].startswith("SCB5")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze all but the final fc layer")
    args = parser.parse_args()

    data_root = Path(args.data_root or os.environ.get("SCB5_DATA_ROOT", str(PROJ / "data" / "scb5")))
    out_dir = Path(args.out or str(PROJ / "results" / "03_baselines/resnet50_baseline"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Output dir: {out_dir}")

    # Load data (map per-dataset YOLO class_ids to global IDs)
    class_to_id = build_class_index(DATASET_CFG)
    local_to_global = build_global_label_map(DATASET_CFG, class_to_id)
    train_images, train_labels = collect_samples(DATASET_CFG, "train", data_root, local_to_global)
    val_images, val_labels = collect_samples(DATASET_CFG, "val", data_root, local_to_global)

    num_classes = len(class_to_id)
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class mapping: {class_to_id}")

    train_dataset = SCB5Dataset(train_images, train_labels, transform=get_train_transform())
    val_dataset = SCB5Dataset(val_images, val_labels, transform=get_eval_transform())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Compute class weights for imbalance
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts.astype(float) + 1)
    class_weights = class_weights / class_weights.sum() * num_classes  # normalize
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logger.info(f"Class weights (normalized): {dict(zip(class_to_id.keys(), np.round(class_weights, 2)))}")

    # Load pretrained ResNet-50
    logger.info("Loading pretrained ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Replace final layer for our number of classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        logger.info("Frozen backbone (only training fc layer)")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        logger.info(
            f"Epoch {epoch+1:2d}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
            f"Val acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            logger.info(f"  New best val acc: {val_acc:.4f}")

    # Load best model for evaluation
    model.load_state_dict(torch.load(out_dir / "best_model.pt"))
    final_val_acc = evaluate(model, val_loader, device)
    logger.info(f"Best val accuracy: {best_val_acc:.4f} (final: {final_val_acc:.4f})")

    # Per-class accuracy on val
    class_accs = per_class_accuracy(model, val_loader, num_classes, device, class_to_id)
    logger.info("Per-class ResNet-50 accuracy:")
    for cls_name, acc in sorted(class_accs.items(), key=lambda x: -x[1]):
        logger.info(f"  {cls_name:28s}: {acc:5.1f}%")

    # Load MLLM accuracy and compute correlation
    mllm_accs = load_mllm_accuracy(PROJ / "results")
    logger.info("\nMLLM mean accuracy (for comparison):")
    for cls_name in sorted(class_accs.keys()):
        mllm_val = mllm_accs.get(cls_name, 0)
        rn_val = class_accs.get(cls_name, 0)
        logger.info(f"  {cls_name:28s}: ResNet-50={rn_val:5.1f}%  MLLM={mllm_val:5.1f}%")

    # Spearman correlation: ResNet-50 per-class acc vs MLLM per-class acc
    common_classes = [c for c in class_accs if c in mllm_accs]
    rn_vals = [class_accs[c] for c in common_classes]
    mllm_vals = [mllm_accs[c] for c in common_classes]
    rho, p = spearmanr(rn_vals, mllm_vals)

    logger.info(f"\n=== ResNet-50 vs MLLM Accuracy ===")
    logger.info(f"Common classes: {len(common_classes)}")
    logger.info(f"Spearman rho: {rho:.3f} (p={p:.4f})")

    # Compare with DINOv2 baseline
    logger.info(f"\nComparison with existing baselines:")
    logger.info(f"  DINOv2 kNN vs MLLM:  rho=+0.050 (p=0.87)")
    logger.info(f"  AnchorScore vs MLLM: rho=+0.769 (p=0.002)")
    logger.info(f"  ResNet-50 vs MLLM:   rho={rho:+.3f} (p={p:.4f})")

    # Save results
    results = {
        "best_val_acc": round(best_val_acc, 4),
        "final_val_acc": round(final_val_acc, 4),
        "per_class_accuracy": class_accs,
        "per_class_mllm_mean": {c: round(mllm_accs.get(c, 0), 1) for c in sorted(class_accs.keys())},
        "spearman_vs_mllm": {"rho": round(rho, 4), "p": round(p, 4), "n": len(common_classes)},
        "args": vars(args),
    }
    out_path = out_dir / "resnet50_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # Interpretation
    logger.info(f"\n=== Interpretation ===")
    if rho > 0.4:
        logger.info(f"ResNet-50 correlates substantially with MLLM accuracy (rho={rho:.3f}).")
        logger.info(f"This supports the 'shared task difficulty' interpretation: any")
        logger.info(f"supervised classifier trained on SCB5 captures similar per-class")
        logger.info(f"difficulty patterns as MLLMs, suggesting the signal is at least")
        logger.info(f"partially driven by intrinsic task difficulty rather than CLIP-specific")
        logger.info(f"vision-language alignment.")
    elif rho < 0.2:
        logger.info(f"ResNet-50 shows no significant correlation with MLLM accuracy (rho={rho:.3f}, p={p:.4f}).")
        logger.info(f"Like DINOv2 (rho=+0.050), a supervised vision-only classifier does not")
        logger.info(f"predict MLLM per-class difficulty. This is strong evidence that the")
        logger.info(f"AnchorScore signal is specific to CLIP's vision-language alignment")
        logger.info(f"and cannot be replicated by any classifier trained on the same data.")
    else:
        logger.info(f"ResNet-50 shows moderate correlation with MLLM accuracy (rho={rho:.3f}, p={p:.4f}).")
        logger.info(f"This suggests partial support for the shared task difficulty account,")
        logger.info(f"but the signal may still benefit from CLIP-specific alignment.")


if __name__ == "__main__":
    main()
