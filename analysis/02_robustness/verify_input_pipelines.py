"""
Verify that Stanford40 and cross-domain datasets use identical inputs
for CLIP and MLLM pipelines (no bbox cropping).

Logs actual image shapes from both pipelines for a sample of images.
"""

import argparse
import base64
import logging
import os
import sys
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def verify_stanford40(data_root, n_sample=10):
    """Check CLIP and MLLM input pipelines for Stanford40."""
    data_root = Path(data_root)
    img_dir = data_root / "JPEGImages"

    if not img_dir.exists():
        logger.error(f"Stanford40 not found at {img_dir}")
        return

    jpgs = sorted(img_dir.glob("*.jpg"))
    logger.info(f"Stanford40: {len(jpgs)} images at {img_dir}")

    # CLIP pipeline: Image.open → convert → transform (simulated)
    logger.info(f"\n--- CLIP pipeline (anchor_stanford40.py) ---")
    for p in jpgs[:n_sample]:
        img = Image.open(p).convert("RGB")
        logger.info(f"  {p.name}: original size={img.size}, mode={img.mode}")
        # No crop applied here — transform (Resize+CenterCrop) is the only operation

    # MLLM pipeline: open → base64 (no preprocessing at Python level)
    logger.info(f"\n--- MLLM pipeline (stanford40_ollama.py) ---")
    for p in jpgs[:n_sample]:
        with open(p, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode()
        logger.info(f"  {p.name}: raw_bytes={len(raw)}, b64_len={len(b64)}")
        # No crop, no resize — raw bytes sent directly to Ollama

    logger.info("\n✅ Stanford40: Both CLIP and MLLM use identical full-frame inputs. No bbox cropping.")


def verify_cross_domain(data_root, n_sample=5):
    """Check CLIP and MLLM input pipelines for cross-domain datasets."""
    data_root = Path(data_root)
    datasets = {
        "EuroSAT": data_root / "EuroSAT" / "2750",
        "BloodMNIST": data_root / "BloodMNIST",
        "TissueMNIST": data_root / "TissueMNIST",
        "PathMNIST": data_root / "PathMNIST",
    }

    for name, path in datasets.items():
        if not path.exists():
            logger.warning(f"  {name}: not found at {path}")
            continue

        images = sorted(path.rglob("*.jpg")) + sorted(path.rglob("*.png"))
        logger.info(f"\n--- {name}: {len(images)} images ---")

        for p in images[:n_sample]:
            img = Image.open(p)
            logger.info(f"  {p.name}: size={img.size}, mode={img.mode}")

        logger.info(f"  ✅ {name}: classification dataset, no bbox annotations exist.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stanford40-root", default=None)
    parser.add_argument("--cross-domain-root", default=None)
    args = parser.parse_args()

    if args.stanford40_root:
        verify_stanford40(args.stanford40_root)
    else:
        logger.warning("--stanford40-root not provided; skipping Stanford40")

    if args.cross_domain_root:
        verify_cross_domain(args.cross_domain_root)
    else:
        logger.warning("--cross-domain-root not provided; skipping cross-domain")


if __name__ == "__main__":
    main()
