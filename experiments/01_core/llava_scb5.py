#!/usr/bin/env python3
"""
llava_scb5.py
==============
Run LLaVA-1.5-7B on SCB5 classroom behavior datasets (responds to reviewer M6).
Adds a third model family to the main experiment.

Usage:
  python llava_scb5.py --data-root /path/to/scb5/data
"""
import os, json, time, random, argparse, glob
# Set HF_ENDPOINT if using a mirror (e.g., https://hf-mirror.com)
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import read_label

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default=None)
parser.add_argument("--data-root", type=str, required=True, help="Path to SCB5 dataset root")
parser.add_argument("--samples", type=int, default=100)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

model_path = args.model_path
data_root = args.data_root
samples_per_class = args.samples
out_path = Path(args.out) if args.out else Path(__file__).resolve().parent.parent.parent / "results/01_core/llava_scb5"
out_path.mkdir(parents=True, exist_ok=True)
device_str = "cpu" if args.cpu else ("cuda:0" if torch.cuda.is_available() else "cpu")

# SCB5 dataset configs (from active_learning_capped_b32)
DATASET_CFG = {
    "TeacherBehavior": {
        "dir": "SCB5_TeacherBehavior",
        "subdir": "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2",
        "classes": ["guide", "answer", "On-stage interaction", "blackboard-writing",
                     "teacher", "stand", "screen", "blackBoard"],
    },
    "HandriseReadWrite": {
        "dir": "SCB5_HandriseReadWrite",
        "subdir": None,
        "classes": ["hand-raising", "read", "write"],
    },
    "BowTurnHead": {
        "dir": "SCB_BowTurnHead",
        "subdir": None,
        "classes": ["BowHead", "TurnHead"],
    },
}

# Prompt templates for SCB5 classes (matching original MLLM experiments)
PROMPT_PREFIX = "This is a classroom behavior image. Classify the dominant behavior: "
PROMPT_SUFFIX = ". Output only the behavior name."

def load_scb5_dataset(ds_name, cfg):
    """Load images from SCB5 val split, sampling N per class."""
    ds_dir = os.path.join(data_root, cfg["dir"])
    if cfg.get("subdir"):
        ds_dir = os.path.join(ds_dir, cfg["subdir"])

    img_dir = os.path.join(ds_dir, "images", "val")
    lbl_dir = os.path.join(ds_dir, "labels", "val")

    if not os.path.exists(img_dir):
        print(f"  {img_dir} not found")
        return [], [], cfg["classes"]

    classes = cfg["classes"]
    idx_per_class = {i: [] for i in range(len(classes))}

    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        lbl_path = os.path.join(lbl_dir, fname.rsplit('.', 1)[0] + '.txt')
        if not os.path.exists(lbl_path):
            continue
        lbl = read_label(lbl_path)
        if lbl is not None and lbl in idx_per_class:
            idx_per_class[lbl].append(os.path.join(img_dir, fname))

    images, labels = [], []
    for ci in range(len(classes)):
        indices = idx_per_class[ci]
        k = min(samples_per_class, len(indices))
        chosen = random.sample(indices, k) if k < len(indices) else indices
        for fpath in chosen:
            try:
                img = Image.open(fpath).convert("RGB").resize((336, 336), Image.BILINEAR)
                images.append(img)
                labels.append(ci)
            except:
                pass
        print(f"  {classes[ci]}: {k}/{len(indices)} sampled")

    return images, labels, classes

def run_inference(model, processor, images, labels, class_names, ds_name):
    prompt = PROMPT_PREFIX + ", ".join(class_names) + PROMPT_SUFFIX
    results = {c: {"correct": 0, "total": 0} for c in class_names}
    total = len(images)
    done = 0
    t0 = time.time()

    for img, tl in zip(images, labels):
        try:
            text = f"USER: <image>\n{prompt}\nASSISTANT:"
            inputs = processor(text=text, images=img, return_tensors="pt").to(device_str)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=15)

            resp = processor.decode(outputs[0], skip_special_tokens=True)
            resp = resp.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in resp else resp

            pred = -1
            for i, c in enumerate(class_names):
                if c.lower().strip() in resp.lower().strip():
                    pred = i
                    break

            cname = class_names[tl]
            results[cname]["total"] += 1
            if pred == tl:
                results[cname]["correct"] += 1

            done += 1
            if done % 20 == 0 or done == total:
                el = time.time()-t0; rate = done/el if el>0 else 0
                acc = sum(r["correct"] for r in results.values())/max(1,sum(r["total"] for r in results.values()))
                eta = (total-done)/rate if rate > 0 else 0
                print(f"  [{ds_name} {done}/{total}] acc={acc:.3f} {rate:.1f}/s ETA={eta:.0f}s", flush=True)
        except Exception as e:
            print(f"  ERR: {e}")
            results[class_names[tl]]["total"] += 1
            done += 1

    return {c: {"correct": r["correct"], "total": r["total"],
                "acc": round(r["correct"]/r["total"]*100, 1) if r["total"] > 0 else 0}
            for c, r in results.items()}

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading LLaVA-1.5-7B on {device_str}...")
    kwargs = {"torch_dtype": torch.float32 if device_str == "cpu" else torch.float16}
    if model_path:
        mp = os.path.expanduser(model_path)
        model = LlavaForConditionalGeneration.from_pretrained(mp, **kwargs)
        processor = AutoProcessor.from_pretrained(mp)
    else:
        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", **kwargs)
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = model.to(device_str)
    print(f"  loaded")

    random.seed(42)
    all_res = {}

    for ds_name, cfg in DATASET_CFG.items():
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === {ds_name} ===")
        try:
            images, labels, classes = load_scb5_dataset(ds_name, cfg)
            print(f"  Loaded {len(images)} images ({len(classes)} classes)")
            if len(images) == 0:
                print(f"  No images found, skipping")
                continue
        except Exception as e:
            print(f"  Load failed: {e}")
            continue

        results = run_inference(model, processor, images, labels, classes, ds_name)
        all_res[ds_name] = results

        # Save intermediate
        out_file = out_path / f"llava15_7b_{ds_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {out_file.name}", flush=True)

    # Summary
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === Summary ===")
    for ds_name, results in all_res.items():
        tc = sum(r["correct"] for r in results.values())
        tt = sum(r["total"] for r in results.values())
        print(f"  {ds_name}: {tc}/{tt} = {tc/tt*100:.1f}%")

    out_combined = out_path / f"llava15_7b_scb5_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_combined, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"Saved to {out_combined}")

if __name__ == "__main__":
    main()
