#!/usr/bin/env python3
"""
Cross-domain MLLM evaluation on EuroSAT and TissueMNIST using Qwen2-VL-7B.
Samples N images per class, runs MLLM inference, saves per-class accuracy.
"""
import os, json, time, random, numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

# Set HF_ENDPOINT if using a mirror (e.g., https://hf-mirror.com)
# Set HUGGINGFACE_CO_RESOLVE_ENDPOINT similarly for resolve links

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# ===== Config =====
SAMPLES_PER_CLASS = 100
PROJ = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(os.environ.get("OUT", str(PROJ / "results" / "02_robustness" / "cross_domain_mllm")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== Dataset definitions =====
# EuroSAT classes (from torchvision/EuroSAT)
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

# TissueMNIST classes (from MedMNIST)
TISSUEMNIST_CLASSES = [
    "Collecting Duct", "Connecting Tubule", "Distal Convoluted Tubule",
    "Glomerular Endothelial Cells", "Interstitial Endothelial Cells",
    "Leukocytes", "Podocytes", "Proximal Tubule Segments",
]

# ===== Prompt =====
def make_prompt(dataset_name, class_names):
    labels = ", ".join(class_names)
    return (
        f"Classify this image. Choose exactly one category: {labels}. "
        f"Output only the category name, nothing else."
    )

# ===== Load EuroSAT =====
def load_eurosat():
    try:
        from torchvision.datasets import EuroSAT as TVEuroSAT
        root = os.environ.get("EUROSAT_DIR", str(PROJ / "data" / "eurosat_rgb" / "2750"))
        ds = TVEuroSAT(root=root, download=True)
        images, labels = [], []
        for img, lbl in ds:
            images.append(img)
            labels.append(lbl)
        print(f"Loaded EuroSAT: {len(images)} images")
        return images, labels, EUROSAT_CLASSES
    except Exception as e:
        print(f"EuroSAT failed from torchvision: {e}")
        # Fallback: try direct download
        import requests, zipfile, io
        url = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
        print(f"Downloading EuroSAT from Zenodo...")
        r = requests.get(url, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        images, labels = [], []
        for fname in z.namelist():
            if fname.endswith((".jpg", ".png", ".jpeg")):
                class_name = fname.split("/")[-2]
                if class_name in EUROSAT_CLASSES:
                    try:
                        img = Image.open(z.open(fname)).convert("RGB")
                        images.append(img)
                        labels.append(EUROSAT_CLASSES.index(class_name))
                    except Exception:
                        pass
        print(f"Loaded EuroSAT (fallback): {len(images)} images")
        return images, labels, EUROSAT_CLASSES

# ===== Load TissueMNIST =====
def load_tissuemnist():
    try:
        import medmnist
        from medmnist import TissueMNIST
        ds = TissueMNIST(split="test", download=True)
        images, labels = [], []
        for img, lbl in ds:
            # medmnist returns numpy arrays
            img_pil = Image.fromarray(img).convert("RGB")
            images.append(img_pil)
            labels.append(int(lbl))
        print(f"Loaded TissueMNIST: {len(images)} images")
        return images, labels, TISSUEMNIST_CLASSES
    except Exception as e:
        print(f"TissueMNIST error: {e}")
        return None, None, None

# ===== Sample images per class =====
def sample_per_class(images, labels, class_names, n_per_class):
    idx_per_class = {i: [] for i in range(len(class_names))}
    for idx, lbl in enumerate(labels):
        idx_per_class[lbl].append(idx)
    sampled = {}
    for cls_idx, indices in idx_per_class.items():
        n = min(n_per_class, len(indices))
        chosen = random.sample(indices, n)
        sampled[cls_idx] = [(images[i], labels[i]) for i in chosen]
        print(f"  {class_names[cls_idx]}: {n}/{len(indices)} sampled")
    return sampled

# ===== Run MLLM inference =====
def run_inference(model, processor, sampled, class_names, dataset_name):
    prompt = make_prompt(dataset_name, class_names)
    results = {cname: {"correct": 0, "total": 0} for cname in class_names}
    total = sum(len(v) for v in sampled.values())
    done = 0
    t0 = time.time()

    for cls_idx, items in sampled.items():
        for img, true_label in items:
            try:
                conversation = [
                    {"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ]}
                ]
                text = processor.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
                response = processor.batch_decode(
                    output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )[0].strip()

                # Match response to class
                pred = -1
                for i, cname in enumerate(class_names):
                    if cname.lower() in response.lower():
                        pred = i
                        break

                cname = class_names[true_label]
                results[cname]["total"] += 1
                if pred == true_label:
                    results[cname]["correct"] += 1

                done += 1
                if done % 50 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (total - done) / rate if rate > 0 else 0
                    acc_so_far = sum(r["correct"] for r in results.values()) / max(1, sum(r["total"] for r in results.values()))
                    print(f"  [{done}/{total}] acc={acc_so_far:.3f}, {rate:.1f} im/s, ETA={eta:.0f}s")
            except Exception as e:
                print(f"  ERROR on {class_names[true_label]}: {e}")
                results[class_names[true_label]]["total"] += 1

    return results

# ===== Main =====
def main():
    random.seed(42)

    print(f"[{datetime.now()}] Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)
    print(f"Model loaded, params={model.num_parameters():,}")

    all_results = {}

    # ---- EuroSAT ----
    print(f"\n[{datetime.now()}] Loading EuroSAT...")
    e_images, e_labels, e_classes = load_eurosat()
    if e_images:
        print(f"Sampling {SAMPLES_PER_CLASS} per class...")
        e_sampled = sample_per_class(e_images, e_labels, e_classes, SAMPLES_PER_CLASS)
        print(f"Running inference on EuroSAT ({sum(len(v) for v in e_sampled.values())} images)...")
        e_results = run_inference(model, processor, e_sampled, e_classes, "satellite image")
        all_results["EuroSAT"] = {cname: {"correct": r["correct"], "total": r["total"], "acc": round(r["correct"]/r["total"]*100, 1) if r["total"]>0 else 0} for cname, r in e_results.items()}

    # ---- TissueMNIST ----
    print(f"\n[{datetime.now()}] Loading TissueMNIST...")
    t_images, t_labels, t_classes = load_tissuemnist()
    if t_images:
        print(f"Sampling {SAMPLES_PER_CLASS} per class...")
        t_sampled = sample_per_class(t_images, t_labels, t_classes, SAMPLES_PER_CLASS)
        print(f"Running inference on TissueMNIST ({sum(len(v) for v in t_sampled.values())} images)...")
        t_results = run_inference(model, processor, t_sampled, t_classes, "kidney tissue microscopy image")
        all_results["TissueMNIST"] = {cname: {"correct": r["correct"], "total": r["total"], "acc": round(r["correct"]/r["total"]*100, 1) if r["total"]>0 else 0} for cname, r in t_results.items()}

    # Save
    out_path = OUT_DIR / f"qwen2vl7b_cross_domain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[{datetime.now()}] Results saved to {out_path}")
    for ds_name, ds_results in all_results.items():
        print(f"\n{ds_name}:")
        for cname, r in ds_results.items():
            print(f"  {cname:40s} {r['correct']:3d}/{r['total']:3d} = {r['acc']:5.1f}%")

if __name__ == "__main__":
    main()
