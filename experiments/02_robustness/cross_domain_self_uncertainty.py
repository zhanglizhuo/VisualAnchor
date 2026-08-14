#!/usr/bin/env python3
"""
cross_domain_self_uncertainty.py
=================================
MLLM self-uncertainty baseline on the cross-domain datasets.
Asks the MLLM to rate its own confidence (1-5) after classification.
Compares self-uncertainty vs AnchorScore as predictors of MLLM accuracy.

Usage:
  python cross_domain_self_uncertainty.py --model-path /path/to/llava
"""
import os, json, time, random, argparse
# Set HF_ENDPOINT if using a mirror (e.g., https://hf-mirror.com)
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default=None)
parser.add_argument("--samples", type=int, default=50)
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()

model_path = args.model_path
samples_per_class = args.samples
out_path = Path(args.out) if args.out else Path(__file__).resolve().parent.parent.parent / "results/02_robustness/cross_domain_self_uncertainty"
out_path.mkdir(parents=True, exist_ok=True)

device_str = "cuda:0" if torch.cuda.is_available() else "cpu"

EUROSAT_CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
TISSUE_CLASSES = ["Collecting Duct", "Connecting Tubule", "Distal Convoluted Tubule", "Glomerular Endothelial Cells", "Interstitial Endothelial Cells", "Leukocytes", "Podocytes", "Proximal Tubule Segments"]
PATH_CLASSES = ["Adipose tissue", "Background", "Debris", "Lymphocytes", "Mucus", "Smooth muscle", "Normal colon mucosa", "Cancer-associated stroma", "Tumor epithelium"]
BLOOD_CLASSES = ["Basophil", "Eosinophil", "Erythroblast", "Immature granulocyte", "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"]

def load_eurosat():
    root = os.environ.get("EUROSAT_DIR", str(Path(__file__).resolve().parent.parent.parent / "data" / "eurosat_rgb" / "2750"))
    images, labels = [], []
    for ci, cn in enumerate(EUROSAT_CLASSES):
        dir_path = os.path.join(root, cn)
        files = sorted([f for f in os.listdir(dir_path) if f.endswith(".jpg")])[:200]
        for f in files:
            img = Image.open(os.path.join(dir_path, f)).convert("RGB")
            images.append(img); labels.append(ci)
    return images, labels

def load_medmnist(name, classes):
    import medmnist
    DataClass = getattr(medmnist, name)
    ds = DataClass(split="test", download=True)
    images, labels = [], []
    for arr, lbl in zip(ds.imgs, ds.labels.squeeze().tolist()):
        if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1: arr = np.concatenate([arr]*3, axis=-1)
        img = Image.fromarray(arr.astype(np.uint8)).resize((336,336), Image.BILINEAR)
        images.append(img)
        labels.append(lbl)
    return images, labels

def sample_per_class(images, labels, class_names, n):
    idx_per = {i: [] for i in range(len(class_names))}
    for idx, l in enumerate(labels):
        l = int(l)
        if l in idx_per: idx_per[l].append(idx)
    sampled = {}
    for ci, indices in idx_per.items():
        k = min(n, len(indices))
        chosen = random.sample(indices, k) if k < len(indices) else indices
        sampled[class_names[ci]] = [(images[i], labels[i]) for i in chosen]
    return sampled

def run_with_confidence(model, processor, sampled, class_names, ds_name):
    """Run classification + confidence rating in a single turn."""
    cls_prompt = "Classify this image. Choose exactly one: " + ", ".join(class_names) + ". Output only the name."
    results = {c: {"correct": 0, "total": 0, "confidences": []} for c in class_names}
    total = sum(len(v) for v in sampled.values())
    done = 0
    t0 = time.time()

    for ci, items in sampled.items():
        for img, tl in items:
            try:
                # Single-turn: ask for classification + confidence
                text = (f"USER: <image>\n{cls_prompt}\n"
                        f"After stating the class name, on a new line write "
                        f"'Confidence: X' where X is 1 (not sure) to 5 (very sure).\n"
                        f"ASSISTANT:")
                inputs = processor(text=text, images=img, return_tensors="pt").to(device_str)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=30)

                resp = processor.decode(outputs[0], skip_special_tokens=True)
                resp = resp.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in resp else resp

                # Parse class prediction
                pred = -1
                for i, c in enumerate(class_names):
                    if c.lower().strip() in resp.lower().strip():
                        pred = i
                        break

                # Parse confidence (1-5)
                conf = 3  # default
                resp_lower = resp.lower()
                if "confidence:" in resp_lower:
                    conf_part = resp_lower.split("confidence:")[-1].strip()[:2]
                    try:
                        conf = int(conf_part[0])
                        conf = max(1, min(5, conf))
                    except:
                        conf = 3
                elif "confident" in resp_lower:
                    # Try to extract number near "confident"
                    for ch in resp_lower:
                        if ch.isdigit():
                            conf = int(ch)
                            break

                results[ci]["total"] += 1
                results[ci]["confidences"].append(conf)
                if pred == list(class_names).index(ci):
                    results[ci]["correct"] += 1

                done += 1
                if done % 10 == 0 or done == total:
                    el = time.time()-t0; rate = done/el if el>0 else 0
                    acc = sum(r["correct"] for r in results.values())/max(1,sum(r["total"] for r in results.values()))
                    print(f"  [{ds_name} {done}/{total}] acc={acc:.3f} {rate:.1f}/s", flush=True)
            except Exception as e:
                print(f"  ERR [{ci}]: {e}")
                results[ci]["total"] += 1
                results[ci]["confidences"].append(3)
                done += 1

    # Compute per-class summary
    summary = {}
    for c in class_names:
        r = results[c]
        acc = r["correct"]/r["total"]*100 if r["total"] > 0 else 0
        mean_conf = np.mean(r["confidences"]) if r["confidences"] else 0
        summary[c] = {
            "correct": r["correct"], "total": r["total"],
            "acc": round(acc, 1),
            "mean_confidence": round(mean_conf, 2),
        }
    return summary

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

    datasets = [
        ("EuroSAT", lambda: load_eurosat(), EUROSAT_CLASSES),
        ("TissueMNIST", lambda: load_medmnist("TissueMNIST", TISSUE_CLASSES), TISSUE_CLASSES),
    ]

    for ds_name, loader, classes in datasets:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === {ds_name} ===")
        try:
            images, labels = loader()
        except Exception as e:
            print(f"  Load failed: {e}")
            continue
        sampled = sample_per_class(images, labels, classes, samples_per_class)
        total_imgs = sum(len(v) for v in sampled.values())
        print(f"  {total_imgs} images ({len(classes)} classes)")
        summary = run_with_confidence(model, processor, sampled, classes, ds_name)
        all_res[ds_name] = summary

        # Save intermediate
        out_file = out_path / f"self_uncertainty_{ds_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {out_file.name}", flush=True)

    # Compute correlation: self-confidence vs accuracy (per class)
    print(f"\n=== Self-Uncertainty vs Accuracy Correlation ===")
    for ds_name, summary in all_res.items():
        accs = [v["acc"] for v in summary.values()]
        confs = [v["mean_confidence"] for v in summary.values()]
        rho, p = spearmanr(confs, accs)
        print(f"  {ds_name}: n={len(accs)}, rho={rho:.3f}, p={p:.3f}")
        print(f"    accuracy range: {min(accs):.1f}-{max(accs):.1f}%")
        print(f"    confidence range: {min(confs):.1f}-{max(confs):.1f}")

    out_combined = out_path / f"self_uncertainty_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_combined, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nSaved summary to {out_combined}")

if __name__ == "__main__":
    main()
