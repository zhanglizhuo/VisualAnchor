#!/usr/bin/env python3
"""
cross_domain_llava.py
======================
LLaVA-1.5-7B cross-domain MLLM evaluation on EuroSAT, PathMNIST, BloodMNIST, TissueMNIST.

Usage:
  python cross_domain_llava.py [--cpu] [--samples N] [--out DIR]
  --model-path PATH  explicit path to model files (e.g., model-scope cache)
  --datasets D1,D2  comma-separated list of datasets to run (default: all)
"""

import os, json, time, random, sys
from pathlib import Path
from datetime import datetime
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--samples", type=int, default=50)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--model-path", type=str, default=None)
parser.add_argument("--datasets", type=str, default="EuroSAT,PathMNIST,BloodMNIST,TissueMNIST")
args = parser.parse_args()

force_cpu = args.cpu
samples_per_class = args.samples
PROJ = Path(__file__).resolve().parent.parent.parent
out_path = Path(args.out) if args.out else PROJ / "results" / "02_robustness" / "cross_domain_mllm"
model_path = args.model_path
dataset_list = [d.strip() for d in args.datasets.split(",")]

out_path.mkdir(parents=True, exist_ok=True)

# Tensor device config
if force_cpu:
    device_str = "cpu"
else:
    try:
        import torch
        device_str = "cuda:0" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() else "cpu"
    except:
        device_str = "cpu"

# Lazy torch import
import torch

# Dataset definitions
EUROSAT_CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
PATH_CLASSES = ["Adipose tissue", "Background", "Debris", "Lymphocytes", "Mucus", "Smooth muscle", "Normal colon mucosa", "Cancer-associated stroma", "Tumor epithelium"]
BLOOD_CLASSES = ["Basophil", "Eosinophil", "Erythroblast", "Immature granulocyte", "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"]
TISSUE_CLASSES = ["Collecting Duct", "Connecting Tubule", "Distal Convoluted Tubule", "Glomerular Endothelial Cells", "Interstitial Endothelial Cells", "Leukocytes", "Podocytes", "Proximal Tubule Segments"]

DATASET_REGISTRY = {}

def register(cls):
    DATASET_REGISTRY[cls.__name__] = cls
    return cls

class DatasetBase:
    name = ""
    classes = []

class EuroSAT(DatasetBase):
    name = "EuroSAT"
    classes = EUROSAT_CLASSES

    def load(self):
        from torchvision import datasets as tv_datasets
        root = os.environ.get("EUROSAT_DIR", str(PROJ / "data" / "eurosat_rgb" / "2750"))
        ds = tv_datasets.ImageFolder(root=root)
        images, labels = [], []
        for img, lbl in ds:
            images.append(img.convert("RGB"))
            labels.append(lbl)
        return images, labels, self.classes

class PathMNIST(DatasetBase):
    name = "PathMNIST"
    classes = PATH_CLASSES
    def load(self):
        return self._load_medmnist("PathMNIST", self.classes)

    def _load_medmnist(self, name, classes):
        import medmnist
        DataClass = getattr(medmnist, name)
        ds = DataClass(split="test", download=True)
        images, labels = [], []
        for arr, lbl in zip(ds.imgs, ds.labels.squeeze().tolist()):
            img = Image.fromarray(arr).convert("RGB")
            images.append(img)
            labels.append(int(lbl))
        return images, labels, classes

class BloodMNIST(PathMNIST):
    name = "BloodMNIST"
    classes = BLOOD_CLASSES
    def load(self):
        return self._load_medmnist("BloodMNIST", self.classes)

class TissueMNIST(PathMNIST):
    name = "TissueMNIST"
    classes = TISSUE_CLASSES
    def load(self):
        return self._load_medmnist("TissueMNIST", self.classes)

for cls in [EuroSAT, PathMNIST, BloodMNIST, TissueMNIST]:
    register(cls)

def sample_per_class(images, labels, class_names, n):
    idx_per = {i: [] for i in range(len(class_names))}
    for idx, l in enumerate(labels):
        l = int(l)
        if l in idx_per:
            idx_per[l].append(idx)
    sampled = {}
    for ci, indices in idx_per.items():
        k = min(n, len(indices))
        chosen = random.sample(indices, k) if k < len(indices) else indices
        sampled[class_names[ci]] = [(images[i], labels[i]) for i in chosen]
    return sampled

# Inference loop is inlined in main()

def main():
    if not os.path.isdir(out_path):
        os.makedirs(out_path, exist_ok=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading LLaVA-1.5-7B on {device_str}...")
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    t0 = time.time()
    kwargs = {"torch_dtype": torch.float32 if device_str == "cpu" else torch.float16}
    if model_path:
        mp = os.path.expanduser(model_path)
        model = LlavaForConditionalGeneration.from_pretrained(mp, **kwargs)
        processor = AutoProcessor.from_pretrained(mp)
    else:
        # Set HF_ENDPOINT if using a mirror (e.g., https://hf-mirror.com)
        model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", **kwargs)
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = model.to(device_str)
    print(f"  loaded in {time.time()-t0:.1f}s")

    t0_global = time.time()
    prompt_prefix = "Classify this image. Choose exactly one: "
    prompt_suffix = ". Output only the name."

    for dataset_name in dataset_list:
        if dataset_name not in DATASET_REGISTRY:
            print(f"Unknown dataset {dataset_name}, skipping.")
            continue
        cls = DATASET_REGISTRY[dataset_name]
        dataloader = cls()
        images, labels, class_names = dataloader.load()

        if images is None or len(images) == 0:
            print(f"  Could not load {dataset_name}. Skipping.")
            continue

        random.seed(42)
        sampled = sample_per_class(images, labels, class_names, samples_per_class)
        total_imgs = sum(len(v) for v in sampled.values())
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {dataset_name}: {total_imgs} images ({len(class_names)} classes)")

        prompt = prompt_prefix + ", ".join(class_names) + prompt_suffix
        results = {c: {"correct": 0, "total": 0} for c in class_names}
        total = total_imgs
        done = 0
        t0 = time.time()

        for ci, items in sampled.items():
            for img, tl in items:
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
                    results[ci]["total"] += 1
                    if pred == list(class_names).index(ci):
                        results[ci]["correct"] += 1
                    done += 1
                    if done % 10 == 0 or done == total:
                        el = time.time()-t0; rate = done/el if el>0 else 0; eta = (total-done)/rate if rate>0 else 0
                        acc = sum(r["correct"] for r in results.values())/max(1,sum(r["total"] for r in results.values()))
                        print(f"  [{done}/{total}] acc={acc:.3f} {rate:.1f}/s ETA={eta:.0f}s", flush=True)
                except Exception as e:
                    print(f"  ERR [{ci}]: {e}")
                    results[ci]["total"] += 1
                    done += 1

        out_file = out_path / f"llava15_7b_{dataset_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_data = {c: {"correct": r["correct"], "total": r["total"],
                        "acc": round(r["correct"]/r["total"]*100,1) if r["total"]>0 else 0}
                    for c, r in results.items()}
        with open(out_file, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"  Saved: {out_file.name}", flush=True)

    t_elapsed = time.time() - t0_global
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Done in {t_elapsed/60:.1f} min")

if __name__ == "__main__":
    main()
