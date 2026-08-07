#!/usr/bin/env python3
"""
Unified cross-domain MLLM evaluation.
Supports multiple model families on EuroSAT + MedMNIST.
Usage:
  python cross_domain_mllm.py --model llava-hf/llava-v1.6-mistral-7b-hf [--samples 50] [--datasets EuroSAT,PathMNIST]
  python cross_domain_mllm.py --model Qwen/Qwen2.5-VL-7B-Instruct
  python cross_domain_mllm.py --model OpenGVLab/InternVL2-8B
"""
import os, json, time, random, sys, argparse
from pathlib import Path
from datetime import datetime
from PIL import Image

PROJ = Path(__file__).resolve().parent.parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf")
parser.add_argument("--samples", type=int, default=50)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--datasets", type=str, default="EuroSAT,PathMNIST,BloodMNIST,TissueMNIST")
args = parser.parse_args()

MODEL_NAME = args.model
samples_per_class = args.samples
out_path = Path(args.out) if args.out else PROJ / "results" / "02_robustness" / "cross_domain_mllm"
out_path.mkdir(parents=True, exist_ok=True)
dataset_list = [d.strip() for d in args.datasets.split(",")]

import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}, Model: {MODEL_NAME}")

# --- Model loading ---
MODEL_LOWER = MODEL_NAME.lower()

if "next" in MODEL_LOWER or "1.6" in MODEL_LOWER:
    from transformers import LlavaNextForConditionalGeneration, AutoProcessor
    t0 = time.time()
    model = LlavaNextForConditionalGeneration.from_pretrained(MODEL_NAME, dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = model.to(device)
    print(f"  loaded in {time.time()-t0:.1f}s")
    def run_inference(img, classes_str):
        prompt = f"[INST] <image>\nClassify this image. Choose exactly one: {classes_str}. Output only the name. [/INST]"
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=15)
        resp = processor.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in resp:
            resp = resp.split("[/INST]")[-1].strip()
        return resp.lower()

elif "llava" in MODEL_LOWER:
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    t0 = time.time()
    model = LlavaForConditionalGeneration.from_pretrained(MODEL_NAME, dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = model.to(device)
    print(f"  loaded in {time.time()-t0:.1f}s")
    def make_prompt(classes_str):
        return f"USER: <image>\nClassify this image. Choose exactly one: {classes_str}. Output only the name.\nASSISTANT:"
    def run_inference(img, classes_str):
        prompt = make_prompt(classes_str)
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=15)
        resp = processor.decode(outputs[0], skip_special_tokens=True)
        if "ASSISTANT:" in resp:
            resp = resp.split("ASSISTANT:")[-1].strip()
        return resp.lower()

elif "qwen2.5" in MODEL_LOWER or "qwen2_5" in MODEL_LOWER:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = model.to(device)
    print(f"  loaded in {time.time()-t0:.1f}s")
    def run_inference(img, classes_str):
        prompt = f"Classify this image. Choose exactly one: {classes_str}. Output only the name."
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=15)
        resp = processor.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in resp.lower():
            resp = resp.lower().split("assistant")[-1].strip().lstrip("\n").strip()
        return resp.lower()

elif "qwen" in MODEL_LOWER:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    t0 = time.time()
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = model.to(device)
    print(f"  loaded in {time.time()-t0:.1f}s")
    def run_inference(img, classes_str):
        prompt = f"Classify this image. Choose exactly one: {classes_str}. Output only the name."
        from transformers import Qwen2VLProcessor
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=15)
        resp = processor.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in resp.lower():
            resp = resp.lower().split("assistant")[-1].strip().lstrip("\n").strip()
        return resp.lower()

elif "internvl" in MODEL_LOWER:
    from transformers import AutoModel, AutoTokenizer
    t0 = time.time()
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.to(device)
    print(f"  loaded in {time.time()-t0:.1f}s")
    def run_inference(img, classes_str):
        prompt = f"Classify this image. Choose exactly one: {classes_str}. Output only the name."
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        pixel_values = model.extract_feature(img).to(device)
        with torch.no_grad():
            outputs = model.generate(pixel_values=pixel_values, **inputs, max_new_tokens=15)
        resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return resp.lower()

else:
    raise ValueError(f"Unknown model type: {MODEL_NAME}. Supported: llava, qwen, internvl")

# --- Dataset definitions ---
EUROSAT_CLASSES = ["AnnualCrop","Forest","HerbaceousVegetation","Highway","Industrial","Pasture","PermanentCrop","Residential","River","SeaLake"]
PATH_CLASSES = ["Adipose tissue","Background","Debris","Lymphocytes","Mucus","Smooth muscle","Normal colon mucosa","Cancer-associated stroma","Tumor epithelium"]
BLOOD_CLASSES = ["Basophil","Eosinophil","Erythroblast","Immature granulocyte","Lymphocyte","Monocyte","Neutrophil","Platelet"]
TISSUE_CLASSES = ["Collecting Duct","Connecting Tubule","Distal Convoluted Tubule","Glomerular Endothelial Cells","Interstitial Endothelial Cells","Leukocytes","Podocytes","Proximal Tubule Segments"]

# --- EuroSAT ---
def load_eurosat():
    root = os.environ.get("EUROSAT_DIR", str(PROJ / "data" / "eurosat_rgb" / "2750"))
    if not os.path.isdir(root):
        print(f"EuroSAT data not found at {root}. Set EUROSAT_DIR or download to data/eurosat_rgb/2750/")
        return None, None, None
    from torchvision import datasets as tv_datasets
    ds = tv_datasets.ImageFolder(root=root)
    images, labels = [], []
    for img, lbl in ds:
        images.append(img.convert("RGB"))
        labels.append(lbl)
    return images, labels, EUROSAT_CLASSES

# --- MedMNIST ---
import medmnist

def load_medmnist(name, classes):
    DataClass = getattr(medmnist, name)
    ds = DataClass(split="test", download=True)
    images, labels = [], []
    for arr, lbl in zip(ds.imgs, ds.labels.squeeze().tolist()):
        img = Image.fromarray(arr).convert("RGB")
        images.append(img)
        labels.append(int(lbl))
    return images, labels, classes

def sample_per_class(images, labels, class_names, n):
    idx_per = {i: [] for i in range(len(class_names))}
    for idx, l in enumerate(labels):
        if l in idx_per:
            idx_per[l].append(idx)
    sampled = {}
    for ci, indices in idx_per.items():
        k = min(n, len(indices))
        chosen = random.sample(indices, k) if k < len(indices) else indices
        sampled[class_names[ci]] = [(images[i], labels[i]) for i in chosen]
    return sampled

# --- Main loop ---
DATASETS = {
    "EuroSAT": load_eurosat,
    "PathMNIST": lambda: load_medmnist("PathMNIST", PATH_CLASSES),
    "BloodMNIST": lambda: load_medmnist("BloodMNIST", BLOOD_CLASSES),
    "TissueMNIST": lambda: load_medmnist("TissueMNIST", TISSUE_CLASSES),
}

model_slug = MODEL_NAME.split("/")[-1].replace(".", "_")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
all_results = {}

for ds_name in dataset_list:
    if ds_name not in DATASETS:
        print(f"Unknown dataset: {ds_name}, skip")
        continue
    print(f"\n=== {ds_name} ===")
    images, labels, class_names = DATASETS[ds_name]()
    if images is None:
        print(f"  {ds_name}: no data, skip")
        continue
    random.seed(42)
    sampled = sample_per_class(images, labels, class_names, samples_per_class)
    total_imgs = sum(len(v) for v in sampled.values())
    print(f"  {total_imgs} images ({len(class_names)} classes)")
    classes_str = ", ".join(class_names)
    results = {c: {"correct": 0, "total": 0, "responses": []} for c in class_names}
    t0 = time.time()
    done = 0
    for ci, items in sampled.items():
        for img, tl in items:
            resp = run_inference(img, classes_str)
            correct = any(c.lower() in resp for c in [ci]) and not any(c.lower() in resp for c in class_names if c != ci)
            results[ci]["correct"] += 1 if correct else 0
            results[ci]["total"] += 1
            results[ci]["responses"].append(resp)
            done += 1
            elapsed = time.time() - t0
            avg = elapsed / done
            remaining = (total_imgs - done) * avg
            print(f"  [{done}/{total_imgs}] {ci}: {resp[:20]} -> {'OK' if correct else 'WR'} ({elapsed:.0f}s, ETA {remaining:.0f}s)")
    accs = {c: 100 * v["correct"] / max(1, v["total"]) for c, v in results.items()}
    print(f"  Per-class acc: {json.dumps({k: round(v,1) for k,v in accs.items()})}")
    print(f"  Mean acc: {sum(accs.values())/len(accs):.1f}%")
    all_results[ds_name] = {
        "accuracy": {c: round(v, 2) for c, v in accs.items()},
        "mean_accuracy": round(sum(accs.values()) / len(accs), 2),
        "per_class": {c: {"correct": results[c]["correct"], "total": results[c]["total"]} for c in class_names},
    }

out_file = out_path / f"{model_slug}_{timestamp}.json"
with open(out_file, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_file}")
