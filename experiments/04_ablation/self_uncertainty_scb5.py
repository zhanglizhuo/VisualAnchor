#!/usr/bin/env python3
"""
self_uncertainty_scb5.py

MLLM self-uncertainty baseline on SCB5 (responds to reviewers R1, R3, DA).
Asks each MLLM to classify the image AND rate its own confidence (1-5).
Compares self-uncertainty vs AnchorScore as predictors of per-class MLLM accuracy.

Two modes:
  --source ollama  : Query ollama models via HTTP (Qwen3.x, Gemma4)
  --source llava   : Load LLaVA-1.5-7B via HF transformers

Usage:
  python self_uncertainty_scb5.py --source ollama --samples 50
  python self_uncertainty_scb5.py --source llava  --samples 50 --model-path /path/to/llava
"""

import os, json, time, base64, random, argparse
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

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

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODELS = [
    "qwen3.5:27b",
    "qwen3.6:27b",
    "qwen3.6:35b-a3b",
    "gemma4:26b",
]

CLASS_DISPLAY = {
    "guide": "guiding students",
    "answer": "answering questions",
    "On-stage interaction": "on-stage interaction",
    "blackboard-writing": "writing on blackboard",
    "teacher": "teacher at the front",
    "stand": "standing",
    "screen": "looking at screen",
    "blackBoard": "near a blackboard",
    "hand-raising": "raising hand",
    "read": "reading",
    "write": "writing",
    "BowHead": "bowing head",
    "TurnHead": "turning head",
}


def load_scb5_samples(ds_name, cfg, samples_per_class, seed=42):
    """Load SCB5 val samples, N per class."""
    random.seed(seed)
    base = SERVER_DATA / cfg["dir"]
    sub = cfg["subdir"]
    img_dir = (base / sub / "images" / "val" if sub else base / "images" / "val")
    lbl_dir = (base / sub / "labels" / "val" if sub else base / "labels" / "val")

    if not img_dir.exists():
        return []

    classes = cfg["classes"]
    idx_per_class = {i: [] for i in range(len(classes))}
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        lbl_path = lbl_dir / (os.path.splitext(fname)[0] + ".txt")
        if not lbl_path.exists():
            continue
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from utils import read_label
        lbl = read_label(lbl_path)
        if lbl is not None and lbl in idx_per_class:
            idx_per_class[lbl].append(str(img_dir / fname))

    samples = []
    for ci, cls in enumerate(classes):
        paths = idx_per_class[ci]
        k = min(samples_per_class, len(paths))
        chosen = random.sample(paths, k) if k < len(paths) else paths
        for p in chosen:
            samples.append({"path": p, "class_id": ci, "class_name": cls, "ds": ds_name})
    return samples


def build_prompt(class_names):
    display = [CLASS_DISPLAY.get(c, c) for c in class_names]
    return (
        f"Classify the classroom behavior in this image. "
        f"Choose exactly one: {', '.join(display)}. "
        f"First output the class name, then on a new line write "
        f"'Confidence: X' where X is 1 (not sure) to 5 (very sure)."
    )


def parse_response(resp, class_names):
    """Parse class prediction and confidence from response text."""
    resp_lower = resp.lower().strip()
    pred = -1
    for i, c in enumerate(class_names):
        from utils import match_class_name
        if match_class_name(resp, c, class_names):
            pred = i
            break

    conf = 3
    if "confidence:" in resp_lower:
        part = resp_lower.split("confidence:")[-1].strip()[:2]
        try:
            conf = int(part[0])
            conf = max(1, min(5, conf))
        except Exception:
            conf = 3
    elif "confident" in resp_lower:
        for ch in resp_lower:
            if ch.isdigit():
                conf = max(1, min(5, int(ch)))
                break
    return pred, conf


def classify_ollama(model, img_path, prompt, max_retries=2):
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 48},
            }, timeout=120)
            content = resp.json().get("message", {}).get("content", "").strip()
            return content
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
            else:
                return f"ERROR: {e}"


def run_ollama_model(model_name, samples_by_ds, prompt_by_ds):
    """Run one ollama model on all samples. Returns per-ds per-class summary."""
    model_tag = model_name.replace(":", "_").replace(".", "_")
    out_path = PROJ / "results" / "04_ablation" / "self_uncertainty" / f"ollama_{model_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"  [{model_name}] cached, skipping")
        with open(out_path) as f:
            return json.load(f)

    all_summary = {}
    for ds_name, samples in samples_by_ds.items():
        classes = DATASET_CFG[ds_name]["classes"]
        prompt = prompt_by_ds[ds_name]
        results = {c: {"correct": 0, "total": 0, "confidences": []} for c in classes}
        total = len(samples)
        t0 = time.time()

        def process(s):
            content = classify_ollama(model_name, s["path"], prompt)
            pred, conf = parse_response(content, classes)
            return s["class_name"], pred, conf, s["class_id"]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process, s): s for s in samples}
            done = 0
            for future in as_completed(futures):
                cls_name, pred, conf, cid = future.result()
                results[cls_name]["total"] += 1
                results[cls_name]["confidences"].append(conf)
                if pred == cid:
                    results[cls_name]["correct"] += 1
                done += 1
                if done % 20 == 0 or done == total:
                    el = time.time() - t0
                    print(f"    [{model_name} {ds_name} {done}/{total}] "
                          f"{done/el:.1f}/s", flush=True)

        summary = {}
        for c in classes:
            r = results[c]
            acc = r["correct"] / r["total"] * 100 if r["total"] > 0 else 0
            mc = float(np.mean(r["confidences"])) if r["confidences"] else 0
            summary[c] = {
                "correct": r["correct"], "total": r["total"],
                "acc": round(acc, 1),
                "mean_confidence": round(mc, 2),
            }
        all_summary[ds_name] = summary

    with open(out_path, "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"  Saved {out_path.name}")
    return all_summary


def run_llava(model_path, samples_by_ds, prompt_by_ds, device_str, samples_per_class):
    import torch
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    print(f"  Loading LLaVA-1.5-7B on {device_str}...")
    kwargs = {"torch_dtype": torch.float32 if device_str == "cpu" else torch.float16}
    mp = os.path.expanduser(model_path) if model_path else "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(mp, **kwargs)
    processor = AutoProcessor.from_pretrained(mp)
    model = model.to(device_str)
    print("  loaded")

    out_path = PROJ / "results" / "04_ablation" / "self_uncertainty" / "llava15_7b.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_summary = {}
    for ds_name, samples in samples_by_ds.items():
        classes = DATASET_CFG[ds_name]["classes"]
        prompt = prompt_by_ds[ds_name]
        results = {c: {"correct": 0, "total": 0, "confidences": []} for c in classes}
        total = len(samples)
        t0 = time.time()

        for idx, s in enumerate(samples):
            try:
                img = Image.open(s["path"]).convert("RGB").resize((336, 336), Image.BILINEAR)
                text = f"USER: <image>\n{prompt}\nASSISTANT:"
                inputs = processor(text=text, images=img, return_tensors="pt").to(device_str)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=40)
                resp = processor.decode(outputs[0], skip_special_tokens=True)
                resp = resp.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in resp else resp
                pred, conf = parse_response(resp, classes)

                cname = s["class_name"]
                results[cname]["total"] += 1
                results[cname]["confidences"].append(conf)
                if pred == s["class_id"]:
                    results[cname]["correct"] += 1
            except Exception as e:
                print(f"    ERR: {e}")
                results[s["class_name"]]["total"] += 1
                results[s["class_name"]]["confidences"].append(3)

            if (idx + 1) % 20 == 0 or (idx + 1) == total:
                el = time.time() - t0
                rate = (idx + 1) / el if el > 0 else 0
                print(f"    [{ds_name} {idx+1}/{total}] {rate:.1f}/s", flush=True)

        summary = {}
        for c in classes:
            r = results[c]
            acc = r["correct"] / r["total"] * 100 if r["total"] > 0 else 0
            mc = float(np.mean(r["confidences"])) if r["confidences"] else 0
            summary[c] = {
                "correct": r["correct"], "total": r["total"],
                "acc": round(acc, 1),
                "mean_confidence": round(mc, 2),
            }
        all_summary[ds_name] = summary

        out_inter = out_path.parent / f"llava15_7b_{ds_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_inter, "w") as f:
            json.dump({ds_name: summary}, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"  Saved {out_path.name}")
    return all_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["ollama", "llava"], required=True)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--data-root", type=str, default=None)
    args = parser.parse_args()

    global SERVER_DATA
    if args.data_root:
        SERVER_DATA = Path(args.data_root)

    random.seed(42)
    samples_by_ds = {}
    prompt_by_ds = {}
    for ds_name, cfg in DATASET_CFG.items():
        samples = load_scb5_samples(ds_name, cfg, args.samples)
        samples_by_ds[ds_name] = samples
        prompt_by_ds[ds_name] = build_prompt(cfg["classes"])
        print(f"  {ds_name}: {len(samples)} samples ({len(cfg['classes'])} classes)")

    if args.source == "ollama":
        for model in OLLAMA_MODELS:
            print(f"\n--- Ollama: {model} ---")
            run_ollama_model(model, samples_by_ds, prompt_by_ds)
    else:
        import torch
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"\n--- LLaVA-1.5-7B ---")
        run_llava(args.model_path, samples_by_ds, prompt_by_ds, device_str, args.samples)

    print("\nDone. Run analysis/self_uncertainty_correlation.py to compute correlations.")


if __name__ == "__main__":
    main()
