#!/usr/bin/env python3
"""
prompt_optimization.py

AnchorScore-guided MLLM prompt optimization.

CLIP confusion matrix reveals visual similarities between classes.
Use this information to design enhanced prompts that help MLLMs
disambiguate hard classes.

Pipeline:
  1. Load CLIP confusion matrix from clip_scb5_predictions.py output
  2. Identify classes with high confusion (low accuracy, high off-diagonal)
  3. Generate enhanced prompts with disambiguation guidance
  4. Run MLLMs with enhanced prompts on affected classes
  5. Compare accuracy vs standard prompt

Usage:
  # Local: analyze confusion and design prompts
  python experiments/04_ablation/prompt_optimization.py --mode analyze

  # Server: run MLLM with enhanced prompts
  python experiments/04_ablation/prompt_optimization.py --mode run --mllm ollama --model qwen35 --samples 50 --workers 1
"""

import os, json, sys, time, random, argparse
from pathlib import Path
import numpy as np
import base64
from PIL import Image
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    requests = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PROJ = Path(__file__).resolve().parent.parent.parent
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODELS = {
    "qwen35": "qwen3.5:27b",
    "qwen36": "qwen3.6:27b",
}

LLAVA_PATH = None


def load_confusion_matrix(path=None):
    """Load confusion matrix from clip_scb5_predictions.py output."""
    if path is None:
        path = PROJ / "results" / "01_core" / "clip_per_image" / "confusion_matrix.json"
    path = Path(path)
    if not path.exists():
        print(f"Confusion matrix not found: {path}")
        print("  Run `python experiments/01_core/clip_scb5_predictions.py` first")
        return None

    with open(path) as f:
        return json.load(f)


def load_mllm_accuracies(path=None):
    """Load per-class MLLM accuracies for comparison."""
    if path is None:
        path = PROJ / "results" / "01_core" / "paper_data" / "mllm_raw.json"
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def analyze_confusion(cm_data):
    """Analyze confusion matrix to find classes needing prompt help.

    For each class, finds the most confused class (highest off-diagonal).
    Returns a dict mapping dataset -> class -> {acc, most_confused, confusion_rate}
    """
    analysis = {}
    for ds_name, cm_entry in cm_data.items():
        classes = cm_entry["classes"]
        matrix = np.array(cm_entry["matrix"])

        ds_analysis = {}
        for i, cls in enumerate(classes):
            total = matrix[i].sum()
            if total == 0:
                continue
            acc = 100.0 * matrix[i, i] / total
            # Find most confused class (highest off-diagonal)
            off_diag = matrix[i].copy()
            off_diag[i] = 0
            most_confused_idx = off_diag.argmax()
            confusion_rate = 100.0 * off_diag[most_confused_idx] / total

            ds_analysis[cls] = {
                "acc": round(acc, 2),
                "most_confused": classes[most_confused_idx],
                "confusion_rate": round(confusion_rate, 2),
            }

        # Sort by accuracy (ascending) to find hardest classes
        analysis[ds_name] = {
            "classes": classes,
            "per_class": ds_analysis,
            "sorted_hardest": sorted(ds_analysis.keys(), key=lambda c: ds_analysis[c]["acc"]),
        }

    return analysis


def generate_enhanced_prompt(analysis, k=3):
    """Generate enhanced prompts for the k hardest classes per dataset.

    Returns dict: dataset -> class -> {standard_prompt, enhanced_prompt}
    """
    prompts = {}
    for ds_name, info in analysis.items():
        classes = info["classes"]
        hardest = info["sorted_hardest"][:k]

        ds_prompts = {}
        for cls in hardest:
            cls_info = info["per_class"][cls]
            confused_with = cls_info["most_confused"]

            # Standard prompt (aligned with main analysis scb5_llm_ollama.py)
            options_str = ", ".join(classes)
            standard = (
                f"Classify the classroom activity in this image. "
                f"Choose exactly one category from the list: {options_str}. "
                f"Output only the category name, nothing else."
            )

            # Enhanced prompt with confusion awareness (main prompt + disambiguation hint)
            cls_display = CLASS_DISPLAY.get(cls, cls)
            confused_display = CLASS_DISPLAY.get(confused_with, confused_with)
            enhanced = (
                f"Classify the classroom activity in this image. "
                f"Note: this image may show {cls_display}, not {confused_display}. "
                f"Be careful to distinguish between these two. "
                f"Choose exactly one category from the list: {options_str}. "
                f"Output only the category name, nothing else."
            )

            ds_prompts[cls] = {
                "standard_prompt": standard,
                "enhanced_prompt": enhanced,
                "anchor_score": cls_info["acc"],
                "most_confused": confused_with,
                "confusion_rate": cls_info["confusion_rate"],
            }

        prompts[ds_name] = ds_prompts

    return prompts


def load_images_for_class(ds_name, cls_name, samples_per_class=50):
    """Load image paths for a specific class from SCB5 val set."""
    cfg = DATASET_CFG[ds_name]
    data_root = os.environ.get("SCB5_DATA_ROOT", str(PROJ / "data" / "scb5"))
    base = Path(data_root) / cfg["dir"]
    sub = cfg["subdir"]
    if sub and (base / sub).exists():
        img_dir = base / sub / "images" / "val"
        lbl_dir = base / sub / "labels" / "val"
    else:
        img_dir = base / "images" / "val"
        lbl_dir = base / "labels" / "val"

    from utils import read_label
    paths = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        lbl_path = lbl_dir / (os.path.splitext(fname)[0] + ".txt")
        if not lbl_path.exists():
            continue
        lbl = read_label(lbl_path)
        if lbl is not None:
            cls_idx = cfg["classes"].index(cls_name)
            if lbl == cls_idx:
                paths.append(str(img_dir / fname))

    random.seed(42)
    return random.sample(paths, min(samples_per_class, len(paths)))


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_ollama(model_name, prompt, image_path):
    """Query Ollama model with an image and prompt."""
    b64 = encode_image(image_path)
    payload = {
        "model": OLLAMA_MODELS.get(model_name, model_name),
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64],
            }
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 32},
        "keep_alive": -1,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
        if resp.status_code == 200:
            return resp.json()["message"]["content"].strip().lower()
    except Exception as e:
        print(f"  Ollama error: {e}")
    return None


def query_llava(prompt, image_path):
    """Query LLaVA model via HuggingFace transformers."""
    import torch
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

    if LLAVA_PATH is None:
        print("LLAVA_PATH not set")
        return None

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        LLAVA_PATH, None, "llava", torch.bfloat16, device_map="auto"
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)

    input_text = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
    input_ids = tokenizer_image_token(input_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.half().cuda(),
            do_sample=False,
            temperature=0.0,
            max_new_tokens=16,
        )
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
    return response


def run_mllm_inference(mllm_type, model_name, prompts, samples_per_class=50, max_workers=4):
    """Run MLLM inference with standard and enhanced prompts.

    Args:
        mllm_type: "ollama" or "llava"
        model_name: model identifier
        prompts: dict from generate_enhanced_prompt()
        samples_per_class: images per class

    Returns:
        results dict with per-image predictions
    """
    results = {}
    for ds_name, ds_prompts in prompts.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        ds_results = {}

        # Only include low-AnchorScore classes (hard classes need prompt help)
        for cls_name, prompt_data in ds_prompts.items():
            if prompt_data['anchor_score'] > 50:
                print(f"\n  Skipping {cls_name} (AnchorScore={prompt_data['anchor_score']}% > 50)")
                ds_results[cls_name] = {
                    "anchor_score": prompt_data["anchor_score"],
                    "most_confused": prompt_data["most_confused"],
                    "skipped": True,
                }
                continue
            print(f"\n  Class: {cls_name} (AnchorScore={prompt_data['anchor_score']}%, "
                  f"most confused with {prompt_data['most_confused']})")

            images = load_images_for_class(ds_name, cls_name, samples_per_class)
            print(f"  Loaded {len(images)} images")

            cls_results = {"standard": [], "enhanced": []}

            for prompt_type in ["standard", "enhanced"]:
                prompt_text = prompt_data[f"{prompt_type}_prompt"]
                correct = 0
                total = 0

                def process_one(img_path):
                    if mllm_type == "ollama":
                        resp = query_ollama(model_name, prompt_text, img_path)
                    else:
                        resp = query_llava(prompt_text, img_path)

                    if resp is None:
                        return None

                    from utils import match_class_name
                    is_correct = match_class_name(resp, cls_name, DATASET_CFG[ds_name]["classes"])
                    return {
                        "response": resp,
                        "is_correct": is_correct,
                    }

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_one, ip) for ip in images]
                    for f in as_completed(futures):
                        result = f.result()
                        if result is not None:
                            cls_results[prompt_type].append(result)

                acc = 100.0 * sum(r["is_correct"] for r in cls_results[prompt_type]) / len(cls_results[prompt_type])
                print(f"    {prompt_type}: {len(cls_results[prompt_type])} valid, acc={acc:.1f}%")

            ds_results[cls_name] = {
                "anchor_score": prompt_data["anchor_score"],
                "most_confused": prompt_data["most_confused"],
                "n_standard": len(cls_results["standard"]),
                "n_enhanced": len(cls_results["enhanced"]),
                "standard_acc": round(100.0 * sum(r["is_correct"] for r in cls_results["standard"]) / max(1, len(cls_results["standard"])), 2),
                "enhanced_acc": round(100.0 * sum(r["is_correct"] for r in cls_results["enhanced"]) / max(1, len(cls_results["enhanced"])), 2),
                "standard_results": cls_results["standard"],
                "enhanced_results": cls_results["enhanced"],
            }

        results[ds_name] = ds_results

    return results


def analyze_mode(cm_path=None):
    """Mode: analyze confusion and design prompts (no MLLM inference needed)."""
    cm_data = load_confusion_matrix(cm_path)
    if cm_data is None:
        return

    analysis = analyze_confusion(cm_data)

    print("\n=== CLIP Confusion Analysis ===")
    all_hard = []
    for ds_name, info in analysis.items():
        print(f"\n{ds_name}:")
        print(f"{'Class':>25} {'Acc%':>7} {'Most confused':>25} {'Rate%':>7}")
        print("-" * 70)
        for cls in info["sorted_hardest"]:
            ci = info["per_class"][cls]
            print(f"{cls:>25} {ci['acc']:>7.1f} {ci['most_confused']:>25} {ci['confusion_rate']:>7.1f}")
            all_hard.append((ds_name, cls, ci))

    # Generate enhanced prompts
    prompts = generate_enhanced_prompt(analysis, k=3)
    prompt_path = PROJ / "results" / "01_core" / "clip_per_image" / "enhanced_prompts.json"
    with open(prompt_path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"\nEnhanced prompts saved to {prompt_path}")

    # Summary of which classes to run
    print("\n=== Classes Recommended for Prompt Optimization ===")
    for ds_name, ds_prompts in prompts.items():
        print(f"\n{ds_name}:")
        for cls, pdata in ds_prompts.items():
            print(f"  {cls} (acc={pdata['anchor_score']}%, confused→{pdata['most_confused']}):")
            print(f"    Standard: {pdata['standard_prompt'][:80]}...")
            print(f"    Enhanced: {pdata['enhanced_prompt'][:80]}...")

    # Plot confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for idx, (ds_name, cm_entry) in enumerate(cm_data.items()):
        classes = cm_entry["classes"]
        matrix = np.array(cm_entry["matrix"])
        # Convert to percentages
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix_pct = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums > 0)

        ax = axes[idx]
        im = ax.imshow(matrix_pct, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels([c[:10] for c in classes], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([c[:10] for c in classes], fontsize=8)
        ax.set_title(ds_name)

        for i in range(len(classes)):
            for j in range(len(classes)):
                val = matrix_pct[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    plt.suptitle("CLIP Confusion Matrices (row-normalized)", fontsize=14)
    plt.tight_layout()
    plot_path = PROJ / "results" / "01_core" / "clip_per_image" / "confusion_matrix.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nConfusion matrix plot saved to {plot_path}")

    return prompts


def run_mode(mllm_type, model_name, samples_per_class, max_workers):
    """Mode: run MLLM inference with enhanced prompts."""
    prompt_path = PROJ / "results" / "01_core" / "clip_per_image" / "enhanced_prompts.json"
    if not prompt_path.exists():
        print("Enhanced prompts not found. Running analysis first...")
        cm_path = PROJ / "results" / "01_core" / "clip_per_image" / "confusion_matrix.json"
        prompts = analyze_mode(cm_path)
    else:
        with open(prompt_path) as f:
            prompts = json.load(f)

    if not prompts:
        print("No prompts to evaluate")
        return

    print(f"\nRunning MLLM inference: {mllm_type} / {model_name}")
    print(f"Samples per class: {samples_per_class}, workers: {max_workers}")

    results = run_mllm_inference(mllm_type, model_name, prompts, samples_per_class, max_workers)

    output_path = PROJ / "results" / "04_ablation" / "prompt_optimization" / f"{mllm_type}_{model_name}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print("\n=== Prompt Optimization Results ===")
    for ds_name, ds_results in results.items():
        print(f"\n{ds_name}:")
        total_std = 0
        total_enh = 0
        std_correct = 0
        enh_correct = 0
        for cls, r in ds_results.items():
            if r.get("skipped", False):
                print(f"  {cls:>25}: skipped (AnchorScore={r['anchor_score']}%)")
                continue
            n_s = r["n_standard"]
            n_e = r["n_enhanced"]
            acc_s = r["standard_acc"]
            acc_e = r["enhanced_acc"]
            delta = acc_e - acc_s
            total_std += n_s
            total_enh += n_e
            std_correct += n_s * acc_s / 100
            enh_correct += n_e * acc_e / 100
            sign = "+" if delta >= 0 else ""
            print(f"  {cls:>25}: std={acc_s:.1f}% → enh={acc_e:.1f}% ({sign}{delta:.1f}pp, n={n_s})")

        if total_std > 0 and total_enh > 0:
            print(f"  {'Overall':>25}: std={100*std_correct/total_std:.1f}% → enh={100*enh_correct/total_enh:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["analyze", "run"], default="analyze",
                        help="analyze=confusion+prompts, run=MLLM inference")
    parser.add_argument("--mllm", choices=["ollama", "llava"], default="ollama")
    parser.add_argument("--model", default="qwen35",
                        help="Model name (ollama: qwen35/qwen36, llava: path)")
    parser.add_argument("--samples", type=int, default=50,
                        help="Images per class for MLLM inference")
    parser.add_argument("--workers", type=int, default=4,
                        help="Max parallel workers for MLLM inference")
    parser.add_argument("--confusion-matrix", default=None,
                        help="Path to confusion matrix JSON")
    args = parser.parse_args()

    if args.mode == "analyze":
        analyze_mode(args.confusion_matrix)
    elif args.mode == "run":
        run_mode(args.mllm, args.model, args.samples, args.workers)


if __name__ == "__main__":
    main()
