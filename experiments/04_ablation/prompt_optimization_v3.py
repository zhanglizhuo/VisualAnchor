#!/usr/bin/env python3
"""
prompt_optimization_v3.py

Three-condition prompt optimization experiment on the five low-AnchorScore
SCB5 classes (follow-up to prompt_optimization.py):

  Condition 1: standard    -- baseline classification prompt
  Condition 2: enhanced    -- negation-style hint (v1, as in the paper):
                              "this image may show {cls}, not {confused}"
  Condition 3: direct      -- direct visual-distinction description (v2):
                              names the target and confuser and describes how
                              to tell them apart visually

Same 20 images per class (seed 42, as in v1), both Qwen3.5-27B and Qwen3.6-27B
via Ollama on the V100 server. Per-image predictions are stored for paired
bootstrap analysis.

Usage (on V100, from repo root):
    python experiments/04_ablation/prompt_optimization_v3.py --model qwen35 [--workers 6]
    python experiments/04_ablation/prompt_optimization_v3.py --model qwen36

Output:
    results/04_ablation/prompt_optimization/ollama_{model}_results_v3.json
"""

import os, sys, json, random, argparse
from pathlib import Path
import base64
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(str(PROJ))

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODELS = {
    "qwen35": "qwen3.5:27b",
    "qwen36": "qwen3.6:27b",
}

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

# Per-class: (dataset, most-confused class, direct visual distinction guide)
# Pairs taken from the v1 analysis (results/04_ablation/prompt_optimization/*).
DISTINCTIONS = {
    "blackBoard": ("TeacherBehavior", "teacher",
        "'blackBoard' means the teacher is standing near the blackboard, not writing; "
        "'teacher' means the teacher is at the front of the classroom in general, "
        "not necessarily near the blackboard."),
    "answer": ("TeacherBehavior", "stand",
        "'answer' means the teacher is answering a student's question or talking to "
        "students; 'stand' means the teacher is simply standing, not talking to students."),
    "stand": ("TeacherBehavior", "answer",
        "'stand' means the teacher is simply standing, not talking to students; 'answer' "
        "means the teacher is answering a student's question or talking to students."),
    "read": ("HandriseReadWrite", "write",
        "'read' means the student is reading, looking at text or a book; 'write' means "
        "the student is writing with a pen or pencil."),
    "BowHead": ("BowTurnHead", "TurnHead",
        "'BowHead' means the head is bent downward; 'TurnHead' means the head is turned "
        "to the side."),
}


def build_prompts():
    """Build the three prompt variants for each class."""
    out = {}
    for cls, (ds_name, confused, distinction) in DISTINCTIONS.items():
        classes = DATASET_CFG[ds_name]["classes"]
        options = ", ".join(classes)
        standard = (
            f"Classify the classroom activity in this image. "
            f"Choose exactly one category from the list: {options}. "
            f"Output only the category name, nothing else."
        )
        cls_display = CLASS_DISPLAY[cls]
        confused_display = CLASS_DISPLAY[confused]
        enhanced = (
            f"Classify the classroom activity in this image. "
            f"Note: this image may show {cls_display}, not {confused_display}. "
            f"Be careful to distinguish between these two. "
            f"Choose exactly one category from the list: {options}. "
            f"Output only the category name, nothing else."
        )
        direct = (
            f"Classify the classroom activity in this image. "
            f"Distinction guide: {distinction} "
            f"Choose exactly one category from the list: {options}. "
            f"Output only the category name, nothing else."
        )
        out[cls] = {
            "dataset": ds_name,
            "most_confused": confused,
            "standard_prompt": standard,
            "enhanced_prompt": enhanced,
            "direct_prompt": direct,
        }
    return out


def load_images_for_class(ds_name, cls_name, samples_per_class=20):
    """Load image paths for a class (same images as v1: seed 42, sorted listing)."""
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
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
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


def run_condition(model_name, prompt_text, images, ds_name, cls_name, max_workers):
    from utils import match_class_name
    classes = DATASET_CFG[ds_name]["classes"]

    def process_one(img_path):
        resp = query_ollama(model_name, prompt_text, img_path)
        if resp is None:
            return None
        return {
            "image": Path(img_path).name,
            "response": resp,
            "is_correct": match_class_name(resp, cls_name, classes),
        }

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one, ip) for ip in images]
        for f in as_completed(futures):
            r = f.result()
            if r is not None:
                results.append(r)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen35", "qwen36"], default="qwen35")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    prompts = build_prompts()
    print(f"Running {args.model} ({OLLAMA_MODELS[args.model]}), "
          f"{args.samples} images/class, 3 conditions, {args.workers} workers")

    out = {}
    for cls, pdata in prompts.items():
        ds_name = pdata["dataset"]
        images = load_images_for_class(ds_name, cls, args.samples)
        print(f"\n[{cls}] ({ds_name}, confused->{pdata['most_confused']}): {len(images)} images")

        cls_out = {"dataset": ds_name, "most_confused": pdata["most_confused"],
                   "n": len(images)}
        for cond in ["standard", "enhanced", "direct"]:
            prompt_text = pdata[f"{cond}_prompt"]
            res = run_condition(args.model, prompt_text, images, ds_name, cls, args.workers)
            n_correct = sum(r["is_correct"] for r in res)
            acc = 100.0 * n_correct / max(1, len(res))
            cls_out[f"{cond}_acc"] = round(acc, 2)
            cls_out[f"{cond}_results"] = res
            print(f"  {cond:9s}: {len(res)} valid, acc={acc:.1f}%")
        out[cls] = cls_out

    out_path = PROJ / "results" / "04_ablation" / "prompt_optimization" / f"ollama_{args.model}_results_v3.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
