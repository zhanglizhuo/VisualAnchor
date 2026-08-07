#!/usr/bin/env python3
"""Sequential MLLM inference on SCB5 TeacherBehavior (no threading)."""
import os, json, time, base64, sys
from pathlib import Path
import requests

PROJ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJ / "experiments"))
from utils import CLASS_NAME_MAP, read_label

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.6:35b-a3b"
DATA_ROOT = Path("/home/broadsense/works/lizhuo/AutoResearchClaw/datasets_scb")

CLASSES = [
    "guide", "answer", "On-stage interaction", "blackboard-writing",
    "teacher", "stand", "screen", "blackBoard",
]

DS_DIR = DATA_ROOT / "SCB5_TeacherBehavior" / "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2"
IMG_DIR = DS_DIR / "images" / "val"
LBL_DIR = DS_DIR / "labels" / "val"

classes_str = ", ".join(CLASS_NAME_MAP.get(c, c) for c in CLASSES)
PROMPT = (
    "Classify the classroom activity in this image. "
    "Choose exactly one category from the list: {classes}. "
    "Output only the category name, nothing else."
).format(classes=classes_str)

samples = []
for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    lbl_path = LBL_DIR / (os.path.splitext(fname)[0] + ".txt")
    if not lbl_path.exists():
        continue
    lbl = read_label(lbl_path)
    if lbl is not None and 0 <= lbl < len(CLASSES):
        samples.append((str(IMG_DIR / fname), lbl))

print(f"Total samples: {len(samples)}")
n_total = len(samples)

cls_correct = {c: 0 for c in CLASSES}
cls_total = {c: 0 for c in CLASSES}
t0 = time.time()

for idx, (img_path, true_cid) in enumerate(samples):
    true_class = CLASSES[true_cid]
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    content = ""
    for attempt in range(3):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT, "images": [img_b64]}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 32},
            }, timeout=60)
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip().lower()
            break
        except Exception as e:
            print(f"  Error [{img_path}]: {e}", flush=True)
            if attempt < 2:
                time.sleep(5)

    from utils import match_class_name
    is_correct = match_class_name(content, true_class, CLASSES)
    cls_total[true_class] += 1
    if is_correct:
        cls_correct[true_class] += 1

    if (idx + 1) % 100 == 0:
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        print(f"  {idx+1}/{n_total} ({rate:.2f}/s, {elapsed:.0f}s)", flush=True)

elapsed = time.time() - t0
results = {}
tot_correct = 0
tot_count = 0
for c in CLASSES:
    n = cls_total[c]
    nc = cls_correct[c]
    acc = round(100.0 * nc / n, 2) if n > 0 else 0.0
    results[c] = {"correct": nc, "total": n, "acc": acc}
    tot_correct += nc
    tot_count += n
    print(f"  {c:28s}  n={n:5d}  {acc:6.2f}%")

overall = round(100.0 * tot_correct / tot_count, 2)
print(f"Overall: {overall}% ({tot_correct}/{tot_count}, {elapsed:.0f}s)")

out_dir = PROJ / "results" / "01_core" / "mllm_image_level"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "image_level_qwen3_6_35b-a3b.json"
with open(out_path, "w") as f:
    json.dump({"TeacherBehavior": {"overall_acc": overall, "per_class": results}}, f, indent=2)
print(f"Saved to {out_path}")
