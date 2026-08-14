"""
realized_routing_v100_v2.py — R1 realized routing, NAME-REPLY protocol (v3: full-frame fallback, no bbox crop).

v1 used the companion's index-reply prompt; on the Ollama-served qwen3.5:27b
the index replies proved unreliable (teacher -> '0' collapse, out-of-range
'8' replies), so v2 switches to the category-name protocol already validated
on this backend by prompt_optimization_v3.py. Everything else is unchanged:
bbox crop (first person box + 5% margin), tau=45 routed arm + 30/class calib
arm, 4 per-GPU Ollama shards with resume.
"""
import argparse
import base64
import io
import json
import os
import re
import subprocess
import signal
import atexit
import time
import threading
from pathlib import Path

import requests
from PIL import Image

HOME = Path.home()
DATA_ROOT = HOME / "works" / "lizhuo" / "AutoResearchClaw" / "datasets_scb"
TB_SUB = Path("SCB5_TeacherBehavior") / "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2"
IMG_DIR = DATA_ROOT / TB_SUB / "images" / "val"
LBL_DIR = DATA_ROOT / TB_SUB / "labels" / "val"
OUT_DIR = Path(__file__).resolve().parent / "realized_routing_out_v3"
OLLAMA_BIN = "/usr/local/bin/ollama"
MODEL = "qwen3.5:27b"
CROP_MARGIN = 0.05
PORTS = [11437, 11438, 11439, 11440]
N_GPU = 4

# display names as shown in the prompt (from prompt_optimization_v3.py)
CLASS_DISPLAY = {
    "guide": "guiding students",
    "answer": "answering questions",
    "on-stage interaction": "on-stage interaction",
    "blackboard-writing": "writing on blackboard",
    "teacher": "teacher at the front",
    "stand": "standing",
    "screen": "looking at screen",
    "blackboard": "near a blackboard",
}


def build_prompt(classes):
    options = ", ".join(CLASS_DISPLAY[c] for c in classes)
    return (
        "Classify the classroom activity in this image. "
        "Choose exactly one category from the list: "
        f"{options}. Output only the category name, nothing else."
    )


def read_first_bbox(lbl_path, W, H):
    with open(lbl_path) as f:
        line = f.readline().strip()
    if not line:
        return None
    parts = line.split()
    cx, cy, w, h = map(float, parts[1:5])
    x1 = max(0, (cx - w / 2 - CROP_MARGIN) * W)
    y1 = max(0, (cy - h / 2 - CROP_MARGIN) * H)
    x2 = min(W, (cx + w / 2 + CROP_MARGIN) * W)
    y2 = min(H, (cy + h / 2 + CROP_MARGIN) * H)
    return (x1, y1, x2, y2)


def crop_to_b64(img_path):
    img = Image.open(img_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def infer_pred_from_text(raw, classes):
    text = raw.lower().strip()
    if not text:
        return None
    for idx, cls_name in enumerate(classes):
        display = CLASS_DISPLAY[cls_name].lower()
        tokens = [re.escape(tok) for tok in re.split(r"[-\s]+", display) if tok]
        if not tokens:
            continue
        pattern = r"\b" + r"[-\s]+".join(tokens) + r"\b"
        if re.search(pattern, text):
            return idx
    return None


servers = []


def start_ollama(gpu_id, port):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    env["OLLAMA_MODELS"] = "/usr/share/ollama/.ollama/models"
    env["OLLAMA_KEEP_ALIVE"] = "10m"
    proc = subprocess.Popen([OLLAMA_BIN, "serve"], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    servers.append(proc)
    for _ in range(60):
        try:
            if requests.get(f"http://127.0.0.1:{port}/api/tags", timeout=2).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        raise RuntimeError(f"Ollama GPU{gpu_id} port {port} failed")
    r = requests.post(f"http://127.0.0.1:{port}/api/generate",
                      json={"model": MODEL, "prompt": "warmup", "stream": False,
                            "options": {"temperature": 0, "num_predict": 1}}, timeout=600)
    r.raise_for_status()
    print(f"[GPU{gpu_id}] port {port} ready ({r.elapsed.total_seconds():.0f}s load)", flush=True)


def cleanup():
    for p in servers:
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


atexit.register(cleanup)
signal.signal(signal.SIGTERM, lambda *_: cleanup())
signal.signal(signal.SIGINT, lambda *_: cleanup())


def classify(port, prompt, img_b64, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(f"http://127.0.0.1:{port}/api/chat", json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
                "stream": False, "think": False,
                "options": {"temperature": 0, "num_predict": 16},
            }, timeout=180)
            return resp.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            print(f"  retry {attempt+1}: {e}", flush=True)
            if attempt < max_retries:
                time.sleep(3)
    return ""


def run_shard(shard_id, items, prompt):
    out_path = OUT_DIR / f"shard_{shard_id}.jsonl"
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for ln in f:
                try:
                    done.add(json.loads(ln)["f"])
                except Exception:
                    pass
    port = PORTS[shard_id % N_GPU]
    todo = [it for it in items if it["f"] not in done]
    print(f"[shard {shard_id}] {len(todo)}/{len(items)} to run (port {port})", flush=True)
    t0 = time.time()
    with open(out_path, "a") as f:
        for k, it in enumerate(todo):
            img_path = IMG_DIR / it["f"]
            try:
                b64 = crop_to_b64(img_path)
                raw = classify(port, prompt, b64)
                pred = infer_pred_from_text(raw, CLASSES)
            except Exception as e:
                pred = None
                raw = f"ERROR: {e}"
            rec = {"f": it["f"], "arm": it.get("arm"), "true": it["true"], "pred": pred,
                   "correct": (pred == it["true"]) if pred is not None else None, "raw": raw[:60]}
            f.write(json.dumps(rec) + "\n")
            if (k + 1) % 25 == 0:
                f.flush()
                el = time.time() - t0
                print(f"[shard {shard_id}] {k+1}/{len(todo)} ({el/60:.1f}m, {(k+1)/el:.2f} img/s)", flush=True)
    print(f"[shard {shard_id}] DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    manifest = json.load(open(args.manifest))
    global CLASSES
    CLASSES = manifest["classes"]
    prompt = build_prompt(CLASSES)
    print("PROMPT:", prompt, flush=True)

    items = []
    for it in manifest["routed"]:
        items.append({"f": it["f"], "true": it["true"], "arm": "routed"})
    for it in manifest["calib"]:
        items.append({"f": it["f"], "true": it["true"], "arm": "calib"})
    items.sort(key=lambda x: x["f"])

    OUT_DIR.mkdir(exist_ok=True)
    for gpu_id, port in enumerate(PORTS):
        start_ollama(gpu_id, port)

    shards = [[] for _ in range(N_GPU)]
    for i, it in enumerate(items):
        shards[i % N_GPU].append(it)
    threads = []
    for sid, sh in enumerate(shards):
        t = threading.Thread(target=run_shard, args=(sid, sh, prompt))
        t.start()
        threads.append(t)
        time.sleep(2)
    for t in threads:
        t.join()
    print("ALL SHARDS DONE", flush=True)


if __name__ == "__main__":
    main()
