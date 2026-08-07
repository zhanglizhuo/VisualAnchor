#!/usr/bin/env python3
"""
mllm_image_level_scb5.py

Image-level MLLM zero-shot classification on SCB5 val set.

Purpose: Compare image-level MLLM accuracy (this script) with the existing
bbox-level accuracy (from llm-annotation project) to verify that the
granularity mismatch does not affect the AnchorScore–MLLM correlation.

Protocol (identical to what the paper describes):
  - Full val images (no bbox cropping)
  - Ollama /api/chat endpoint
  - Prompt: "Classify the classroom activity in this image. Choose exactly one
    category from the list: {classes}. Output only the category name, nothing else."
  - Raw image bytes (no resize)
  - temperature=0, num_predict=32, think=False
  - Parser: label.lower() in response (recall-based, same as scb5_llm_ollama.py)

Usage:
  python experiments/01_core/mllm_image_level_scb5.py --model qwen3.5:27b
  python experiments/01_core/mllm_image_level_scb5.py --model qwen3.5:27b --workers 4
"""

import os, json, time, base64, argparse, logging, threading, queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests

PROJ = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import CLASS_NAME_MAP, read_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OLLAMA_URLS = (os.environ.get("OLLAMA_URLS") or
               "http://localhost:11434/api/chat").split(",")
_url_lock = threading.Lock()
_url_counter = 0

def _next_url():
    global _url_counter
    with _url_lock:
        url = OLLAMA_URLS[_url_counter % len(OLLAMA_URLS)]
        _url_counter += 1
    return url

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

PROMPT_TEMPLATE = (
    "Classify the classroom activity in this image. "
    "Choose exactly one category from the list: {classes}. "
    "Output only the category name, nothing else."
)


def collect_samples(ds_name, data_root):
    """Collect all (image_path, class_id) pairs from val split."""
    cfg = DATASET_CFG[ds_name]
    base = Path(data_root) / cfg["dir"]
    sub = cfg["subdir"]
    if sub and (base / sub).exists():
        img_dir = base / sub / "images" / "val"
        lbl_dir = base / sub / "labels" / "val"
    else:
        img_dir = base / "images" / "val"
        lbl_dir = base / "labels" / "val"

    if not img_dir.exists():
        logger.warning(f"  {ds_name}: {img_dir} not found")
        return []

    classes = cfg["classes"]
    samples = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        lbl_path = lbl_dir / (os.path.splitext(fname)[0] + ".txt")
        if not lbl_path.exists():
            continue
        lbl = read_label(lbl_path)
        if lbl is not None and 0 <= lbl < len(classes):
            samples.append((str(img_dir / fname), lbl))
    return samples


# Response normalization before class matching: the models frequently emit
# surface variants that never substring-match the canonical class names
# (e.g. "guiding students in classroom" for guide, "on stage interaction"
# for On-stage interaction). Normalizing these is free accuracy — the raw
# response is preserved in the jsonl, only the parsed label changes.
PARSE_SYNONYMS = {
    "guiding": "guide",
    "on stage": "on-stage interaction",
}


def parse_predicted_class(response, all_classes):
    """Parse the predicted class from an MLLM response (longest-match rule).

    Mirrors utils.match_class_name but returns the matched class name
    (or None if no valid class appears in the response). The longest-match
    rule disambiguates substring conflicts (e.g. blackBoard vs
    blackboard-writing). Surface variants (see PARSE_SYNONYMS) are
    normalized first.
    """
    resp = response.lower().strip().rstrip(".,;:!?\n\t ")
    for src, dst in PARSE_SYNONYMS.items():
        resp = resp.replace(src, dst)
    valid = [c.lower() for c in all_classes]
    matched = [c for c in valid if c in resp]
    if matched:
        return max(matched, key=len)
    return None


def classify_image(model, img_path, prompt, max_retries=2):
    """Send raw image bytes to Ollama, return lowercased response."""
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    for attempt in range(max_retries + 1):
        try:
            url = _next_url()
            resp = requests.post(url, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 8},
            }, timeout=120)
            if not resp.ok:
                logger.warning(f"  Ollama {resp.status_code}: {resp.text[:200]}")
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                return ""
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip().lower()
            return content
        except Exception as e:
            logger.warning(f"  classify_image error (attempt {attempt+1}): {e}")
            if attempt < max_retries:
                time.sleep(2)
            else:
                return ""


def scan_pred_jsonl(pred_path):
    """Scan the per-image jsonl: done_fnames (all entries, incl. null preds),
    ok_fnames (non-null preds), and per-class ok counts.

    The jsonl is the ground truth for resume state; the checkpoint is a
    convenience cache rebuilt by --clean and updated during runs.
    """
    done_fnames, ok_fnames = set(), set()
    cls_ok = {}
    if pred_path.exists():
        for line in pred_path.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            fname, tc, pc = d.get("fname"), d.get("true_class"), d.get("pred_class")
            if not fname or not tc:
                continue
            done_fnames.add(fname)
            if pc is not None:
                ok_fnames.add(fname)
                cls_ok[tc] = cls_ok.get(tc, 0) + 1
    return done_fnames, ok_fnames, cls_ok


def clean_dataset(model, ds_name, out_dir):
    """Dedupe the jsonl (best-result per image: prefer non-null pred, then
    most recent), rebuild the checkpoint from it, and print per-class status.

    Returns True if the dataset already satisfies the per-class target.
    """
    cfg = DATASET_CFG[ds_name]
    classes = cfg["classes"]
    model_tag = model.replace(":", "_").replace(".", "_")
    ckpt_path = out_dir / f"ckpt_{model_tag}_{ds_name}.json"
    pred_path = out_dir / f"predictions_{model_tag}_{ds_name}.jsonl"

    best = {}  # fname -> record (any ok beats any null; else keep most recent)
    if pred_path.exists():
        for line in pred_path.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            fname = d.get("fname")
            if not fname:
                continue
            cur = best.get(fname)
            if cur is None:
                best[fname] = d
            elif d.get("pred_class") is not None:
                best[fname] = d  # ok replaces null or older ok (most recent wins)
            # else: d is null and cur is ok -> keep cur (ok beats null)
        lines = [json.dumps(best[f]) + "\n" for f in sorted(best)]
        if lines:
            pred_path.write_text("".join(lines))
        logger.info(f"  {ds_name}: jsonl deduped -> {len(best)} unique images")

    done_fnames, ok_fnames, cls_ok = scan_pred_jsonl(pred_path)
    ckpt = {
        "done_fnames": sorted(done_fnames),
        "cls_ok": cls_ok,
        "cls_correct": {c: 0 for c in classes},
        "cls_total": {c: 0 for c in classes},
    }
    ckpt_path.write_text(json.dumps(ckpt, indent=2))

    total_ok = sum(cls_ok.get(c, 0) for c in classes)
    logger.info(f"  {ds_name}: {len(done_fnames)} done, {total_ok} ok (non-null)")
    for c in classes:
        logger.info(f"    {c:28s} ok={cls_ok.get(c, 0):4d}")
    return cls_ok


class ThreadSafeWriter:
    """Append-only jsonl writer safe for concurrent workers.

    Workers persist each prediction directly (lock + flush + fsync), so data
    survives even if the main thread stalls — persistence never depends on
    the main loop (unlike as_completed-based writes).
    """

    def __init__(self, path):
        self.fp = open(path, "a")
        self.lock = threading.Lock()

    def append(self, obj):
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self.lock:
            self.fp.write(line)
            self.fp.flush()
            os.fsync(self.fp.fileno())

    def close(self):
        with self.lock:
            self.fp.close()


# Classes whose images are cheap for Ollama (fast, reliable) run first, so
# the jsonl starts growing immediately; slow classes (guide, blackBoard) are
# deferred to the end of the queue.
EASY_CLASS_FIRST = [
    "teacher", "answer", "stand", "blackboard-writing", "screen",
    "On-stage interaction", "guide", "blackBoard",
]


def _ordered_classes(classes):
    return [c for c in EASY_CLASS_FIRST if c in classes] + \
           [c for c in classes if c not in EASY_CLASS_FIRST]


def run_dataset(model, ds_name, data_root, workers, out_dir, limit=None,
                per_class_limit=None, target_classes=None):
    """Run MLLM on val images of one dataset.

    With per_class_limit=L: for each class, only images still missing to reach
    L *successful (non-null)* predictions are run, in deterministic fname
    order, plus a 25% safety buffer (failed/timeout images are not retried —
    untried images are plentiful and cheaper than retries).
    """
    cfg = DATASET_CFG[ds_name]
    classes = cfg["classes"]
    samples = collect_samples(ds_name, data_root)
    n_total = len(samples)
    logger.info(f"  {ds_name}: {n_total} images")

    model_tag = model.replace(":", "_").replace(".", "_")
    ckpt_path = out_dir / f"ckpt_{model_tag}_{ds_name}.json"
    pred_path = out_dir / f"predictions_{model_tag}_{ds_name}.jsonl"

    # Resume state always reconciled against the jsonl (ground truth).
    done_fnames, ok_fnames, cls_ok = scan_pred_jsonl(pred_path)
    logger.info(f"  {ds_name}: resuming ({len(done_fnames)} done, "
                f"{sum(cls_ok.values())} ok)")

    cls_total = {c: 0 for c in classes}
    cls_correct = {c: 0 for c in classes}

    if per_class_limit:
        remaining = []
        for c in _ordered_classes(classes):
            have = cls_ok.get(c, 0)
            need = max(0, per_class_limit - have)
            if need > 0:
                need = int(need * 1.25) + 2  # buffer for timeout/failed preds
            cands = sorted(
                img for img, cid in samples
                if cid == classes.index(c) and Path(img).name not in done_fnames
            )[:need]
            if len(cands) < need:
                logger.warning(f"    {c}: only {len(cands)} untried images "
                               f"left (ok={have}, want {per_class_limit})")
            remaining.extend((img, classes.index(c)) for img in cands)
    else:
        remaining = [(img, cid) for img, cid in samples if Path(img).name not in done_fnames]
        if limit:
            remaining = remaining[:limit]

    n_total = len(done_fnames) + len(remaining)
    n_remaining = len(remaining)

    if n_remaining == 0:
        elapsed = 0
    else:
        classes_str = ", ".join(CLASS_NAME_MAP.get(c, c) for c in classes)
        prompt = PROMPT_TEMPLATE.format(classes=classes_str)
        t0 = time.time()

        # Append-only per-image prediction log (for confusion-matrix analyses).
        # Workers write directly (ThreadSafeWriter) so persistence is
        # decoupled from the main thread; the checkpoint is rebuilt from the
        # jsonl on resume.
        writer = ThreadSafeWriter(pred_path)
        results_q = queue.Queue()

        # Watchdog: if the jsonl stops growing for STALL_SECONDS, something is
        # wrong (e.g. poisoned Ollama request queue); exit hard so the runner
        # relaunches and resumes. Prevents silent infinite hangs.
        STALL_SECONDS = 300
        WATCHDOG_POLL = 30

        def _watchdog(start_mark):
            last_growth = time.time()
            while True:
                time.sleep(WATCHDOG_POLL)
                try:
                    n = sum(1 for _ in open(pred_path, "rb"))
                except OSError:
                    n = -1
                if n > start_mark:
                    start_mark = n
                    last_growth = time.time()
                elif time.time() - last_growth >= STALL_SECONDS:
                    logger.error(
                        f"  WATCHDOG: no jsonl growth for {STALL_SECONDS}s "
                        f"({start_mark} -> {n}); exiting to force relaunch")
                    os._exit(1)

        try:
            start_mark = sum(1 for _ in open(pred_path, "rb")) if pred_path.exists() else 0
        except OSError:
            start_mark = 0
        threading.Thread(target=_watchdog, args=(start_mark,), daemon=True).start()

        # Diagnostic: log latency of the first requests of each attempt so a
        # wedged pipeline is visible in the log instead of silent.
        req_log_lock = threading.Lock()
        req_logged = {"n": 0}

        def process(img_path, true_cid):
            true_class = classes[true_cid]
            fname = Path(img_path).name
            t_req = time.time()
            content = classify_image(model, img_path, prompt)
            dt = time.time() - t_req
            with req_log_lock:
                if req_logged["n"] < 10:
                    req_logged["n"] += 1
                    logger.info(f"    req#{req_logged['n']} {fname} "
                                f"({dt:.1f}s) -> {content[:40]!r}")
            pred_class = parse_predicted_class(content, classes)
            writer.append({
                "fname": fname,
                "true_class": true_class,
                "pred_class": pred_class,
                "response": content,
            })
            results_q.put((true_class, pred_class, fname))
            return true_class

        completed_fnames = set()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for img, cid in remaining:
                executor.submit(process, img, cid)
            n_done = 0
            while n_done < len(remaining):
                try:
                    true_class, pred_class, fname = results_q.get(timeout=1.0)
                except queue.Empty:
                    continue
                n_done += 1
                completed_fnames.add(fname)
                cls_total[true_class] += 1
                if pred_class is not None and pred_class == true_class.lower():
                    cls_correct[true_class] += 1
                if pred_class is not None:
                    cls_ok[true_class] = cls_ok.get(true_class, 0) + 1
                done_so_far = len(done_fnames) + n_done
                if done_so_far % 10 == 0 or done_so_far == n_total:
                    elapsed = time.time() - t0
                    rate = done_so_far / elapsed if elapsed > 0 else 0
                    logger.info(f"    {ds_name}: {done_so_far}/{n_total} ({rate:.1f}/s)")
                    ckpt = {
                        "done_fnames": sorted(done_fnames | completed_fnames),
                        "cls_ok": cls_ok,
                        "cls_correct": cls_correct,
                        "cls_total": cls_total,
                    }
                    ckpt_path.write_text(json.dumps(ckpt, indent=2))
        writer.close()

    # Final per-class report
    elapsed = time.time() - t0 if n_remaining > 0 else 0
    per_class = {}
    tot_correct = 0
    tot_count = 0
    for c in classes:
        n = cls_total[c]
        nc = cls_correct[c]
        acc = round(100.0 * nc / n, 2) if n > 0 else 0.0
        per_class[c] = {"correct": nc, "total": n, "acc": acc}
        tot_correct += nc
        tot_count += n
        logger.info(f"    {c:28s}  n={n:5d}  {acc:6.2f}%")

    overall = round(100.0 * tot_correct / tot_count, 2) if tot_count > 0 else 0
    logger.info(f"  {ds_name} overall: {overall}% ({tot_correct}/{tot_count}, {elapsed:.0f}s)")
    n_pred_records = sum(1 for _ in open(pred_path)) if pred_path.exists() else 0
    return {"overall_acc": overall, "per_class": per_class,
            "predictions_path": str(pred_path), "n_predictions": n_pred_records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5:27b", help="Ollama model name")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--datasets", nargs="*", default=None, help="Subset of datasets")
    parser.add_argument("--limit", type=int, default=None, help="Cap images per dataset (smoke test)")
    parser.add_argument("--per-class-limit", type=int, default=None,
                        help="Stratified subsample: run only images still needed to reach "
                             "N successful (non-null) predictions per class, deterministic "
                             "fname order, 25% buffer for timeouts")
    parser.add_argument("--clean", action="store_true",
                        help="Dedupe jsonl (best-result per image), rebuild checkpoint, "
                             "print per-class status; exit 0 if per-class target already met")
    args = parser.parse_args()

    data_root = args.data_root or os.environ.get("SCB5_DATA_ROOT") or str(PROJ / "data" / "scb5")
    model_tag = args.model.replace(":", "_").replace(".", "_")
    out_dir = Path(args.out or PROJ / "results" / "01_core" / "mllm_image_level")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"image_level_{model_tag}.json"

    ds_list = args.datasets or list(DATASET_CFG.keys())

    logger.info(f"Model: {args.model}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Datasets: {ds_list}")
    logger.info(f"Workers: {args.workers}")
    if args.clean:
        logger.info("Mode: --clean (dedupe jsonl + rebuild checkpoint + status)")
        all_done = True
        for ds_name in ds_list:
            logger.info(f"\n{'='*60}\n{ds_name} [clean]")
            cls_ok = clean_dataset(args.model, ds_name, out_dir)
            if args.per_class_limit:
                classes = DATASET_CFG[ds_name]["classes"]
                done = all(cls_ok.get(c, 0) >= args.per_class_limit for c in classes)
                all_done = all_done and done
                if not done:
                    missing = {c: max(0, args.per_class_limit - cls_ok.get(c, 0))
                               for c in classes if cls_ok.get(c, 0) < args.per_class_limit}
                    logger.info(f"  {ds_name}: STILL MISSING {missing}")
            else:
                all_done = False
        logger.info(f"\nCLEAN_RESULT all_done={all_done}")
        sys.exit(0 if all_done else 1)

    results = {}
    for ds_name in ds_list:
        logger.info(f"\n{'='*60}\n{ds_name}")
        results[ds_name] = run_dataset(
            args.model, ds_name, data_root, args.workers, out_dir,
            limit=args.limit, per_class_limit=args.per_class_limit,
        )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
