#!/usr/bin/env python3
import json
from pathlib import Path
from scipy.stats import spearmanr


def vals(accs):
    return [accs[k] for k in accs]


def main():
    RESULTS = Path(__file__).resolve().parent.parent.parent / "results"

    # SCB5_LLM CLIP AnchorScore (from anchor_scb5_llm.py output)
    with open(RESULTS / "01_core" / "scb5_llm_expansion" / "clip_anchor_fixed_prompts.json") as f:
        scb5_llm_clip = json.load(f)

    # LLaVA-1.5-7B on SCB5_LLM
    with open(RESULTS / "01_core" / "scb5_llm_expansion" / "llava_results.json") as f:
        llava_scb = {k: v["acc"] for k, v in json.load(f).items()}

    # Ollama models on SCB5_LLM
    with open(RESULTS / "01_core" / "scb5_llm_expansion" / "ollama_all_models.json") as f:
        ollama = json.load(f)
    ollama_models = list(ollama.keys())

    # Existing 13 classes (from pooled_class_level_results.json)
    with open(RESULTS / "01_core" / "correlation" / "pooled_class_level_results.json") as f:
        pooled = json.load(f)

    existing_clip = {}
    existing_6mllm = {}
    existing_llava = {}
    for entry in pooled["data"]:
        if entry["domain"].startswith("SCB5"):
            existing_clip[entry["class"]] = entry["anchor_score"]
            existing_6mllm[entry["class"]] = entry["mllm_mean"]

    # Existing LLaVA-1.5-7B (from llava_scb5 summary)
    llava_scb5_files = sorted((RESULTS / "01_core" / "llava_scb5").glob("llava15_7b_scb5_summary_*.json"))
    if not llava_scb5_files:
        raise FileNotFoundError("No llava15_7b_scb5_summary_*.json found in " + str(RESULTS / "01_core" / "llava_scb5"))
    with open(llava_scb5_files[0]) as f:
        llava_scb5_data = json.load(f)
    for ds_name, ds_data in llava_scb5_data.items():
        for cls, cls_data in ds_data.items():
            existing_llava[cls] = cls_data["acc"]

    # SCB5_LLM 5-MLLM mean
    classes_scb = list(scb5_llm_clip.keys())
    scb_5mllm = {}
    for cn in classes_scb:
        accs = [llava_scb[cn]] + [ollama[m][cn]["acc"] for m in ollama_models]
        scb_5mllm[cn] = round(sum(accs) / len(accs), 1)

    # ========== ANALYSIS ==========
    print("=" * 70)
    print("FULL CORRELATION ANALYSIS: SCB5 + SCB5_LLM")
    print("=" * 70)

    # 1. Existing 13 (6-MLLM mean)
    r_ex, p_ex = spearmanr(vals(existing_clip), vals(existing_6mllm))
    print(f"\n1. Existing SCB5 n=13 (6-MLLM): ρ={r_ex:.4f}, p={p_ex:.6f}")

    # 2. SCB5_LLM 10 (5-MLLM mean)
    r_scb, p_scb = spearmanr(vals(scb5_llm_clip), vals(scb_5mllm))
    print(f"2. SCB5_LLM n=10 (5-MLLM):  ρ={r_scb:.4f}, p={p_scb:.6f}")

    # 3. Combined 23 (6-MLLM for existing, 5-MLLM for SCB5_LLM)
    clip_comb = vals(existing_clip) + vals(scb5_llm_clip)
    mllm_comb = vals(existing_6mllm) + vals(scb_5mllm)
    r_comb, p_comb = spearmanr(clip_comb, mllm_comb)
    print(f"3. Combined n=23 (6/5-MLLM): ρ={r_comb:.4f}, p={p_comb:.6f}")

    # 4. Combined 23 (LLaVA-1.5-7B only, fair comparison)
    llava_comb = vals(existing_llava) + vals(llava_scb)
    r_l23, p_l23 = spearmanr(clip_comb, llava_comb)
    print(f"4. Combined n=23 (LLaVA only): ρ={r_l23:.4f}, p={p_l23:.6f}")

    # 5. Existing 13 (LLaVA only)
    r_ex_l, p_ex_l = spearmanr(vals(existing_clip), vals(existing_llava))
    print(f"5. Existing n=13 (LLaVA only):  ρ={r_ex_l:.4f}, p={p_ex_l:.6f}")

    # 6. SCB5_LLM 10 (LLaVA only)
    r_scb_l, p_scb_l = spearmanr(vals(scb5_llm_clip), vals(llava_scb))
    print(f"6. SCB5_LLM n=10 (LLaVA only):  ρ={r_scb_l:.4f}, p={p_scb_l:.6f}")

    # 7. Non-overlap SCB5_LLM 8 (5-MLLM)
    existing_lower = set(k.lower() for k in existing_clip)
    non_overlap = [k for k in classes_scb if k.lower() not in existing_lower]
    clip_no = [scb5_llm_clip[k] for k in non_overlap]
    mllm_no = [scb_5mllm[k] for k in non_overlap]
    r_no, p_no = spearmanr(clip_no, mllm_no)
    print(f"7. SCB5_LLM non-overlap n=8:    ρ={r_no:.4f}, p={p_no:.6f}")

    # Per-class SCB5_LLM table
    print("\n" + "=" * 70)
    print("PER-CLASS SCB5_LLM RESULTS")
    print("=" * 70)
    header = f"{'Class':35s} {'CLIP':>6s} {'LLaVA':>6s}"
    model_shorts = [m.split(":")[0].replace(".", "")[:10] for m in ollama_models]
    for s in model_shorts:
        header += f"{s:>10s}"
    header += f"{'5MLLM':>8s}"
    print(header)
    print("-" * (35 + 6 + 6 + 10*len(ollama_models) + 8))
    for cn in classes_scb:
        line = f"{cn:35s} {scb5_llm_clip[cn]:>6.1f} {llava_scb[cn]:>6.1f}"
        for m in ollama_models:
            line += f"{ollama[m][cn]['acc']:>10.1f}"
        line += f"{scb_5mllm[cn]:>8.1f}"
        print(line)

    # Existing + SCB5_LLM combined table
    print("\n" + "=" * 70)
    print("COMBINED PER-CLASS DATA (ensemble mean)")
    print("=" * 70)
    print(f"{'Class':35s} {'CLIP':>8s} {'MLLM-mean':>10s} {'Ensemble':>10s}")
    print("-" * 65)
    for cn, ap in sorted(existing_clip.items()):
        print(f"{cn:35s} {ap:>8.2f} {existing_6mllm[cn]:>10.1f} {'6-MLLM':>10s}")
    for cn in classes_scb:
        print(f"{cn:35s} {scb5_llm_clip[cn]:>8.1f} {scb_5mllm[cn]:>10.1f} {'5-MLLM':>10s}")

    # Save
    output = {
        "results": {
            "existing_13_6mllm": {"rho": round(r_ex, 4), "p": round(p_ex, 6)},
            "scb5_llm_10_5mllm": {"rho": round(r_scb, 4), "p": round(p_scb, 6)},
            "combined_23_mixed_ensemble": {"rho": round(r_comb, 4), "p": round(p_comb, 6)},
            "combined_23_llava_only": {"rho": round(r_l23, 4), "p": round(p_l23, 6)},
            "existing_13_llava_only": {"rho": round(r_ex_l, 4), "p": round(p_ex_l, 6)},
            "scb5_llm_10_llava_only": {"rho": round(r_scb_l, 4), "p": round(p_scb_l, 6)},
            "scb5_llm_8_non_overlap_5mllm": {"rho": round(r_no, 4), "p": round(p_no, 6)},
        },
        "scb5_llm_per_class": {cn: {
            "clip": scb5_llm_clip[cn], "llava": llava_scb[cn],
            "ollama": {m: ollama[m][cn]["acc"] for m in ollama_models},
            "5mllm_mean": scb_5mllm[cn]
        } for cn in classes_scb},
        "models_used": {"scb_5mllm": ["LLaVA-1.5-7B"] + ollama_models,
                        "existing_6mllm": "pooled_class_level_results.json"},
    }
    with open(RESULTS / "01_core" / "scb5_llm_expansion" / "ensemble_correlation.json", "w") as f:
        json.dump(output, f, indent=2, allow_nan=False)
    print(f"\nSaved to {RESULTS / '01_core' / 'scb5_llm_expansion' / 'ensemble_correlation.json'}")


if __name__ == "__main__":
    main()
