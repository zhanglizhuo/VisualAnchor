# AnchorScore: A CLIP-Based Diagnostic of MLLM Annotation Difficulty

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2608.16690-b31b1b.svg)](https://arxiv.org/abs/2608.16690)
[![Status: Under review](https://img.shields.io/badge/Status-Under%20review-important.svg)]()

MLLMs are increasingly used for automated image annotation, but their per-class accuracy varies
widely and is expensive to measure: evaluating one 27B MLLM on 5,416 validation images takes
roughly 14 hours (4 GPUs), while CLIP inference completes in 3 minutes. **AnchorScore** — the
per-class zero-shot accuracy of a frozen CLIP model —
flags the classes MLLMs are least likely to annotate reliably, at a fraction of the cost.

![AnchorScore concept: cost asymmetry and per-class difficulty](paper/figures/fig1_concept.png)

## Key Results

| Result | Value | Evidence |
|--------|-------|----------|
| Class-level correlation (SCB5, 13 classes, 6 MLLMs) | **ρ = 0.769** (p = 0.002) | `results/01_core/correlation/unified_results.json` |
| Replication (Stanford40 Actions, 40 classes) | **ρ = 0.817** (p < 0.001) | `results/02_robustness/stanford40/` |
| Three-study meta-analysis (activity recognition) | **ρ = 0.781**, 95% CI [0.653, 0.865], I² = 3.0% | `results/05_applications/meta_analysis_results.json` (`subgroup_all_scene`) |
| Cross-domain validation (4 datasets, pooled per-class) | **ρ = 0.462** (p = 0.006, n = 34) | `results/01_core/correlation/cross_domain_mllm_subset.json` |
| Deployable hybrid CLIP/MLLM routing | up to **+23.3 pp** over CLIP-only at 43.5% MLLM cost saving (TeacherBehavior); +21.6 pp at 60.8% (BowTurnHead); +1.4 pp at 76.8% (Stanford40); none on HandriseReadWrite. Realized runs (Qwen3.5-27B): **+21.7 pp** (canonical bbox protocol), **+17.5 pp** (full-frame deployment condition) | `results/05_applications/hybrid/hybrid_deployable.json`, `results/05_applications/hybrid/realized_routing.json`, `results/05_applications/hybrid/realized_routing_fullframe.json` |
| Review-priority ranking (Stanford40) | **AUC = 0.842** (permutation p < 0.001) | `results/05_applications/ranking/stanford40_ranking.json` |
| Prompt disambiguation (exploratory) | +25.0 pp mean with direct visual-distinction prompts (3/10 significant) | `results/04_ablation/prompt_optimization/prompt_opt_v3_bootstrap.json` |
| SigLIP / DINOv2 / ResNet-50 baselines | no significant class-level correlation (ρ ≤ 0.201) | `results/01_core/anchor_score_scb5/{siglip,dinov2}_correlation.json`, `results/03_baselines/resnet50_baseline/resnet50_results.json` |

The signal is most consistent with a **shared class-difficulty factor**: a cross-model consensus control shows AnchorScore tracks difficulty shared across CLIP and MLLMs rather than a CLIP-specific mechanism, and its value lies in being a low-compute, label-seeded cold-start proxy of that shared factor (no MLLM inference; only a small labeled validation set). It supports *ranking* (which classes are relatively harder) rather than *calibration* (exact accuracy estimates); full caveats in the paper.

![Per-dataset correlation between AnchorScore and MLLM accuracy](paper/figures/per_dataset_correlation.png)

## Pipeline

![AnchorScore pipeline](paper/figures/fig5_pipeline.png)

**(A)** Per-class images + domain prompts → frozen CLIP (428M, ~3 min) → AnchorScore (per-class
zero-shot accuracy). **(B)** Downstream applications: (i) *hybrid routing* — each image is routed by
CLIP's predicted class: high-AnchorScore predictions are accepted from CLIP, low-AnchorScore ones
go to the MLLM (up to +23 pp over CLIP-only, e.g., +23 pp at ~44% cost savings on TeacherBehavior); (ii) *prompt
disambiguation* — CLIP confusion matrix guides MLLM prompt refinements (+25.0 pp mean with direct
visual-distinction descriptions, exploratory); (iii) *review priority* — AnchorScore ranking predicts which classes most need human
verification.

## Quick Start (Analysis Only)

Lightweight: no GPU, no raw data. All analysis scripts read pre-generated result JSONs committed
in `results/`.

```bash
pip install scipy numpy
python analysis/01_core/unified_correlation.py         # Main ρ, scatter plots
python analysis/01_core/pooled_class_level_correlation.py
python analysis/02_robustness/multi_backbone_correlation.py
python analysis/01_core/class_level_correlation.py
python analysis/03_baselines/baselines_correlation.py
```

## Full Reproduction

### Environment

```bash
# Analysis-only mode (no GPU, no raw data): only scipy + numpy needed
pip install scipy numpy

# Full reproduction (GPU):
pip install -r requirements.txt
```

Version notes (see `requirements.txt`):

- `open-clip-torch` is pinned to **2.32.0** with `transformers<5.0` — open_clip's T5 tokenizer path breaks under transformers 5.x.
- ViT-L/14 inference runs in FP16 (hard-coded `model.half()` in the scripts); FP32 requires ~41 GB and OOMs on 40 GB GPUs (FP16 ≈ 17 GB at batch 64).

### Data Dependencies

| Data | Source | Needed By |
|------|--------|-----------|
| SCB5 (13 classes, ~16k images) | [`wintonYF/SCB-Dataset`](https://huggingface.co/wintonYF/SCB-Dataset) (see `experiments/01_core/download_scb5.py`) | `anchor_scb5`, multi-backbone, active learning |
| SCB5-LLM-202506 (10 cls, 494 imgs) | `wintonYF/SCB-Dataset` | SCB5-LLM experiments |
| EuroSAT RGB | `torchvision.datasets.EuroSAT` (auto) | cross-domain |
| MedMNIST (Blood/Tissue/PathMNIST) | `medmnist` (auto) | cross-domain |
| Stanford40 Actions (40 classes, 9,532 images) | [Stanford40](https://vision.stanford.edu/Datasets/40actions.html) (images under `JPEGImages/`; see `experiments/02_robustness/stanford40_clip_per_image.py`) | `anchor_stanford40`, `stanford40_clip_per_image`, hybrid routing |

Trained checkpoints (e.g., the ResNet-50 baseline) are regenerated by their generating scripts (`experiments/03_baselines/resnet50_baseline_scb5.py`) and are not stored in the repository.

Custom data paths:

- `SCB5_DATA_ROOT` — SCB5 root directory
- `STANFORD40_DIR` — Stanford40 root directory (containing `JPEGImages/`)
- `HF_ENDPOINT` — Hugging Face mirror (e.g. `https://hf-mirror.com` in China)

### Experiment Pipeline (Ordered)

```bash
# Step 0: Download SCB5 data (~5.4GB)
python experiments/01_core/download_scb5.py --output_dir datasets_scb
export SCB5_DATA_ROOT=$(pwd)/datasets_scb

# Step 1: Core AnchorScore
python experiments/01_core/anchor_scb5.py

# Step 1b: MLLM annotation (optional, requires a GPU server running Ollama)
# The paper's MLLM accuracy values are also committed as
# results/01_core/paper_data/mllm_full.json, so the analysis-only mode
# above already reproduces the headline ρ without this step.
python experiments/01_core/llava_scb5.py            # LLaVA models (GPU)
python experiments/01_core/scb5_llm_ollama.py       # Ollama VLMs on SCB-LLM (~4 h per model on a 32 GB GPU; see file header)

# Step 2: Downstream analyses
python experiments/02_robustness/anchor_cross_domain.py
python experiments/02_robustness/anchor_scb5_multi_backbone.py
python experiments/05_applications/active_learning_naive.py
python experiments/05_applications/active_learning_capped_b32.py
python experiments/05_applications/active_learning_capped_l14.py
python experiments/04_ablation/prompt_robustness_ablation.py
python experiments/03_baselines/siglip_anchor_scb5.py
python experiments/03_baselines/dinov2_anchor_scb5.py
python experiments/03_baselines/resnet50_baseline_scb5.py
python experiments/03_baselines/vl_baseline_correlate.py --vl-results results/01_core/anchor_score_scb5/siglip_anchor.json

# Step 3: Generate analysis figures and tables
python analysis/01_core/unified_correlation.py
```

### Data Flow

```
experiments/*/*.py  ──(json.dump)──>  results/*.json  ←── analysis/*/*.py
```

- Experiments read raw data, compute results, write to `results/`.
- Analysis scripts read only from `results/`, never from raw data or logs.
- Logs in `logs/` are optional stdout captures (not committed, not consumed by any script).

### Outputs

All paper figures and tables come from:

1. Analysis scripts → summarized JSON in `results/`
2. `paper/generate_figures.py` → `.png` figures in `paper/figures/`
3. LaTeX compilation → `paper/VisualAnchor.pdf`

## Project Structure

```
VisualAnchor/
├── experiments/{01_core,02_robustness,03_baselines,04_ablation,05_applications}/
├── analysis/{01_core,02_robustness,03_baselines,04_ablation,05_applications,06_consensus_control,07_mechanism}/
├── paper/                # LaTeX / paper source (journal version; paper/arxiv/ = arXiv preprint source)
├── results/              # All experiment outputs (tracked in git)
├── logs/                 # Execution logs (gitignored)
└── data/                 # Dataset configs + small auxiliary data
```

`experiments/` and `analysis/` share the same phase layout (`01_core` → `05_applications`), with two
additional analysis-only phases: `06_consensus_control` (cross-model shared-difficulty control) and
`07_mechanism` (confusion-structure transitivity test).

## Experiment Groups

All paths below are relative to the respective directory (e.g. `01_core/anchor_scb5.py` means
`experiments/01_core/anchor_scb5.py`).

| Phase | Scripts |
|-------|---------|
| **01_core** — AnchorScore + primary MLLM correlation | `anchor_scb5.py`, `clip_scb5_predictions.py`, `download_scb5.py`, `llava_scb5.py`, `scb5_llm_ollama.py` |
| **02_robustness** — cross-domain, multi-backbone, Stanford40 | `anchor_cross_domain.py`, `anchor_scb5_multi_backbone.py`, `anchor_stanford40.py`, `cross_domain_{probe,qwen,llava,medclip,self_uncertainty}.py`, `validation_size_ablation.py` |
| **03_baselines** — non-CLIP vision models | `siglip_anchor_scb5.py`, `dinov2_anchor_scb5.py`, `resnet50_baseline_scb5.py`, `blip2_anchor_scb5.py`, `vl_baseline_correlate.py` |
| **04_ablation** — prompt robustness, self-uncertainty | `prompt_robustness_ablation.py`, `prompt_optimization.py`, `self_uncertainty_scb5.py` |
| **05_applications** — active learning, hybrid annotation | `active_learning_{naive,capped_b32,capped_l14}.py`, `hybrid_annotation.py`, `run_prompt_opt_simple.py` |
| **06_consensus_control** *(analysis-only)* — cross-model shared-difficulty control | `consensus_control_exp2.py`, `exp{1,3,4,5}_*.py` |
| **07_mechanism** *(analysis-only)* — confusion-structure transitivity (Mantel test) | `confusion_transitivity.py` |

Scripts are named by research question, not execution order — each is independent and can be run
separately.

## Citation

Preprint: https://arxiv.org/abs/2608.16690 (arXiv:2608.16690 [cs.CV])

```bibtex
@misc{ma2026anchorscore,
  title={AnchorScore: A CLIP-Based Diagnostic of MLLM Annotation Difficulty},
  author={Ma, Yan and Zhang, Lizhuo},
  year={2026},
  eprint={2608.16690},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  doi={10.48550/arXiv.2608.16690}
}
```

## License

This project is released under the [MIT License](LICENSE).
