import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr, t as t_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 12, "figure.dpi": 300, "font.family": "serif"})

C_CLIP = "#1a9988"
C_MLLM = "#e86850"

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
ANCHOR_SCB5 = RESULTS_DIR / "01_core" / "anchor_score_scb5" / "anchor_scores.json"
MLLM_RAW = RESULTS_DIR / "01_core" / "paper_data" / "mllm_raw.json"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(ANCHOR_SCB5) as f:
    scb5 = json.load(f)

with open(MLLM_RAW) as f:
    MLLM_DATA = json.load(f)

# Filter out metadata keys (_provenance, _llava_provenance) from MLLM_DATA
MLLM_DATA = {ds: {k: v for k, v in models.items() if not k.startswith("_")}
             for ds, models in MLLM_DATA.items() if not ds.startswith("_")}

colors_ds = {"TeacherBehavior": "#E69F00", "HandriseReadWrite": "#56B4E9", "BowTurnHead": "#CC79A7"}
markers_ds = {"TeacherBehavior": "o", "HandriseReadWrite": "s", "BowTurnHead": "^"}
labels_ds = {"TeacherBehavior": "TeacherBehavior (8 cls)",
             "HandriseReadWrite": "HandriseReadWrite (3 cls)",
             "BowTurnHead": "BowTurnHead (2 cls)"}
ds_order = ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]

def add_ols_fit(ax, x_vals, y_vals):
    """Draw OLS regression line with 95% CI band on (x, y) data."""
    if len(x_vals) < 3:
        return
    ma = np.array(x_vals)
    mm = np.array(y_vals)
    slope, intercept = np.polyfit(ma, mm, 1)
    n = len(ma)
    residuals = mm - (slope * ma + intercept)
    mse = (residuals**2).sum() / (n - 2)
    x_mean = ma.mean()
    Sxx = ((ma - x_mean)**2).sum()
    x_fit = np.linspace(0, 100, 200)
    y_fit = slope * x_fit + intercept
    se_fit = np.sqrt(mse * (1 / n + (x_fit - x_mean)**2 / Sxx))
    t_val = t_dist.ppf(0.975, n - 2)
    ax.fill_between(x_fit, y_fit - t_val * se_fit, y_fit + t_val * se_fit,
                    color="#ccc", alpha=0.3, zorder=0)
    ax.plot(x_fit, y_fit, "--", color="#888", linewidth=1.3, alpha=0.6, zorder=1)

# ── Figure 1: Concept figure (cost, intuition, result preview) ──
fig1, axes1 = plt.subplots(2, 1, figsize=(7.2, 5.5),
                           gridspec_kw={"height_ratios": [1.55, 1.25], "hspace": 0.08})

# Panel A: Cost comparison
ax = axes1[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")
ax.set_facecolor("white")

left_box = mpatches.FancyBboxPatch((0.3, 1.5), 3.2, 2.0,
    boxstyle="round,pad=0.15", facecolor="#fee0d2", edgecolor=C_MLLM, linewidth=2)
ax.add_patch(left_box)
ax.text(1.9, 3.8, "Full MLLM Pipeline", ha="center", fontsize=10, fontweight="bold", color=C_MLLM)
ax.text(1.9, 3.2, "Full MLLM inference", ha="center", fontsize=8, color="#555")
ax.text(1.9, 2.7, "on 5,416 images", ha="center", fontsize=8, color="#555")
ax.text(1.9, 2.2, "~14 hours", ha="center", fontsize=11, fontweight="bold", color=C_MLLM)
ax.text(1.9, 1.7, "7B--27B params, 4 GPUs\n(Autoregressive Generation)", ha="center", fontsize=8, color="#777")

right_box = mpatches.FancyBboxPatch((6.5, 1.5), 3.2, 2.0,
    boxstyle="round,pad=0.15", facecolor="#e0f2f1", edgecolor=C_CLIP, linewidth=2)
ax.add_patch(right_box)
ax.text(8.1, 3.8, "AnchorScore", ha="center", fontsize=10, fontweight="bold", color=C_CLIP)
ax.text(8.1, 3.2, "CLIP zero-shot inference", ha="center", fontsize=8, color="#555")
ax.text(8.1, 2.7, "on 5,416 images", ha="center", fontsize=8, color="#555")
ax.text(8.1, 2.2, "~3 minutes", ha="center", fontsize=11, fontweight="bold", color=C_CLIP)
ax.text(8.1, 1.7, "428M params, single GPU\n(Feature embedding)", ha="center", fontsize=8, color="#777")

ax.annotate("", xy=(6.2, 2.5), xytext=(3.8, 2.5),
            arrowprops=dict(arrowstyle="->", color="#333", lw=3))
ax.text(5.0, 2.8, "$\\approx$270$\\times$", ha="center", fontsize=14, fontweight="bold", color="#333",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff4e6", edgecolor="#ff922b", linewidth=2))
ax.text(5.0, 3.3, "more FLOPs", ha="center", fontsize=8, color="#777")
ax.text(5.0, 1.8, "no additional training", ha="center", fontsize=8, color="#777")
ax.text(5.0, 0.5, "Input: unlabeled validation images", fontsize=8, color="#555", ha="center")
# Π-shaped arrows: horizontal segments stop at text edges
ax.plot([1.9, 2.7], [0.55, 0.55], color="#999", lw=2, solid_capstyle="round")
ax.plot([7.3, 8.1], [0.55, 0.55], color="#999", lw=2, solid_capstyle="round")
ax.annotate("", xy=(1.9, 1.2), xytext=(1.9, 0.51),
            arrowprops=dict(arrowstyle="->", color="#999", lw=2))
ax.annotate("", xy=(8.1, 1.2), xytext=(8.1, 0.51),
            arrowprops=dict(arrowstyle="->", color="#999", lw=2))
ax.text(5.0, 4.6, "(a) Cost asymmetry", fontsize=10, fontweight="bold", color="#333", ha="center", va="top")

# Panel B: Example classes
# Values computed from evidence files (no hardcoding):
# - CLIP bars: per-class zero-shot accuracy from ANCHOR_SCB5
# - MLLM bars: 6-MLLM mean accuracy from MLLM_DATA (the six canonical MLLMs)
tb_pc = scb5["TeacherBehavior"]["per_class_acc"]
tb_mllm = {cls: float(np.mean([MLLM_DATA["TeacherBehavior"][m][cls] for m in MLLM_DATA["TeacherBehavior"]]))
           for cls in ["teacher", "screen", "stand", "answer"]}
ax = axes1[1]
ax.set_xlim(0, 10)
ax.set_ylim(-5, 136)
ax.axis("off")

ax.text(5.0, 127, "(b) From high to low AnchorScore", fontsize=10, fontweight="bold", color="#333", ha="center", va="top")

examples = [
    ("teacher", tb_pc["teacher"]["acc"], tb_mllm["teacher"], "High AnchorScore", True),
    ("screen", tb_pc["screen"]["acc"], tb_mllm["screen"], "Mid-high AnchorScore", True),
    ("stand", tb_pc["stand"]["acc"], tb_mllm["stand"], "Low AnchorScore", True),
    ("answer", tb_pc["answer"]["acc"], tb_mllm["answer"], "Very low AnchorScore", True),
]
for i, (cls_name, clip_val, mllm_val, caption, consistent) in enumerate(examples):
    x_base = i * 2.5
    ax.bar(x_base + 0.2, clip_val, width=0.8, bottom=8, color=C_CLIP, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.text(x_base + 0.2, clip_val + 11, f"CLIP {clip_val:.1f}%", ha="center", fontsize=8, color=C_CLIP, fontweight="bold")
    ax.bar(x_base + 1.3, mllm_val, width=0.8, bottom=8, color=C_MLLM, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.text(x_base + 1.3, mllm_val + 11, f"MLLM {mllm_val:.1f}%", ha="center", fontsize=8, color=C_MLLM, fontweight="bold")
    ax.text(x_base + 0.75, -2, cls_name.replace("-", " ").title(), ha="center", fontsize=8, fontweight="bold", color="#bbb")

fig1.savefig(OUT_DIR / "anchorscore_concept.png", bbox_inches="tight", dpi=300, pad_inches=0.1)
fig1.savefig(OUT_DIR / "anchorscore_concept.pdf", bbox_inches="tight", pad_inches=0.1)
print(f"Saved {OUT_DIR / 'anchorscore_concept.png'}")

# ── Figure 2: Pipeline diagram (matplotlib, style-matches Fig 1) ──
fig2, ax2 = plt.subplots(figsize=(7.2, 5.5))
ax2.set_xlim(0, 12); ax2.set_ylim(-1.0, 7.5)
ax2.axis("off")

C_CLIP_LIGHT = "#e0f2f1"
C_MLLM_LIGHT = "#fee0d2"
C_ANCHOR_FILL = "#fff4e6"
C_ANCHOR_EDGE = "#ff922b"
C_GROUP_FILL = "#F8FAFC"
C_GROUP_EDGE = "#CBD5E1"
C_DATA_FILL  = "#EEEEF5"
C_DATA_EDGE  = "#7C7CB0"  # neutral slate for input / result boxes

def ov_box(x, y, w, h, text, fill, edge, fs=8, fc="#333"):
    ax2.add_patch(mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.08", facecolor=fill, edgecolor=edge, linewidth=1.5))
    ax2.text(x, y, text, ha="center", va="center", fontsize=fs, color=fc)

def ov_arrow(x1, y1, x2, y2, color="#555", lw=1.5):
    ax2.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=0, shrinkB=0))

def ov_elbow(x1, y1, x2, y2, via="h", color="#555", lw=1.5):
    """Right-angle (L-shaped) arrow. via='h': horizontal then vertical; via='v': vertical then horizontal."""
    if via == "h":
        ax2.plot([x1, x2], [y1, y1], color=color, lw=lw, solid_capstyle="round")
        ax2.annotate("", xy=(x2, y2), xytext=(x2, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=0, shrinkB=0))
    else:
        ax2.plot([x1, x1], [y1, y2], color=color, lw=lw, solid_capstyle="round")
        ax2.annotate("", xy=(x2, y2), xytext=(x1, y2),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=0, shrinkB=0))

# Group A
grp_a = mpatches.FancyBboxPatch((0.3, 4.0), 11.4, 2.2,
    boxstyle="round,pad=0.15", facecolor=C_GROUP_FILL, edgecolor=C_GROUP_EDGE, linewidth=1.2, linestyle="--")
ax2.add_patch(grp_a)
ax2.text(6.0, 6.1, "(A) AnchorScore Computation", fontsize=9, fontweight="bold", color="#374151", ha="center", va="top")

ov_box(2, 5.1, 2.6, 1.0, "Per-class images\n+ domain prompts", C_DATA_FILL, C_DATA_EDGE)
ov_box(6, 5.1, 2.6, 1.0, "CLIP ViT-L/14\nzero-shot inference", C_CLIP_LIGHT, C_CLIP)
ov_box(10, 5.1, 2.6, 1.0, "AnchorScore\n(per-class accuracy)", C_ANCHOR_FILL, C_ANCHOR_EDGE)
ov_arrow(3.38, 5.1, 4.62, 5.1)
ov_arrow(7.38, 5.1, 8.62, 5.1)

# Group B
grp_b = mpatches.FancyBboxPatch((0.3, -0.6), 11.4, 3.8,
    boxstyle="round,pad=0.15", facecolor=C_GROUP_FILL, edgecolor=C_GROUP_EDGE, linewidth=1.2, linestyle="--")
ax2.add_patch(grp_b)
ax2.text(6.0, -0.4, "(B) Downstream Applications", fontsize=9, fontweight="bold", color="#374151", ha="center", va="bottom")

# AnchorScore → dogleg to midpoint → split into both sub-flows (symmetric T-junction)
ax2.plot([10, 10], [4.52, 4.05], color=C_ANCHOR_EDGE, lw=1.2)    # down from AnchorScore (plain line)
ax2.plot([10, 6], [4.05, 4.05], color=C_ANCHOR_EDGE, lw=1.2)    # dogleg left → midpoint
ov_arrow(6, 4.05, 6, 3.5, C_ANCHOR_EDGE, 1.2)                 # dogleg down (head matches C5/C6)
ax2.plot([3.0, 8.97], [3.5, 3.5], color=C_ANCHOR_EDGE, lw=1.2) # bar (stops short of right branch)
ov_arrow(3.0, 3.5, 3.0, 2.98, C_ANCHOR_EDGE, 1.2)             # left branch
ov_arrow(9.0, 3.5, 9.0, 2.98, C_ANCHOR_EDGE, 1.2)             # right branch

# -- (i) Hybrid Annotation (3 layers, symmetric with ii) --
ax2.text(3.0, 0.1, "(i) Hybrid Annotation", fontsize=8, fontweight="bold", color="#374151", ha="center", va="center")
ov_box(3.0, 2.7, 2.0, 0.4, "Apply threshold $\\tau$\nto AnchorScore", C_ANCHOR_FILL, C_ANCHOR_EDGE, fs=7)
ov_box(1.5, 1.7, 2.0, 0.4, "High \u2192 CLIP\n(cheap)", C_CLIP_LIGHT, C_CLIP, fs=6.5)
ov_box(4.5, 1.7, 2.0, 0.4, "Low \u2192 MLLM\n(expensive)", C_MLLM_LIGHT, C_MLLM, fs=6.5)
ov_box(3.0, 0.7, 2.0, 0.4, "+3\u201311pp accuracy\n53\u201377% cost saved", C_DATA_FILL, C_DATA_EDGE, fs=6.5)
ov_elbow(1.92, 2.7, 1.5, 1.98, via="h")  # threshold left edge → High top-center
ov_elbow(4.08, 2.7, 4.5, 1.98, via="h")  # threshold right edge → Low top-center
ov_elbow(1.5, 1.42, 1.92, 0.7, via="v")  # High bottom-center → result left edge
ov_elbow(4.5, 1.42, 4.08, 0.7, via="v")  # Low bottom-center → result right edge

# -- (ii) Prompt Disambiguation (3 layers, y-aligned with i) --
ax2.text(9.0, 0.1, "(ii) Prompt Disambiguation", fontsize=8, fontweight="bold", color="#374151", ha="center", va="center")
ov_box(9.0, 2.7, 2.0, 0.4, "Confusion\nmatrix", C_CLIP_LIGHT, C_CLIP, fs=7)
ov_box(9.0, 1.7, 2.0, 0.4, "Enhanced prompts\n(disambiguation)", C_MLLM_LIGHT, C_MLLM, fs=7)
ov_box(9.0, 0.7, 2.0, 0.4, "MLLM accuracy\n+11.0pp avg", C_DATA_FILL, C_DATA_EDGE, fs=7)
ov_arrow(9.0, 2.42, 9.0, 1.98)
ov_arrow(9.0, 1.42, 9.0, 0.98)

fig2.savefig(OUT_DIR / "pipeline_overview.png", bbox_inches="tight", dpi=300, pad_inches=0.1)
fig2.savefig(OUT_DIR / "pipeline_overview.pdf", bbox_inches="tight", pad_inches=0.1)
print(f"Saved {OUT_DIR / 'pipeline_overview.png'}")

# ── Figure 3: Strip plot — class-level mean + individual MLLM runs ──
fig3, ax = plt.subplots(figsize=(7, 5))

rng = np.random.default_rng(42)
JITTER = 2.5

# Collect per-class means for trend line
all_means_A, all_means_M, all_means_name = [], [], []
n_models = len({m for ds in MLLM_DATA.values() for m in ds})

legend_added = set()
for ds_name in ds_order:
    anchor_acc = scb5[ds_name]["per_class_acc"]
    class_list = list(anchor_acc.keys())
    for cname in class_list:
        a_val = anchor_acc[cname]["acc"]
        mllm_vals = []
        for mname, mdata in MLLM_DATA.get(ds_name, {}).items():
            if cname in mdata:
                mllm_vals.append(mdata[cname])
        if not mllm_vals:
            continue
        mllm_arr = np.array(mllm_vals)
        m_std = mllm_arr.std(ddof=1) if len(mllm_arr) > 1 else 0.0
        # Individual runs (jittered, on top of mean scatter)
        jitter = rng.uniform(-JITTER, JITTER, size=len(mllm_arr))
        dl = labels_ds[ds_name]
        lbl = dl if dl not in legend_added else None
        ax.scatter(a_val + jitter, mllm_arr, alpha=0.45, s=18,
                   color=colors_ds[ds_name], marker=markers_ds[ds_name],
                   edgecolors="white", linewidth=0.5, label=lbl, zorder=2)
        legend_added.add(dl)
        # Class mean
        m_mean = mllm_arr.mean()
        ax.errorbar(a_val, m_mean, yerr=m_std, fmt="none", ecolor=colors_ds[ds_name],
                    capsize=3, capthick=1.0, linewidth=1.0, alpha=0.6, zorder=4)
        ax.scatter(a_val, m_mean, s=65, color=colors_ds[ds_name],
                   marker=markers_ds[ds_name], edgecolors="white", linewidth=0.8,
                   zorder=5)
        all_means_A.append(a_val)
        all_means_M.append(m_mean)
        all_means_name.append(cname)

means_A = np.array(all_means_A)
means_M = np.array(all_means_M)

# Linear trend on class means with 95% confidence band (visual aid for the reported ρ)
add_ols_fit(ax, means_A, means_M)

r_s_cls, p_s_cls = spearmanr(means_A, means_M)
all_indiv_A, all_indiv_M = [], []
for ds_name in ds_order:
    anchor_acc = scb5[ds_name]["per_class_acc"]
    for cname, cdata in anchor_acc.items():
        a_val = cdata["acc"]
        for mname, mdata in MLLM_DATA.get(ds_name, {}).items():
            if cname in mdata:
                all_indiv_A.append(a_val)
                all_indiv_M.append(mdata[cname])
r_s_pool, p_s_pool = spearmanr(all_indiv_A, all_indiv_M)

ax.set_xlabel("AnchorScore — CLIP Zero-Shot Accuracy (%)", fontsize=13)
ax.set_ylabel("MLLM Annotation Accuracy (%)", fontsize=13)
# Label key outlier classes
for nm, a, mmm in zip(all_means_name, all_means_A, all_means_M):
    if nm == "blackboard-writing":
        ax.annotate(nm, (a, mmm), textcoords="offset points", xytext=(6, -14),
                    fontsize=8, color="#333", zorder=6)
    elif nm == "write":
        ax.annotate(nm, (a, mmm), textcoords="offset points", xytext=(6, -14),
                    fontsize=8, color="#333", ha="left", zorder=6)
ax.legend(fontsize=9, loc="lower right", markerscale=0.9)
ax.grid(True, alpha=0.25)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

stats_text = (f"Class-level $\\rho$ = {r_s_cls:.3f}  (p = {p_s_cls:.3f}, $n=13$ classes)\n"
              f"Pooled $\\rho$ = {r_s_pool:.3f}  ($p < 0.001$, $n=78$)\n"
              f"Filled markers: class means; colors denote dataset")
ax.text(0.98, 0.35, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.90, edgecolor="#999", linewidth=0.8))

plt.tight_layout()
fig3.savefig(OUT_DIR / "fig1_correlation.png", bbox_inches="tight")
fig3.savefig(OUT_DIR / "fig1_correlation.pdf", bbox_inches="tight")
print(f"Saved {OUT_DIR / 'fig1_correlation.png'} ({len(all_indiv_A)} points, {n_models} MLLMs, class ρ={r_s_cls:.3f})")

# ── Figure A1: Per-dataset subplots (TeacherBehavior + HandriseReadWrite) ──
ds_show = ["TeacherBehavior", "HandriseReadWrite"]
fig_a1, axes = plt.subplots(1, 2, figsize=(5.6, 4.6))
for idx, ds_name in enumerate(ds_show):
    ax = axes[idx]
    anchor_acc = scb5[ds_name]["per_class_acc"]
    class_list = list(anchor_acc.keys())
    ds_a, ds_m = [], []
    means_A, means_M = [], []
    class_means = {}
    for cname in class_list:
        a_val = anchor_acc[cname]["acc"]
        vals = []
        for mname, mdata in MLLM_DATA.get(ds_name, {}).items():
            if cname in mdata:
                vals.append(mdata[cname])
                ds_a.append(a_val)
                ds_m.append(mdata[cname])
        if vals:
            m_mean = np.mean(vals)
            m_std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            class_means[cname] = (a_val, m_mean, m_std)
            means_A.append(a_val)
            means_M.append(m_mean)
    rng2 = np.random.default_rng(7)
    jitter = rng2.uniform(-1.5, 1.5, size=len(ds_a))
    ax.scatter(np.array(ds_a) + jitter, ds_m, alpha=0.45, s=18,
               color=colors_ds[ds_name], edgecolors="white", linewidth=0.5)
    # Class means + error bars + labels
    for cname, (a_mean, m_mean, m_std) in sorted(class_means.items(), key=lambda x: x[1][0]):
        ax.errorbar(a_mean, m_mean, yerr=m_std, fmt="none", ecolor=colors_ds[ds_name],
                    capsize=3, capthick=1.0, linewidth=1.0, alpha=0.6, zorder=4)
        ax.scatter(a_mean, m_mean, s=60, color=colors_ds[ds_name],
                   edgecolors="white", linewidth=0.8, zorder=5)
        if ds_name == "TeacherBehavior":
            label = "on-stage" if cname == "on-stage interaction" else cname.replace("-", " ")
            ax.annotate(label, (a_mean, m_mean), textcoords="offset points",
                        xytext=(5, -8), fontsize=12, color="#333", alpha=0.95)
        elif ds_name == "HandriseReadWrite":
            ax.annotate(cname.replace("-", " "), (a_mean, m_mean), textcoords="offset points",
                        xytext=(5, -8), fontsize=12, color="#333", alpha=0.95)
    # OLS trend on class means with CI band
    add_ols_fit(ax, means_A, means_M)
    ax.set_xlabel("AnchorScore (%)", fontsize=15)
    ax.set_ylabel("MLLM Accuracy (%)", fontsize=15)
    ax.set_title(ds_name, fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.25)
    r_s_ds, p_s_ds = spearmanr(means_A, means_M)
    k_cls = len(means_A)
    if k_cls >= 5:
        stat_txt = f"class $\\rho$={r_s_ds:.2f}, p={p_s_ds:.3f}"
    else:
        stat_txt = f"class $\\rho$={r_s_ds:.2f} (k={k_cls}, n.s.)"
    ax.text(0.95, 0.07, stat_txt,
            transform=ax.transAxes, fontsize=13, verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

plt.tight_layout()
fig_a1.savefig(OUT_DIR / "figA1_per_dataset.png", bbox_inches="tight")
fig_a1.savefig(OUT_DIR / "figA1_per_dataset.pdf", bbox_inches="tight")
print(f"Saved {OUT_DIR / 'figA1_per_dataset.png'}")

# ── Figure 4: Cross-domain AnchorScore bar chart (with per-class std error bars) ──
CROSS_DOMAIN_FILE = RESULTS_DIR / "02_robustness" / "anchor_score_cross_domain" / "anchor_scores.json"
with open(CROSS_DOMAIN_FILE) as f:
    cd_data = json.load(f)
# SCB5 per-class from main anchor file
with open(ANCHOR_SCB5) as f:
    scb5_full = json.load(f)

domain_info = [
    ("BloodMNIST\n(blood cell)", cd_data["BloodMNIST"]),
    ("PathMNIST\n(pathology)", cd_data["PathMNIST"]),
    ("TissueMNIST\n(kidney)", cd_data["TissueMNIST"]),
    ("EuroSAT\n(satellite)", cd_data["EuroSAT"]),
]
# Compute SCB5 per-class: weighted by n
scb5_per_class = {}
for ds_name in ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]:
    for cname, cdata in scb5_full[ds_name]["per_class_acc"].items():
        scb5_per_class[cname] = cdata["acc"]
domain_info.append(("SCB5\n(classroom)", {"per_class_acc": {k: {"acc": v} for k, v in scb5_per_class.items()},
                                           "total": sum(scb5_full[ds].get("total",0) for ds in ["TeacherBehavior","HandriseReadWrite","BowTurnHead"]),
                                           "correct": sum(scb5_full[ds].get("correct",0) for ds in ["TeacherBehavior","HandriseReadWrite","BowTurnHead"])}))

fig4, ax = plt.subplots(figsize=(6, 4.5))
domains = [d[0] for d in domain_info]
values = []
errs = []
for dname, ddata in domain_info:
    accs = [v["acc"] for v in ddata["per_class_acc"].values()]
    overall = ddata.get("correct", 0) / max(ddata.get("total", 1), 1) * 100
    values.append(overall)
    errs.append(np.std(accs, ddof=1) / np.sqrt(len(accs)))
base = np.array([0x1a/255, 0x99/255, 0x88/255])
colors_bar = [tuple(base * (0.4 + 0.6 * v / max(values))) for v in values]
bars = ax.bar(domains, values, color=colors_bar, width=0.6, edgecolor="white", linewidth=0.5)
for bar, val, err in zip(bars, values, errs):
    ax.errorbar(bar.get_x() + bar.get_width()/2, val, yerr=err,
                fmt="none", color="#d62728", capsize=4, capthick=1, linewidth=1)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold", zorder=5)
# n= inside bar bottom
n_classes = [8, 9, 8, 10, 13]
# Visually mark SCB5 (last bar) as in-domain reference via a divider
ax.axvline(x=3.5, color="#bbbbbb", linestyle="--", linewidth=1, zorder=0)
for bar, nc in zip(bars, n_classes):
    ax.text(bar.get_x() + bar.get_width()/2, 1.0, f"$n={nc}$",
            ha="center", va="bottom", fontsize=7.5, color="white")
ax.set_xticklabels(domains, fontsize=9)
ax.set_ylabel("AnchorScore (%)", fontsize=12)
ax.set_ylim(0, 55)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
fig4.savefig(OUT_DIR / "fig2_cross_domain.png", bbox_inches="tight")
fig4.savefig(OUT_DIR / "fig2_cross_domain.pdf", bbox_inches="tight")
print(f"Saved {OUT_DIR / 'fig2_cross_domain.png'}")

# ── Figure 6: Calibration reliability diagram (binned) ──
with open(RESULTS_DIR / "01_core" / "correlation" / "pooled_class_level_results.json") as f:
    pooled_data = json.load(f)

anchor_all = np.array([e["anchor_score"] for e in pooled_data["data"]])
mllm_all = np.array([e["mllm_mean"] for e in pooled_data["data"]])

n_bins = 5
bin_edges = np.linspace(0, 100, n_bins + 1)
bin_centers = []
bin_anchor_means = []
bin_mllm_means = []
bin_counts = []
for i in range(n_bins):
    lo, hi = bin_edges[i], bin_edges[i + 1]
    mask = (anchor_all >= lo) & (anchor_all <= hi) if i == n_bins - 1 else (anchor_all >= lo) & (anchor_all < hi)
    bin_centers.append((lo + hi) / 2)
    bin_anchor_means.append(anchor_all[mask].mean() if mask.sum() > 0 else 0)
    bin_mllm_means.append(mllm_all[mask].mean() if mask.sum() > 0 else 0)
    bin_counts.append(mask.sum())

fig6, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6, 5.5),
                                       gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

x = np.arange(n_bins)
width = 0.35
ax_top.bar(x - width/2, bin_anchor_means, width, color=C_CLIP, alpha=0.85, label="AnchorScore")
ax_top.bar(x + width/2, bin_mllm_means, width, color=C_MLLM, alpha=0.85, label="MLLM accuracy")
ax_top.set_ylabel("Accuracy (%)", fontsize=12)
ax_top.set_title("AnchorScore Calibration (Reliability Diagram)", fontsize=13, fontweight="bold")
ax_top.set_xticks(x)
ax_top.set_xticklabels([f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}" for i in range(n_bins)], fontsize=9)
ax_top.legend(fontsize=9, loc="upper left")
ax_top.grid(True, alpha=0.2, axis="y")
for i, c in enumerate(bin_counts):
    y_ref = max(bin_anchor_means[i], bin_mllm_means[i])
    ax_top.text(i, y_ref + 1.5, f"$n$={c}", ha="center", fontsize=7, color="#666")
# Caveat: last bin has only 2 EuroSAT classes
if bin_counts[-1] == 2:
    ax_top.text(n_bins - 1, 5, "n=2\n(EuroSAT\nonly)", ha="center", fontsize=6, color="#999",
                fontstyle="italic")

deltas = [m - a for a, m in zip(bin_anchor_means, bin_mllm_means)]
colors_delta = [C_CLIP if d < 0 else C_MLLM for d in deltas]
ax_bot.bar(x, deltas, width=0.5, color=colors_delta, alpha=0.8)
ax_bot.axhline(0, color="#333", linewidth=0.8)
ax_bot.set_ylabel("MLLM $-$ CLIP (pp)", fontsize=10)
ax_bot.set_xlabel("AnchorScore bin", fontsize=11)
ax_bot.set_xticks(x)
ax_bot.set_xticklabels([f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}" for i in range(n_bins)], fontsize=9)
ax_bot.grid(True, alpha=0.2, axis="y")

# ECE annotation on top panel
ece = sum(abs(a - m) * (c / len(anchor_all)) for a, m, c in zip(bin_anchor_means, bin_mllm_means, bin_counts)) / 100.0
ax_top.text(0.98, 0.05, f"ECE = {ece:.3f}", transform=ax_top.transAxes, fontsize=12,
            ha="right", va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#333", linewidth=0.8))

plt.tight_layout()
fig6.savefig(OUT_DIR / "figA2_calibration.png", bbox_inches="tight")
fig6.savefig(OUT_DIR / "figA2_calibration.pdf", bbox_inches="tight")
print(f"Saved {OUT_DIR / 'figA2_calibration.png'}")

# ── Figure 5: Forest plot — robustness summary ──
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch

# ── Load data from result files ──
unified = json.load((RESULTS_DIR / "01_core" / "correlation" / "unified_results.json").open())
meta    = json.load((RESULTS_DIR / "05_applications" / "meta_analysis_results.json").open())
bb      = json.load((RESULTS_DIR / "02_robustness" / "multi_backbone" / "backbone_correlation_results.json").open())
bbox_repr = json.load((RESULTS_DIR / "02_robustness" / "input_repr" / "input_repr_results.json").open())
s40_corr = json.load((RESULTS_DIR / "02_robustness" / "stanford40" / "stanford40_correlation.json").open())
scbllm   = json.load((RESULTS_DIR / "01_core" / "scb5_llm_expansion" / "ensemble_correlation.json").open())

def fisher_ci(rho, n):
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    return (float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)))

scb5_cl = unified["scb5_class_level"]
cd_cl   = unified["cross_domain_class_level"]
s40_rho = s40_corr["multi_model_mean_6_ollama"]["spearman_rho"]
scbllm_rho = scbllm["results"]["scb5_llm_10_5mllm"]["rho"]

forest_rows = [
    ("section", "Primary Studies"),
    ("data", f"SCB5 (n={scb5_cl['n']})",
     scb5_cl["spearman_rho"], *fisher_ci(scb5_cl["spearman_rho"], scb5_cl["n"]), "o", C_CLIP),
    ("data", f"Stanford40 (n={s40_corr['multi_model_mean_6_ollama']['n']})",
     s40_rho, *fisher_ci(s40_rho, s40_corr["multi_model_mean_6_ollama"]["n"]), "o", C_CLIP),
    ("data", "SCB-LLM (n=10)",
     scbllm_rho, *fisher_ci(scbllm_rho, 10), "o", C_CLIP),
    ("data", f"Cross-domain (n={cd_cl['n']})",
     cd_cl["spearman_rho"], *fisher_ci(cd_cl["spearman_rho"], cd_cl["n"]), "o", C_CLIP),
    ("section", "Meta-Analysis"),
    ("data", "Activity FE (3)",
     meta["subgroup_all_scene"]["r"], *meta["subgroup_all_scene"]["ci_95"], "D", C_MLLM),
    ("data", "4-study RE (4)",
     meta["random_effects"]["r"], *meta["random_effects"]["ci_95"], "D", C_MLLM),
    ("section", "Robustness checks"),
    ("data", "LAION L/14",
     bb["laion_l14"]["spearman_rho"], *fisher_ci(bb["laion_l14"]["spearman_rho"], bb["laion_l14"]["n_points"]), "s", "#5a5a5a"),
    ("data", "OpenAI L/14",
     bb["openai_l14"]["spearman_rho"], *fisher_ci(bb["openai_l14"]["spearman_rho"], bb["openai_l14"]["n_points"]), "s", "#5a5a5a"),
    ("data", "OpenAI B/32",
     bb["openai_b32"]["spearman_rho"], *fisher_ci(bb["openai_b32"]["spearman_rho"], bb["openai_b32"]["n_points"]), "s", "#5a5a5a"),
    ("data", "Bbox-aligned",
     bbox_repr["bbox_aligned"]["spearman_rho"], *fisher_ci(bbox_repr["bbox_aligned"]["spearman_rho"], bbox_repr["n_points"]), "s", "#5a5a5a"),
    ("section", "Summary"),
    ("data", f"Cross-dataset (n={unified['pooled_class_level']['n']})",
     unified["pooled_class_level"]["spearman_rho"], *fisher_ci(unified["pooled_class_level"]["spearman_rho"], unified["pooled_class_level"]["n"]), "^", C_MLLM),
]

fig5, ax5 = plt.subplots(figsize=(8, 0.32 * len(forest_rows) + 1.0))

# zero reference line
ax5.axvline(0, color="#d32f2f", linewidth=0.7, linestyle="--", zorder=1, alpha=0.6)

# section background bands
SEC_BANDS = {"Meta-Analysis": "#e8f4f8", "Robustness checks": "#f0f0e8", "Summary": "#f5f0e8"}

y = 0
all_labels = []
all_ticks = []
for row in forest_rows:
    rtype, label, *rest = row
    if rtype == "section":
        bg = SEC_BANDS.get(label)
        if bg:
            ax5.axhspan(y - 0.5, y + 0.5, facecolor=bg, zorder=0)
        all_labels.append(label)
        all_ticks.append(y)
        y -= 1
        continue
    rho, lo, hi, marker, color = rest
    if rtype == "data":
        ms = 8 if marker == "D" else (7 if marker == "^" else 6)
        mew = 0.5 if marker == "D" else 0.3
        ax5.errorbar(rho, y, xerr=[[rho - lo], [hi - rho]],
                     fmt=marker, color=color, capsize=2, capthick=0.8,
                     markersize=ms, markeredgewidth=mew, zorder=4)
        ax5.text(hi + 0.03, y, f"{rho:.3f} [{lo:.3f}, {hi:.3f}]", ha="left", va="center",
                 fontsize=8, color=color,
                 path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    all_labels.append(label)
    all_ticks.append(y)
    y -= 1

ax5.set_yticks(all_ticks)
ax5.set_yticklabels(all_labels, fontsize=9)
for tl, r in zip(ax5.get_yticklabels(), forest_rows):
    if r[0] == "section":
        tl.set_fontweight("bold")
        tl.set_fontsize(9.5)

ax5.set_xlabel("Spearman $\\rho$ [95% CI]", fontsize=11)
ax5.set_xlim(-0.35, 1.35)
ax5.xaxis.set_ticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax5.grid(True, alpha=0.15, axis="x")
ax5.tick_params(labelsize=9)
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)

legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_CLIP, markersize=6,
           label="Primary study (per-class $\\rho$)"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=C_MLLM, markersize=8,
           label="Meta-analytic estimate"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#5a5a5a", markersize=6,
           label="Robustness check"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=C_MLLM, markersize=7,
           label="Cross-dataset class-level estimate"),
]
ax5.legend(handles=legend_elements, fontsize=8, loc="upper left", bbox_to_anchor=(0.02, 1), framealpha=0.85)
fig5.savefig(OUT_DIR / "fig3_forest.png", dpi=300, bbox_inches="tight")

fig5.savefig(OUT_DIR / "fig3_forest.pdf", bbox_inches="tight")

print(f"Saved {OUT_DIR / 'fig3_forest.png'}")
# ── Figure 7: Pareto curve — hybrid routing (accuracy vs cost) ──
hybrid_tb = json.load((RESULTS_DIR / "05_applications" / "hybrid" / "hybrid_fixed_mllm.json").open())
stanford_hybrid = json.load((RESULTS_DIR / "05_applications" / "hybrid_stanford40" / "hybrid_simulation.json").open())

fig7, ax7 = plt.subplots(figsize=(7, 4.5))

# TeacherBehavior: scan taus by collecting oracle gains at various thresholds
# From hybrid_fixed_mllm: we have the fixed-MLLM gain range and the clip_acc
tb_clip = hybrid_tb["TeacherBehavior"]["clip_acc"]
hrw_clip = hybrid_tb["HandriseReadWrite"]["clip_acc"]
bth_clip = hybrid_tb["BowTurnHead"]["clip_acc"]

# Stanford40: full sweep from simulation
stanford_taus = sorted([int(k) for k in stanford_hybrid.keys()])
stanford_accs = [stanford_hybrid[str(t)]["accuracy"] for t in stanford_taus]
stanford_costs = [stanford_hybrid[str(t)]["cost_savings_pct"] for t in stanford_taus]

# For SCB5 subsets: we plot the Pareto-optimal point + CLIP baseline
# TB: tau=45, clip=32.56, oracle=43.64, cost_saved=52.9
# HRW: tau=45, clip=60.92, oracle=64.11, cost_saved=63.4
# BTH: tau=55, clip=60.79, oracle=66.94, cost_saved=77.0

datasets_hybrid = [
    ("TeacherBehavior (8 cls)", tb_clip, 43.64, 52.9, C_CLIP),
    ("HandriseReadWrite (3 cls)", hrw_clip, 64.11, 63.4, "#56B4E9"),
    ("BowTurnHead (2 cls)", bth_clip, 66.94, 77.0, "#CC79A7"),
]

for label, clip_acc, hybrid_acc, cost_saved, color in datasets_hybrid:
    # CLIP baseline (cost=100%)
    ax7.scatter(0, clip_acc, s=80, color=color, marker="o", edgecolors="white",
                linewidth=0.8, zorder=5)
    # Hybrid point
    ax7.scatter(cost_saved, hybrid_acc, s=100, color=color, marker="s",
                edgecolors="white", linewidth=0.8, zorder=6)
    # Dashed horizontal: CLIP baseline level
    ax7.hlines(clip_acc, 0, 100, colors=color, linestyles=":", linewidth=0.8, alpha=0.4)
    if label == "TeacherBehavior (8 cls)":
        ax7.annotate(f"{label}\nCLIP→Hybrid", (cost_saved + 2, hybrid_acc - 2),
                     fontsize=7, color=color, fontweight="bold")
    else:
        ax7.annotate(label, (cost_saved + 2, hybrid_acc + 0.5),
                     fontsize=7, color=color)

# Stanford40 Pareto curve
ax7.plot(stanford_costs, stanford_accs, "o-", color=C_MLLM, markersize=5,
         linewidth=1.5, alpha=0.7, zorder=3, label="Stanford40 (40 cls)")
# Mark key thresholds
for tau_show in [65, 80, 90]:
    idx = stanford_taus.index(tau_show)
    s_acc = stanford_accs[idx]
    s_cost = stanford_costs[idx]
    ax7.annotate(f"$\\tau$={tau_show}", (s_cost + 1, s_acc),
                 fontsize=7, color=C_MLLM, fontweight="bold")

# Stanford40 CLIP baseline
ax7.scatter(0, 91.86, s=80, color=C_MLLM, marker="o", edgecolors="white",
            linewidth=0.8, zorder=5)
ax7.hlines(91.86, 0, 100, colors=C_MLLM, linestyles=":", linewidth=0.8, alpha=0.4)

# Legend
legend_elements7 = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#555", markersize=7, label="All-CLIP baseline"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#555", markersize=8, label="Hybrid (Pareto-optimal)"),
]
ax7.legend(handles=legend_elements7, fontsize=9, loc="lower right")

ax7.set_xlabel("MLLM Cost Saved (%)", fontsize=12)
ax7.set_ylabel("Overall Accuracy (%)", fontsize=12)
ax7.set_title("Hybrid Routing: Accuracy--Cost Trade-off", fontsize=13, fontweight="bold")
ax7.set_xlim(-5, 105)
ax7.grid(True, alpha=0.2)
ax7.spines["top"].set_visible(False)
ax7.spines["right"].set_visible(False)

plt.tight_layout()
fig7.savefig(OUT_DIR / "fig4_pareto.png", dpi=300, bbox_inches="tight")
fig7.savefig(OUT_DIR / "fig4_pareto.pdf", bbox_inches="tight")
print(f"Saved {OUT_DIR / 'fig4_pareto.png'}")

print("Done.")
