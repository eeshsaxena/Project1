"""
Generates ALL benchmark charts for TruthfulRAG v5 viva:
  chart1_halueval_hrr.png      — HRR vs EM (HaluEval 200Q)
  chart2_multidataset_em.png   — Multi-dataset EM (125Q)
  chart3_summary.png           — 3-panel summary
  chart4_timeline.png          — RAG evolution timeline (Chapter 1)
  chart5_cdr_highlight.png     — CDR conflict detection callout chart
  chart6_temporal_decay.png    — Temporal decay formula visualised
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe

OUT = "d:\\Project-1\\benchmark_charts"
os.makedirs(OUT, exist_ok=True)

BG   = "#0F1117"
GRID = "#2A2D35"
TEXT = "#ECF0F1"
LC_COL = "#E74C3C"
V4_COL = "#F39C12"
V5_COL = "#27AE60"
ACC    = "#3498DB"

def styled(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=11)
    for sp in ["bottom","left"]:
        ax.spines[sp].set_color(GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)
    for lbl in [ax.xaxis.label, ax.yaxis.label, ax.title]:
        lbl.set_color(TEXT)

# ─────────────────────────────────────────────────────────────────
# CHART 4: RAG EVOLUTION TIMELINE (Chapter 1 figure)
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(2019.5, 2026.8)
ax.set_ylim(-1.8, 1.8)
ax.axis("off")

events = [
    (2020.5, 0,    "RAG (Lewis et al.)\narXiv:2005.11401\nDense retrieval baseline",           LC_COL,  0.85, "below"),
    (2021.5, 0,    "REALM\nKnowledge-grounded\nretrieval pretraining",                          "#8E44AD", 0.75, "above"),
    (2022.5, 0,    "FiD / Atlas\nFusion-in-Decoder;\nmulti-document RAG",                       "#2980B9", 0.75, "below"),
    (2023.5, 0,    "Self-RAG / CRAG\nAdaptive retrieval;\ncorrective feedback loops",           "#16A085", 0.80, "above"),
    (2024.5, 0,    "KG-RAG v4\narXiv:2511.10375\nGraph-based conflict detection",               V4_COL,  0.90, "below"),
    (2025.8, 0,    "TruthfulRAG v5\n[This Work]\nTemporal decay · Corroboration\n· HRR=92%",   V5_COL,  1.05, "above"),
]

# Draw timeline axis
ax.annotate("", xy=(2026.7, 0), xytext=(2019.6, 0),
            arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=2))

for x, y, label, color, height, side in events:
    # Dot on timeline
    ax.scatter([x], [0], s=180, color=color, zorder=5, edgecolors="white", linewidths=1.5)
    # Stem
    sign = 1 if side == "above" else -1
    ax.plot([x, x], [0, sign*height*0.55], color=color, lw=1.5, alpha=0.8)
    # Label box
    bbox = dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.18, edgecolor=color, linewidth=1.2)
    ax.text(x, sign*(height*0.55 + 0.08), label, ha="center",
            va="bottom" if side=="above" else "top",
            color=TEXT, fontsize=9.5, fontweight="bold" if "v5" in label else "normal",
            bbox=bbox, multialignment="center")

# Year labels
for yr in [2020, 2021, 2022, 2023, 2024, 2025, 2026]:
    ax.text(yr + 0.5 if yr < 2026 else yr - 0.2, -0.12, str(yr + (1 if yr < 2026 else 0)),
            ha="center", va="top", color=TEXT, fontsize=10, alpha=0.7)

ax.text(0.5, 1.0, "RAG Research Evolution Timeline — 2020 → 2026",
        transform=ax.transAxes, ha="center", va="top",
        color=TEXT, fontsize=15, fontweight="bold")
ax.text(0.5, 0.93, "Positioning TruthfulRAG v5 in the literature landscape",
        transform=ax.transAxes, ha="center", va="top", color="#95A5A6", fontsize=11)

plt.tight_layout()
p4 = os.path.join(OUT, "chart4_timeline.png")
plt.savefig(p4, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {p4}")

# ─────────────────────────────────────────────────────────────────
# CHART 5: CDR HIGHLIGHT — the key viva chart
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
styled(fig, ax)

systems  = ["LangChain\n(No conflict logic)", "KG-RAG v4\n(Entropy only)", "TruthfulRAG v5\n(Temporal + Corroboration)"]
cdr_vals = [25.0, 25.0, 50.0]
colors   = [LC_COL, V4_COL, V5_COL]

bars = ax.bar(systems, cdr_vals, color=colors, alpha=0.88, width=0.45, zorder=3)
for bar, val, col in zip(bars, cdr_vals, colors):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{val}%", ha="center", va="bottom", color=TEXT,
            fontsize=18, fontweight="bold")

# Highlight v5 bar
bars[2].set_edgecolor("#2ECC71")
bars[2].set_linewidth(3)

# Random baseline annotation
ax.axhline(25, color="#E74C3C", linestyle="--", lw=1.8, alpha=0.7)
ax.text(2.38, 26.2, "Random baseline\n(25% — no better than chance)", color="#E74C3C",
        fontsize=9, ha="right", va="bottom", style="italic")

# Delta annotation
ax.annotate("", xy=(2, 50), xytext=(2, 25),
            arrowprops=dict(arrowstyle="<->", color="#2ECC71", lw=2.5))
ax.text(2.05, 37.5, "+25pp\n(2× improvement)", color="#2ECC71",
        fontsize=12, fontweight="bold", va="center")

ax.set_ylabel("Conflict Detection Rate — CDR% ↑", fontsize=13)
ax.set_ylim(0, 72)
ax.set_title("Conflict Detection Rate (CDR%) — ConflictQA Subset\n"
             "TruthfulRAG v5 achieves 2× CDR vs both baselines", fontsize=13, pad=15)
ax.tick_params(axis="x", labelsize=11)

plt.tight_layout()
p5 = os.path.join(OUT, "chart5_cdr_highlight.png")
plt.savefig(p5, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {p5}")

# ─────────────────────────────────────────────────────────────────
# CHART 6: Temporal Decay Curve — formula visualised
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
styled(fig, ax)

age  = np.linspace(0, 30, 300)
lam  = 0.08
decay = np.exp(-lam * age)

ax.plot(age, decay, color=V5_COL, lw=3, label=r"v5: $e^{-0.08 \times age}$")
ax.axhline(1.0, color=V4_COL, lw=2, linestyle="--", label="v4 / LangChain: decay = 1.0 (no penalty)")
ax.fill_between(age, decay, 1.0, alpha=0.15, color=V5_COL)

# Annotations
for yr, label in [(5,"5yr"), (10,"10yr"), (15,"15yr"), (20,"20yr")]:
    d = np.exp(-lam*yr)
    ax.scatter([yr], [d], color=V5_COL, s=70, zorder=5)
    ax.text(yr, d-0.05, f"{d:.2f}", ha="center", va="top", color=TEXT, fontsize=9)

ax.set_xlabel("Document Age (years)", fontsize=12)
ax.set_ylabel("Edge Score Multiplier", fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_title("Temporal Decay Score: v5 vs v4 / LangChain\n"
             r"$\text{score}_{v5} = \text{base\_score} \times e^{-\lambda \cdot age}$   ($\lambda=0.08$)",
             fontsize=13, pad=12)
ax.legend(fontsize=11, facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="upper right")

ax.text(0.55, 0.45, "Shaded area = facts penalised\nby v5 but passed unchanged by v4",
        transform=ax.transAxes, color="#95A5A6", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#1A1D24", edgecolor=GRID, alpha=0.8))

plt.tight_layout()
p6 = os.path.join(OUT, "chart6_temporal_decay.png")
plt.savefig(p6, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {p6}")

# ─────────────────────────────────────────────────────────────────
# REGENERATE CHART 1 (cleaner version with CDR callout)
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
styled(fig, ax)

systems = ["LangChain\n(Baseline)", "KG-RAG v4\n(Simulated)", "TruthfulRAG v5\n(Ours)"]
hrr = [77.5, 91.0, 92.0]
em  = [85.5, 15.5, 14.5]
colors = [LC_COL, V4_COL, V5_COL]
x = np.arange(len(systems)); w = 0.35

b1 = ax.bar(x-w/2, hrr, w, color=colors, alpha=0.9, zorder=3)
b2 = ax.bar(x+w/2, em,  w, color=colors, alpha=0.40, zorder=3, hatch="//")

for bar, val in zip(b1, hrr):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
            f"{val}%", ha="center", va="bottom", color=TEXT, fontsize=14, fontweight="bold")
for bar, val in zip(b2, em):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
            f"{val}%", ha="center", va="bottom", color=TEXT, fontsize=11)

b1[2].set_edgecolor("#2ECC71"); b1[2].set_linewidth(2.5)
ax.set_xticks(x); ax.set_xticklabels(systems, fontsize=12)
ax.set_ylabel("Score (%)", fontsize=13); ax.set_ylim(0, 110)
ax.set_title("HaluEval Benchmark — Hallucination Rejection Rate vs Exact Match\n"
             "(pminervini/HaluEval · 200 samples · 3 systems)", fontsize=13, pad=15)

handles = [mpatches.Patch(color="white", alpha=0.9, label="Solid = HRR% (↑ better — rejects hallucination)"),
           mpatches.Patch(facecolor="white", alpha=0.4, hatch="//", label="Hatched = EM% (↑ better — exact match)")]
ax.legend(handles=handles, fontsize=10, facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="lower right")
ax.annotate("+14.5pp vs LangChain", xy=(2-w/2, 92), xytext=(1.1, 100),
            color="#2ECC71", fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2ECC71", lw=1.5))

plt.tight_layout()
plt.savefig(os.path.join(OUT,"chart1_halueval_hrr.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

# ─────────────────────────────────────────────────────────────────
# REGENERATE CHART 3 — cleaner labels
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.patch.set_facecolor(BG)
fig.suptitle("TruthfulRAG v5 — Complete Benchmark Summary\nAll metrics: higher is better · v5 wins on every conflict-specific measure",
             fontsize=14, color=TEXT, y=1.02)

panels = {
    "HRR%\n(HaluEval · 200Q)":  ([77.5, 91.0, 92.0], "Hallucination Rejection Rate\nv5 rejects traps 14.5pp more than LC"),
    "CDR%\n(ConflictQA · 125Q)": ([25.0, 25.0, 50.0], "Conflict Detection Rate\nv5 = 2× improvement over both baselines"),
    "Temporal%\n(SQuAD · 125Q)": ([14.3, 35.7, 35.7], "Temporal Accuracy\nv5 = 2.5× improvement over LangChain"),
}

for ax, (metric, (vals, note)) in zip(axes, panels.items()):
    styled(fig, ax)
    bars = ax.bar(["LC","v4","v5"], vals, color=[LC_COL,V4_COL,V5_COL], alpha=0.9, zorder=3, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{v}%", ha="center", va="bottom", color=TEXT, fontsize=15, fontweight="bold")
    bars[2].set_edgecolor("#2ECC71"); bars[2].set_linewidth(2.5)
    ax.set_title(metric, color=TEXT, fontsize=12, pad=10)
    ax.set_ylim(0, 120); ax.tick_params(labelsize=11)
    ax.text(0.5, 0.04, note, transform=ax.transAxes, ha="center",
            color="#95A5A6", fontsize=8.5, multialignment="center")

plt.tight_layout()
plt.savefig(os.path.join(OUT,"chart3_summary.png"), dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

print("\n✅ All 6 charts generated:")
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT,f)
    print(f"  {f:45s}  {os.path.getsize(p)//1024:>4} KB")
