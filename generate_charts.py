"""
generate_charts.py — generates all automatable report figures
Outputs PNGs to:  d:/Project-1/Report_Format___BTech_CSE_IIIT_Manipur__1___1_/report_file/
"""

import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

OUT = r"d:\Project-1\Report_Format___BTech_CSE_IIIT_Manipur__1___1_\report_file"
os.makedirs(OUT, exist_ok=True)

BW   = "#1a1a1a"
GRAY = "#555555"
LG   = "#aaaaaa"
WH   = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": WH,
    "axes.facecolor": WH,
    "axes.edgecolor": BW,
    "axes.labelcolor": BW,
    "xtick.color": BW,
    "ytick.color": BW,
    "text.color": BW,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. GROUPED BAR CHART — accuracy across systems (fig:bar)
# ─────────────────────────────────────────────────────────────────────────────
categories = ["Stable Facts", "Temporal Conflict", "Claim Verification"]
lc  = [87, 48, 0]
v4  = [87, 72, 0]
v5  = [91, 94, 88]

x = np.arange(len(categories))
w = 0.26
fig, ax = plt.subplots(figsize=(8, 4.5))
bars_lc = ax.bar(x - w,   lc,  w, label="LangChain RAG",   color=LG,   edgecolor=BW, linewidth=0.7)
bars_v4 = ax.bar(x,       v4,  w, label="TruthfulRAG v4",  color=GRAY, edgecolor=BW, linewidth=0.7)
bars_v5 = ax.bar(x + w,   v5,  w, label="TruthfulRAG v5",  color=BW,   edgecolor=BW, linewidth=0.7)

for bar in list(bars_lc) + list(bars_v4) + list(bars_v5):
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h}%",
                ha="center", va="bottom", fontsize=7.5, color=BW)

ax.set_ylabel("Accuracy (%)", fontsize=10)
ax.set_title("Answer Accuracy Across System Generations and Question Types", fontsize=11, pad=10)
ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=9)
ax.set_ylim(0, 108)
ax.legend(fontsize=8.5, framealpha=0.3)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, color=LG)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{OUT}/chart1_grouped_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ chart1_grouped_bar.png")

# ─────────────────────────────────────────────────────────────────────────────
# 2. RADAR CHART — six-dimension capability (fig:radar)
# ─────────────────────────────────────────────────────────────────────────────
labels = ["Conflict\nDetection", "Temporal\nAwareness", "Confidence\nScore",
          "Claim\nVerify", "Explanation\nChain", "Domain\nAgnosticism"]
N = len(labels)
lc_vals  = [0, 0, 0, 0, 0, 60]
v5_vals  = [95, 90, 88, 88, 92, 95]
angles   = [n / float(N) * 2 * math.pi for n in range(N)]
angles  += angles[:1]
lc_vals  += lc_vals[:1]
v5_vals  += v5_vals[:1]

fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
ax.set_facecolor(WH)
ax.set_theta_offset(math.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, size=8.5)
ax.set_ylim(0, 100)
ax.set_yticks([25, 50, 75, 100])
ax.set_yticklabels(["25", "50", "75", "100"], size=7, color=GRAY)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, color=LG)
ax.xaxis.grid(True, alpha=0.3, color=LG)

ax.plot(angles, lc_vals,  color=LG,  linewidth=1.5, linestyle="--", label="LangChain RAG")
ax.fill(angles, lc_vals,  color=LG,  alpha=0.18)
ax.plot(angles, v5_vals,  color=BW,  linewidth=2,   linestyle="-",  label="TruthfulRAG v5")
ax.fill(angles, v5_vals,  color=BW,  alpha=0.12)

ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.1), fontsize=8.5)
ax.set_title("Capability Comparison: LangChain RAG vs. TruthfulRAG v5", pad=18, fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/chart2_radar.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ chart2_radar.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SCORE GAP — v4 tied vs v5 gap (fig:gap)
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8), sharey=False)
facts = ["Aspirin\n(1950)", "Ibuprofen\n(1986)"]

v4_scores = [0.82, 0.80]
v5_scores  = [0.31, 0.88]

for ax, scores, title in [(ax1, v4_scores, "TruthfulRAG v4 (scores tied)"),
                           (ax2, v5_scores, "TruthfulRAG v5 (gap created)")]:
    colors = [GRAY, BW] if ax == ax1 else [LG, BW]
    bars = ax.bar(facts, scores, color=colors, edgecolor=BW, linewidth=0.7, width=0.45)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Retrieval Score", fontsize=9)
    ax.set_title(title, fontsize=9.5, pad=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color=LG)
    ax.set_axisbelow(True)

ax1.annotate("No clear winner", xy=(0.5, 0.83), ha="center", fontsize=8,
             color=GRAY, fontweight="bold")
ax2.annotate("Older fact\ndeprioritised", xy=(0, 0.25), ha="center", fontsize=8,
             color=GRAY, fontweight="bold")

fig.suptitle("Score Gap Between Conflicting Facts: v4 vs. v5", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/chart3_score_gap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ chart3_score_gap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TEMPORAL DECAY CURVE (fig:decay)
# ─────────────────────────────────────────────────────────────────────────────
lam  = 0.08
ages = np.linspace(0, 30, 300)
w    = np.exp(-lam * ages)
half = math.log(2) / lam

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(ages, w, color=BW, linewidth=2.2)
ax.fill_between(ages, w, alpha=0.10, color=BW)
ax.axvline(half, color=GRAY, linestyle="--", linewidth=1.2)
ax.axhline(0.5,  color=GRAY, linestyle="--", linewidth=1.2)
ax.annotate(f"Half-life ≈ {half:.1f} yrs", xy=(half, 0.5),
            xytext=(half+2, 0.6), fontsize=8.5, color=GRAY,
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))
ax.scatter([0, 5, 10, 20], [np.exp(-lam*t) for t in [0, 5, 10, 20]],
           color=BW, zorder=5, s=45)
for t in [0, 5, 10, 20]:
    ax.annotate(f"{np.exp(-lam*t):.2f}", (t, np.exp(-lam*t)),
                textcoords="offset points", xytext=(4, 6), fontsize=7.5)
ax.set_xlabel("Fact Age (years)", fontsize=10)
ax.set_ylabel("Retrieval Weight", fontsize=10)
ax.set_title(r"Temporal Decay: $w(t) = e^{-\lambda t},\;\lambda = 0.08$", fontsize=11, pad=10)
ax.set_ylim(0, 1.05)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, color=LG)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{OUT}/chart4_temporal_decay.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ chart4_temporal_decay.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. CORROBORATION BONUS CURVE (fig:corr)
# ─────────────────────────────────────────────────────────────────────────────
supports = np.arange(1, 11)
bonus    = np.log1p(supports)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(supports, bonus, "o-", color=BW, linewidth=2.2, markersize=6)
for x_val, y_val in zip(supports, bonus):
    ax.annotate(f"{y_val:.2f}", (x_val, y_val),
                textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax.set_xlabel("Number of Agreeing Source Documents", fontsize=10)
ax.set_ylabel(r"$\log(1 + \mathrm{support})$ Score Multiplier", fontsize=10)
ax.set_title("[N1] Cross-Document Corroboration Bonus vs. Source Count", fontsize=11, pad=10)
ax.set_xticks(supports)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, color=LG)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{OUT}/chart6_corroboration.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ chart6_corroboration.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. CONFIDENCE SCORE BREAKDOWN (fig:confidence)
# ─────────────────────────────────────────────────────────────────────────────
queries = ["Fever in\nchildren", "PM of\nIndia", "ISRO\nfather", "Space\ntelescope"]
entropy_comp  = [0.14, 0.18, 0.20, 0.22]
support_comp  = [0.12, 0.15, 0.22, 0.20]
recency_comp  = [0.16, 0.14, 0.19, 0.20]
totals        = [sum(e) for e in zip(entropy_comp, support_comp, recency_comp)]

x  = np.arange(len(queries))
w2 = 0.45
fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar(x, [e*100 for e in entropy_comp], w2, label="Entropy Change",  color=BW,   edgecolor=BW, linewidth=0.5)
b2 = ax.bar(x, [e*100 for e in support_comp], w2, bottom=[e*100 for e in entropy_comp],
            label="Corroboration",  color=GRAY, edgecolor=BW, linewidth=0.5)
b3 = ax.bar(x, [e*100 for e in recency_comp], w2,
            bottom=[(e+s)*100 for e,s in zip(entropy_comp, support_comp)],
            label="Temporal Recency", color=LG,   edgecolor=BW, linewidth=0.5)
for i, tot in enumerate(totals):
    ax.text(i, tot*100 + 1.2, f"{tot*100:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Confidence Score (%)", fontsize=10)
ax.set_title("[N5] Calibrated Confidence Score Breakdown per Query", fontsize=11, pad=10)
ax.set_xticks(x); ax.set_xticklabels(queries, fontsize=9)
ax.set_ylim(0, 70)
ax.legend(fontsize=8.5, framealpha=0.3)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, color=LG)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{OUT}/chart7_confidence.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ chart7_confidence.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. ACCURACY GAINS OVER LANGCHAIN (fig:gains)
# ─────────────────────────────────────────────────────────────────────────────
domains = ["Medical", "Legal", "Indian Science", "Space Science"]
lc_acc  = [55,  60,  58,  52]
v5_acc  = [93,  85,  92,  91]
gains   = [v - l for v, l in zip(v5_acc, lc_acc)]

x  = np.arange(len(domains))
w3 = 0.38
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.bar(x - w3/2, lc_acc, w3, label="LangChain RAG",  color=LG,  edgecolor=BW, linewidth=0.7)
ax.bar(x + w3/2, v5_acc, w3, label="TruthfulRAG v5", color=BW,  edgecolor=BW, linewidth=0.7)
for i, (lc_v, v5_v, g) in enumerate(zip(lc_acc, v5_acc, gains)):
    ax.text(i - w3/2, lc_v + 0.8, f"{lc_v}%", ha="center", va="bottom", fontsize=8)
    ax.text(i + w3/2, v5_v + 0.8, f"{v5_v}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.annotate(f"+{g}pp", xy=(i, (lc_v + v5_v)/2),
                ha="center", fontsize=8.5, color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=GRAY, lw=0.6))
ax.set_ylabel("Accuracy on Conflicted Queries (%)", fontsize=10)
ax.set_title("Accuracy Gain of TruthfulRAG v5 over LangChain RAG Baseline", fontsize=11, pad=10)
ax.set_xticks(x); ax.set_xticklabels(domains, fontsize=9)
ax.set_ylim(0, 108)
ax.legend(fontsize=9, framealpha=0.3)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, color=LG)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{OUT}/chart8_gains_over_langchain.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ chart8_gains_over_langchain.png")

print("\nAll 7 charts generated.")
