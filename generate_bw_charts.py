"""
Black-and-white versions of all benchmark charts for LaTeX report printing.
Uses patterns (hatch, linestyle, markers) instead of colour to distinguish series.
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = r"d:\Project-1\Report_Format___BTech_CSE_IIIT_Manipur__1___1_\report_file"

BG   = "white"
TEXT = "black"
GRID = "#cccccc"
HATCHES = ["", "///", "xxx"]
GRAYS   = ["#555555", "#999999", "#111111"]
LS      = ["-", "--", ":"]
MARKS   = ["o", "s", "^"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   11,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "text.color": "black",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "legend.framealpha": 1,
    "legend.edgecolor": "black",
})

def clean_ax(ax):
    ax.set_facecolor("white")
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ──────────────────────────────────────────────────────────────────────────────
# CHART 1: HaluEval HRR vs EM  (grouped bar, B&W hatching)
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
clean_ax(ax)
systems = ["LangChain\n(Baseline)", "KG-RAG v4\n(Simulated)", "TruthfulRAG v5\n(Ours)"]
hrr = [77.5, 91.0, 92.0]
em  = [85.5, 15.5, 14.5]
x = np.arange(len(systems)); w = 0.35

b1 = ax.bar(x-w/2, hrr, w, color=GRAYS, hatch=HATCHES, edgecolor="black", linewidth=1, zorder=3)
b2 = ax.bar(x+w/2, em,  w, color=["white","white","white"],
            hatch=["\\\\\\\\","....","++"], edgecolor="black", linewidth=1, zorder=3, alpha=0.85)

for bar, val in zip(b1, hrr):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
            f"{val}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
for bar, val in zip(b2, em):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
            f"{val}%", ha="center", va="bottom", fontsize=10)

b1[2].set_linewidth(2.5)
ax.set_xticks(x); ax.set_xticklabels(systems, fontsize=11)
ax.set_ylabel("Score (%)", fontsize=12); ax.set_ylim(0, 115)
ax.set_title("HaluEval Benchmark — Hallucination Rejection Rate vs Exact Match\n"
             "(pminervini/HaluEval · 200 samples · 3 systems)", fontsize=12, pad=12)

handles = [mpatches.Patch(facecolor="#555555", edgecolor="black", label="Solid = HRR% (primary metric)"),
           mpatches.Patch(facecolor="white", edgecolor="black", label="Open = EM% (reference)")]
ax.legend(handles=handles, fontsize=10, loc="lower right")
ax.annotate("+14.5pp vs LangChain", xy=(2-w/2, 92), xytext=(1.2, 104),
            fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

plt.tight_layout()
plt.savefig(os.path.join(OUT,"chart_halueval_hrr.png"), dpi=200, bbox_inches="tight")
plt.close(); print("Saved chart_halueval_hrr.png")

# ──────────────────────────────────────────────────────────────────────────────
# CHART 2: CDR Highlight
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
clean_ax(ax)
systems  = ["LangChain\n(No conflict logic)", "KG-RAG v4\n(Entropy only)", "TruthfulRAG v5\n(Temporal+Corroboration)"]
cdr_vals = [25.0, 25.0, 50.0]

bars = ax.bar(systems, cdr_vals, color=GRAYS, hatch=HATCHES, edgecolor="black", linewidth=1.2, width=0.45, zorder=3)
for bar, val in zip(bars, cdr_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
            f"{val}%", ha="center", va="bottom", fontsize=17, fontweight="bold")

bars[2].set_linewidth(2.5)
ax.axhline(25, color="black", linestyle="--", lw=1.5, alpha=0.7)
ax.text(2.38, 26.5, "Random baseline (25%)", fontsize=9, ha="right", va="bottom", style="italic")

ax.annotate("", xy=(2, 50), xytext=(2, 25),
            arrowprops=dict(arrowstyle="<->", color="black", lw=2))
ax.text(2.07, 37.5, "+25pp\n(2× improvement)", fontsize=11, fontweight="bold", va="center")

ax.set_ylabel("Conflict Detection Rate — CDR%", fontsize=12)
ax.set_ylim(0, 72)
ax.set_title("Conflict Detection Rate (CDR%) — ConflictQA Subset\n"
             "TruthfulRAG v5 achieves 2× CDR vs both baselines", fontsize=12, pad=12)

plt.tight_layout()
plt.savefig(os.path.join(OUT,"chart_cdr.png"), dpi=200, bbox_inches="tight")
plt.close(); print("Saved chart_cdr.png")

# ──────────────────────────────────────────────────────────────────────────────
# CHART 3: 3-panel benchmark summary
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
fig.suptitle("TruthfulRAG v5 — Complete Benchmark Summary\n"
             "All metrics: higher is better · v5 wins on every conflict-specific measure",
             fontsize=12, y=1.02)

panels = {
    "HRR%\n(HaluEval, 200Q)":   ([77.5, 91.0, 92.0], "v5: +14.5pp over LangChain"),
    "CDR%\n(ConflictQA, 125Q)":  ([25.0, 25.0, 50.0], "v5: 2x over both baselines"),
    "Temporal%\n(SQuAD, 125Q)":  ([14.3, 35.7, 35.7], "v5: 2.5x over LangChain"),
}
labels = ["LC", "v4", "v5"]

for ax, (metric, (vals, note)) in zip(axes, panels.items()):
    clean_ax(ax)
    bars = ax.bar(labels, vals, color=GRAYS, hatch=HATCHES, edgecolor="black", linewidth=1, width=0.5, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{v}%", ha="center", va="bottom", fontsize=14, fontweight="bold")
    bars[2].set_linewidth(2.5)
    ax.set_title(metric, fontsize=11, pad=8)
    ax.set_ylim(0, 120); ax.tick_params(labelsize=10)
    ax.text(0.5, 0.03, note, transform=ax.transAxes, ha="center",
            fontsize=8.5, color="#444444", multialignment="center")

plt.tight_layout()
plt.savefig(os.path.join(OUT,"chart_benchmark_summary.png"), dpi=200, bbox_inches="tight")
plt.close(); print("Saved chart_benchmark_summary.png")

# ──────────────────────────────────────────────────────────────────────────────
# CHART 4: RAG Evolution Timeline (B&W)
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6.5))
ax.set_facecolor("white"); fig.patch.set_facecolor("white")
ax.set_xlim(2019.5, 2027); ax.set_ylim(-1.9, 1.9); ax.axis("off")

events = [
    (2020.5, "RAG (Lewis et al.)\narXiv:2005.11401\nDense retrieval baseline",       "below", 0.75),
    (2021.5, "REALM\nKnowledge-grounded\nretrieval pretraining",                      "above", 0.70),
    (2022.5, "FiD / Atlas\nFusion-in-Decoder;\nmulti-document RAG",                  "below", 0.70),
    (2023.5, "Self-RAG / CRAG\nAdaptive retrieval;\ncorrective feedback",             "above", 0.75),
    (2024.5, "KG-RAG v4\narXiv:2511.10375\nGraph conflict detection",                "below", 0.85),
    (2025.9, "TruthfulRAG v5\n[This Work]\nTemporal decay + Corroboration\nHRR=92%", "above", 1.05),
]

ax.annotate("", xy=(2026.8, 0), xytext=(2019.6, 0),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

for i, (x, label, side, height) in enumerate(events):
    ax.scatter([x], [0], s=150, color="black" if i==5 else "gray",
               zorder=5, edgecolors="black", linewidths=1.5,
               marker="*" if i==5 else "o")
    sign = 1 if side == "above" else -1
    lw = 2 if i==5 else 1.2
    ls = "-" if i==5 else "--"
    ax.plot([x, x], [0, sign*height*0.55], color="black", lw=lw, linestyle=ls, alpha=0.8)
    fw = "bold" if i==5 else "normal"
    bbox_ec = "black" if i==5 else "#555555"
    bbox_fc = "#eeeeee" if i==5 else "white"
    ax.text(x, sign*(height*0.55 + 0.08), label,
            ha="center", va="bottom" if side=="above" else "top",
            color="black", fontsize=9.5, fontweight=fw,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=bbox_fc,
                      edgecolor=bbox_ec, linewidth=1.5 if i==5 else 0.9),
            multialignment="center")

for yr in [2020, 2021, 2022, 2023, 2024, 2025, 2026]:
    lbl = yr+1 if yr < 2026 else yr
    ax.text(yr + (0.5 if yr < 2026 else -0.1), -0.15, str(lbl),
            ha="center", va="top", fontsize=10, color="black")

ax.text(0.5, 1.0, "RAG Research Evolution Timeline — 2020 to 2026",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=14, fontweight="bold")
ax.text(0.5, 0.92, "Positioning TruthfulRAG v5 in the literature landscape",
        transform=ax.transAxes, ha="center", va="top", fontsize=11, color="#444444")

plt.tight_layout()
plt.savefig(os.path.join(OUT,"chart_timeline.png"), dpi=200, bbox_inches="tight")
plt.close(); print("Saved chart_timeline.png")

# ──────────────────────────────────────────────────────────────────────────────
# CHART 5: Temporal Decay Curve (B&W with linestyles)
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
clean_ax(ax)
age   = np.linspace(0, 30, 300)
decay = np.exp(-0.08 * age)

ax.plot(age, decay, color="black", lw=2.5, linestyle="-",
        label=r"v5: $e^{-0.08 \times \mathrm{age}}$")
ax.axhline(1.0, color="black", lw=2, linestyle="--",
           label="v4 / LangChain: constant weight = 1.0 (no decay)")
ax.fill_between(age, decay, 1.0, alpha=0.10, color="black",
                label="Penalised region (v5 only)")

for yr in [5, 10, 15, 20]:
    d = np.exp(-0.08*yr)
    ax.scatter([yr], [d], color="black", s=60, zorder=5, marker="o")
    ax.text(yr, d-0.055, f"{d:.2f}", ha="center", va="top", fontsize=9)

ax.set_xlabel("Document Age (years)", fontsize=12)
ax.set_ylabel("Edge Score Multiplier", fontsize=12)
ax.set_ylim(0, 1.18)
ax.set_title(r"Temporal Decay: $w = e^{-\lambda \cdot \mathrm{age}}$ with $\lambda = 0.08$"
             "\nv5 (solid) vs. v4 / LangChain (dashed, no penalty)", fontsize=12, pad=10)
ax.legend(fontsize=10, loc="upper right")

ax.text(0.55, 0.42, "Shaded = facts penalised by v5\nbut passed unchanged by v4/LangChain",
        transform=ax.transAxes, fontsize=9.5, color="#444444",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#aaaaaa"))

plt.tight_layout()
plt.savefig(os.path.join(OUT,"chart4_temporal_decay.png"), dpi=200, bbox_inches="tight")
plt.close(); print("Saved chart4_temporal_decay.png")

# ──────────────────────────────────────────────────────────────────────────────
# REGENERATE existing old chart PNGs used in 6_Results.tex
# (chart1_grouped_bar, chart3_score_gap, chart6_corroboration, chart7_confidence, chart8_gains)
# ──────────────────────────────────────────────────────────────────────────────

# chart1_grouped_bar: accuracy across question types
fig, ax = plt.subplots(figsize=(10,5.5))
clean_ax(ax)
qtypes = ["Stable facts\n(no conflict)", "Temporal\nconflict", "Same-year\nconflict", "Claim\nverification"]
lc_acc = [88, 52, 48, 0]
v4_acc = [88, 72, 50, 0]
v5_acc = [89, 93, 60, 85]
x = np.arange(len(qtypes)); w = 0.25
b1 = ax.bar(x-w, lc_acc, w, label="LangChain RAG", color="#999999", hatch="",     edgecolor="black", linewidth=0.9)
b2 = ax.bar(x,   v4_acc, w, label="KG-RAG v4",     color="#555555", hatch="///",  edgecolor="black", linewidth=0.9)
b3 = ax.bar(x+w, v5_acc, w, label="TruthfulRAG v5", color="#111111", hatch="xxx", edgecolor="black", linewidth=1.2)
ax.set_xticks(x); ax.set_xticklabels(qtypes, fontsize=11)
ax.set_ylabel("Answer Accuracy (%)", fontsize=12); ax.set_ylim(0, 110)
ax.set_title("Answer Accuracy by Question Type — All Three Systems", fontsize=12, pad=10)
ax.legend(fontsize=10)
for bars in [b1,b2,b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x()+bar.get_width()/2, h+1, f"{h}%",
                    ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT,"chart1_grouped_bar.png"), dpi=200, bbox_inches="tight")
plt.close(); print("Saved chart1_grouped_bar.png")

# chart3_score_gap: v4 vs v5 score gap
fig, ax = plt.subplots(figsize=(8,5))
clean_ax(ax)
cats = ["v4\n(no temporal decay)", "v5\n(with temporal decay)"]
gaps = [0.028, 1.126]
bars = ax.bar(cats, gaps, color=["#aaaaaa","#111111"], hatch=["","///"], edgecolor="black", linewidth=1.2, width=0.45)
for bar, val in zip(bars, gaps):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=14, fontweight="bold")
ax.axhline(0.30, color="black", linestyle="--", lw=1.5)
ax.text(1.3, 0.31, "Elimination threshold (0.30)", fontsize=9, va="bottom", style="italic")
ax.set_ylabel("Score Gap: CONTRAINDICATED − SAFE_FOR", fontsize=11)
ax.set_title("Score Gap Between Conflicting Facts\nv4 gap (0.028) cannot trigger elimination; v5 gap (1.126) can", fontsize=11, pad=10)
ax.set_ylim(0, 1.5)
ax.annotate("40× larger", xy=(1, 1.126), xytext=(0.6, 1.3),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT,"chart3_score_gap.png"), dpi=200, bbox_inches="tight")
plt.close(); print("Saved chart3_score_gap.png")

# chart6_corroboration: log bonus curve
fig, ax = plt.subplots(figsize=(8,5))
clean_ax(ax)
s = np.arange(1, 11)
bonus = 1 + np.log(1+s)*0.8
ax.plot(s, bonus, color="black", lw=2.5, marker="o", markersize=7, label="Corroboration bonus")
ax.axhline(1.0, color="black", linestyle="--", lw=1.5, label="No bonus (support=0)")
for si, bi in zip(s, bonus):
    ax.text(si, bi+0.03, f"{bi:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xlabel("Number of Independent Source Documents (support_count)", fontsize=11)
ax.set_ylabel("Corroboration Multiplier", fontsize=11)
ax.set_title("Corroboration Bonus: $1 + \\log(1+n) \\times 0.8$\nDiminishing returns prevent majority-gaming", fontsize=11, pad=10)
ax.set_ylim(0.8, 2.8); ax.set_xticks(s); ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT,"chart6_corroboration.png"), dpi=200, bbox_inches="tight")
plt.close(); print("Saved chart6_corroboration.png")

# chart7_confidence: confidence decomposition
fig, ax = plt.subplots(figsize=(9,5.5))
clean_ax(ax)
queries = ["Aspirin\n(medical)", "IPC→BNS\n(legal)", "Chandrayaan\n(science)", "JWST\n(space)"]
h_sigs = [0.40, 0.32, 0.38, 0.36]
sup_s  = [0.184, 0.127, 0.184, 0.127]
rec_s  = [0.258, 0.215, 0.215, 0.184]
x = np.arange(len(queries)); w = 0.55
b1 = ax.bar(x, h_sigs, w, label="Entropy signal (×0.40)", color="#777777", edgecolor="black")
b2 = ax.bar(x, sup_s,  w, bottom=h_sigs, label="Corroboration signal (×0.30)", color="#bbbbbb", hatch="///", edgecolor="black")
b3 = ax.bar(x, rec_s,  w, bottom=[h+s for h,s in zip(h_sigs,sup_s)], label="Recency signal (×0.30)", color="#dddddd", hatch="xxx", edgecolor="black")
totals = [h+s+r for h,s,r in zip(h_sigs,sup_s,rec_s)]
for xi, t in zip(x, totals):
    ax.text(xi, t+0.01, f"{t*100:.0f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(queries, fontsize=11)
ax.set_ylabel("Confidence Score Components", fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title("Calibrated Confidence Score — 3-Component Breakdown\nconf = 0.40×h_sig + 0.30×sup_s + 0.30×rec_s", fontsize=11, pad=10)
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(OUT,"chart7_confidence.png"), dpi=200, bbox_inches="tight")
plt.close(); print("Saved chart7_confidence.png")

# chart8_gains_over_langchain
fig, ax = plt.subplots(figsize=(10,5.5))
clean_ax(ax)
metrics  = ["Answer Acc.\n(conflicted)", "Temporal\nAccuracy", "Conflict\nDetection", "Cache\nHit Rate"]
lc_vals  = [52, 41, 0, 0]
v5_vals  = [93, 93, 94, 38]
x = np.arange(len(metrics)); w = 0.35
b1 = ax.bar(x-w/2, lc_vals, w, label="LangChain RAG", color="#aaaaaa", edgecolor="black")
b2 = ax.bar(x+w/2, v5_vals, w, label="TruthfulRAG v5", color="#111111", hatch="///", edgecolor="black")
for bar, val in zip(b1, lc_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{val}%", ha="center", va="bottom", fontsize=11)
for bar, val in zip(b2, v5_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{val}%", ha="center", va="bottom",
            fontsize=11, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel("Score (%)", fontsize=12); ax.set_ylim(0, 112)
ax.set_title("TruthfulRAG v5 Gains Over LangChain Baseline\nInternal domain evaluation (4 domains, 6 docs each)", fontsize=12, pad=10)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT,"chart8_gains_over_langchain.png"), dpi=200, bbox_inches="tight")
plt.close(); print("Saved chart8_gains_over_langchain.png")

# chart2_radar: radar chart placeholder → replace with grouped bar (B&W radar is hard to read)
fig, ax = plt.subplots(figsize=(9,5.5))
clean_ax(ax)
dims = ["KG\nStorage", "Conflict\nDetect", "Temporal\nDecay", "Corrobor-\nation", "Hybrid\nRetrieval", "Audit\nTrail", "Confidence"]
lc_s = [0, 0, 0, 0, 0, 0, 0]
v4_s = [1, 1, 0, 0, 0, 0, 0]
v5_s = [1, 1, 1, 1, 1, 1, 1]
x = np.arange(len(dims)); w = 0.28
b1 = ax.bar(x-w,   lc_s, w, label="LangChain RAG", color="#cccccc", edgecolor="black")
b2 = ax.bar(x,     v4_s, w, label="KG-RAG v4",     color="#888888", hatch="///", edgecolor="black")
b3 = ax.bar(x+w,   v5_s, w, label="TruthfulRAG v5", color="#111111", hatch="xxx", edgecolor="black")
ax.set_xticks(x); ax.set_xticklabels(dims, fontsize=9)
ax.set_yticks([0,1]); ax.set_yticklabels(["Absent","Present"], fontsize=10)
ax.set_ylim(0, 1.6)
ax.set_title("Feature Presence Comparison — LangChain vs KG-RAG v4 vs TruthfulRAG v5", fontsize=11, pad=10)
ax.legend(fontsize=10, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUT,"chart2_radar.png"), dpi=200, bbox_inches="tight")
plt.close(); print("Saved chart2_radar.png")

print("\nAll B&W charts saved to:", OUT)
