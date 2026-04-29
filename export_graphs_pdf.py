"""
Compile all TruthfulRAG v5 charts into a single PDF.
Run: python export_graphs_pdf.py
Output: all_graphs.pdf
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

# ── Charts in display order ───────────────────────────────
charts = [
    ("chart1_grouped_bar.png",          "Figure 1 — Grouped Bar: HRR & EM Comparison (LangChain vs v4 vs v5)"),
    ("chart2_radar.png",                "Figure 2 — Radar Chart: 6-Dimension Capability Comparison"),
    ("chart3_score_gap.png",            "Figure 3 — Score Gap: v4 (0.028) vs v5 (1.126)"),
    ("chart4_temporal_decay.png",       "Figure 4 — Temporal Decay Curve: e^(-0.08 × age)"),
    ("chart5_formula_heatmap.png",      "Figure 5 — Formula Heatmap: Combined Score Components"),
    ("chart6_corroboration.png",        "Figure 6 — Corroboration Bonus: Log-Scale Support Effect"),
    ("chart7_confidence.png",           "Figure 7 — Confidence Score Distribution"),
    ("chart8_gains_over_langchain.png", "Figure 8 — Gains Over LangChain Baseline"),
    ("pipeline_diagram.png",            "Figure 9 — TruthfulRAG v5 Pipeline Diagram"),
    ("chart_cdr.png",                   "Figure 10 — Conflict Detection Rate: LangChain vs v4 vs v5"),
]

BASE = r"d:\Project-1"
OUT  = os.path.join(BASE, "all_graphs.pdf")

with PdfPages(OUT) as pdf:

    # ── Cover page ────────────────────────────────────────
    fig = plt.figure(figsize=(11.69, 8.27))   # A4 landscape
    fig.patch.set_facecolor("#0a1628")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0a1628")
    ax.axis("off")

    ax.text(0.5, 0.65,
            "TruthfulRAG v5",
            ha="center", va="center",
            fontsize=36, fontweight="bold",
            color="white",
            transform=ax.transAxes)

    ax.text(0.5, 0.52,
            "Conflict-Aware Knowledge-Graph RAG System",
            ha="center", va="center",
            fontsize=16, color="#a0b4d0",
            transform=ax.transAxes)

    ax.text(0.5, 0.40,
            "Evaluation Graphs & Results",
            ha="center", va="center",
            fontsize=20, fontweight="bold",
            color="#4fc3f7",
            transform=ax.transAxes)

    ax.text(0.5, 0.28,
            "Eesh Saxena  |  230101032  |  B.Tech CSE  |  IIIT Manipur",
            ha="center", va="center",
            fontsize=13, color="#708090",
            transform=ax.transAxes)

    ax.text(0.5, 0.18,
            f"Total Figures: {sum(1 for f,_ in charts if os.path.exists(os.path.join(BASE, f)))}  |  April 2026",
            ha="center", va="center",
            fontsize=11, color="#556070",
            transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Cover page added")

    # ── One chart per page ────────────────────────────────
    found = 0
    missing = []

    for filename, title in charts:
        path = os.path.join(BASE, filename)
        if not os.path.exists(path):
            missing.append(filename)
            print(f"  SKIP (not found): {filename}")
            continue

        img = mpimg.imread(path)
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("white")

        # Title bar at top
        fig.text(0.5, 0.97, title,
                 ha="center", va="top",
                 fontsize=13, fontweight="bold",
                 color="#0a1628")

        # Thin coloured line under title
        ax_line = fig.add_axes([0.05, 0.935, 0.90, 0.003])
        ax_line.set_facecolor("#1565c0")
        ax_line.axis("off")

        # Image
        ax_img = fig.add_axes([0.03, 0.03, 0.94, 0.89])
        ax_img.imshow(img)
        ax_img.axis("off")

        # Footer
        fig.text(0.5, 0.01,
                 "TruthfulRAG v5  |  Eesh Saxena 230101032  |  IIIT Manipur",
                 ha="center", va="bottom",
                 fontsize=8, color="#888888")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        found += 1
        print(f"  [OK] Added: {filename}")

    # ── Summary page ──────────────────────────────────────
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("#0a1628")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0a1628")
    ax.axis("off")

    ax.text(0.5, 0.82,
            "Key Results Summary",
            ha="center", va="center",
            fontsize=28, fontweight="bold",
            color="white",
            transform=ax.transAxes)

    results = [
        ("HRR (Hallucination Rejection)",  "77.5%",  "91.0%",  "92.0%",  "+14.5 pp"),
        ("CDR (Conflict Detection)",        "25.0%",  "25.0%",  "50.0%",  "+25.0 pp"),
        ("Temporal Accuracy",               "14.3%",  "35.7%",  "35.7%",  "+21.4 pp"),
        ("Score Gap (conflict resolution)", "—",      "0.028",  "1.126",  "40× larger"),
        ("Avg Query Time",                  "0.47s",  "19.1s",  "4.31s",  "4.4× faster"),
        ("Confidence Score",                "None",   "None",   "84%",    "Novel"),
    ]

    col_x   = [0.04, 0.34, 0.48, 0.62, 0.76]
    headers = ["Metric", "LangChain", "v4", "v5", "Gain"]

    y = 0.70
    for i, h in enumerate(headers):
        ax.text(col_x[i], y, h,
                ha="left", fontsize=11, fontweight="bold",
                color="#4fc3f7", transform=ax.transAxes)

    y -= 0.04
    ax.plot([0.04, 0.96], [y, y], color="#4fc3f7",
            lw=0.8, transform=ax.transAxes, clip_on=False)

    for row in results:
        y -= 0.07
        colors = ["white", "#9e9e9e", "#bdbdbd", "#4fc3f7", "#a5d6a7"]
        for i, (val, col) in enumerate(zip(row, colors)):
            ax.text(col_x[i], y, val,
                    ha="left", fontsize=10,
                    color=col, transform=ax.transAxes)

    ax.text(0.5, 0.04,
            "All results on 325 queries  |  HaluEval (200) + ConflictQA/WebQ/SQuADv2 (125)",
            ha="center", va="bottom",
            fontsize=9, color="#556070",
            transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Summary page added")

    # ── PDF metadata ──────────────────────────────────────
    d = pdf.infodict()
    d["Title"]   = "TruthfulRAG v5 — Evaluation Graphs"
    d["Author"]  = "Eesh Saxena 230101032 IIIT Manipur"
    d["Subject"] = "BTech Project Viva — April 2026"

print(f"\n{'='*50}")
print(f"[DONE] PDF saved: {OUT}")
print(f"   Charts included : {found}")
if missing:
    print(f"   Charts missing  : {missing}")
print(f"{'='*50}")
