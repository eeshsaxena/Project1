"""
Compile all report graphs into one PDF — one per page with figure numbering.
Output: d:\Project-1\report_graphs.pdf
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

BASE  = r"d:\Project-1\Report_Format___BTech_CSE_IIIT_Manipur__1___1_\report_file"
OUT   = r"d:\Project-1\report_graphs.pdf"

# ── All report figures in chapter order ──────────────────
FIGURES = [
    ("chart_timeline.png",
     "Figure 1 (Ch.2)",
     "RAG research evolution timeline: 2020--2026, positioning TruthfulRAG v5"),

    ("pipeline_diagram.png",
     "Figure 2 (Ch.4)",
     "End-to-end pipeline flow of TruthfulRAG v5"),

    ("chart_halueval_hrr.png",
     "Figure 3 (Ch.6)",
     "HaluEval: Hallucination Rejection Rate vs. Exact Match across three systems"),

    ("chart_cdr.png",
     "Figure 4 (Ch.6)",
     "Conflict Detection Rate: LangChain = 25%, v4 = 25%, v5 = 50%"),

    ("chart_benchmark_summary.png",
     "Figure 5 (Ch.6)",
     "Complete benchmark summary: HRR%, CDR%, and Temporal Accuracy% across all systems"),

    ("chart1_grouped_bar.png",
     "Figure 6 (Ch.6)",
     "Answer accuracy by question type across three systems"),

    ("chart2_radar.png",
     "Figure 7 (Ch.6)",
     "Six-dimension capability radar: LangChain vs. v4 vs. v5"),

    ("chart3_score_gap.png",
     "Figure 8 (Ch.6)",
     "Score gap comparison: LangChain=0.000, v4=0.028, v5=1.126"),

    ("chart4_temporal_decay.png",
     "Figure 9 (Ch.6)",
     "Fact retrieval weight vs. age (lambda=0.08). A 2006 fact retains 60% weight; a 1985 fact retains 3%"),

    ("chart6_corroboration.png",
     "Figure 10 (Ch.6)",
     "Corroboration bonus vs. source support count (log-scale)"),

    ("chart7_confidence.png",
     "Figure 11 (Ch.6)",
     "Confidence score breakdown across four queries"),

    ("chart8_gains_over_langchain.png",
     "Figure 12 (Ch.6)",
     "Accuracy gain of TruthfulRAG v5 over LangChain baseline across all domains"),
]

with PdfPages(OUT) as pdf:

    # ── Cover ─────────────────────────────────────────────
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("#0a1628")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0a1628")
    ax.axis("off")

    ax.text(0.5, 0.72, "TruthfulRAG v5",
            ha="center", fontsize=40, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.60, "Report Figures",
            ha="center", fontsize=22, color="#4fc3f7",
            transform=ax.transAxes)
    ax.text(0.5, 0.50,
            "Conflict-Aware Knowledge-Graph RAG System",
            ha="center", fontsize=14, color="#a0b4d0",
            transform=ax.transAxes)
    ax.text(0.5, 0.38,
            "Eesh Saxena  |  230101032  |  B.Tech CSE  |  IIIT Manipur",
            ha="center", fontsize=13, color="#708090",
            transform=ax.transAxes)
    ax.text(0.5, 0.26,
            f"Total Figures: {len(FIGURES)}  |  April 2026",
            ha="center", fontsize=11, color="#4fc3f7",
            transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── One figure per page ───────────────────────────────
    found = 0
    missing = []

    for filename, fig_num, caption in FIGURES:
        path = os.path.join(BASE, filename)
        if not os.path.exists(path):
            missing.append(filename)
            print(f"  SKIP: {filename}")
            continue

        img = mpimg.imread(path)
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("white")

        # Figure number badge top-left
        fig.text(0.03, 0.96, fig_num,
                 ha="left", va="top",
                 fontsize=11, fontweight="bold",
                 color="white",
                 bbox=dict(boxstyle="round,pad=0.3",
                           fc="#1565c0", ec="none"))

        # Caption top-center
        fig.text(0.5, 0.966, caption,
                 ha="center", va="top",
                 fontsize=10.5, color="#1a237e",
                 wrap=True)

        # Thin line under header
        ax_line = fig.add_axes([0.03, 0.935, 0.94, 0.002])
        ax_line.set_facecolor("#1565c0")
        ax_line.axis("off")

        # Image
        ax_img = fig.add_axes([0.03, 0.04, 0.94, 0.885])
        ax_img.imshow(img)
        ax_img.axis("off")

        # Footer
        fig.text(0.5, 0.012,
                 "TruthfulRAG v5  |  Eesh Saxena 230101032  |  IIIT Manipur  |  April 2026",
                 ha="center", va="bottom",
                 fontsize=7.5, color="#888888")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        found += 1
        print(f"  [OK] {fig_num}: {filename}")

    # ── PDF metadata ──────────────────────────────────────
    d = pdf.infodict()
    d["Title"]   = "TruthfulRAG v5 -- Report Figures"
    d["Author"]  = "Eesh Saxena 230101032 IIIT Manipur"
    d["Subject"] = "BTech Project Viva -- April 2026"

print(f"\n{'='*52}")
print(f"[DONE] PDF: {OUT}")
print(f"       Pages   : {found + 1} (cover + {found} figures)")
if missing:
    print(f"       Missing : {missing}")
print(f"{'='*52}")
