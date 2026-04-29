import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

BASE = r"d:\Project-1\Report_Format___BTech_CSE_IIIT_Manipur__1___1_\report_file"
OUT  = r"d:\Project-1\selected_figures.pdf"

FIGURES = [
    ("chart_benchmark_summary.png",
     "Figure 6.3",
     "Complete benchmark summary: HRR%, CDR%, and Temporal Accuracy% across all three systems"),

    ("chart6_corroboration.png",
     "Figure 6.9",
     "Corroboration bonus vs. source support count"),

    ("chart7_confidence.png",
     "Figure 6.10",
     "Confidence score breakdown across four queries"),

    ("chart8_gains_over_langchain.png",
     "Figure 6.11",
     "Accuracy gain of TruthfulRAG v5 over the LangChain RAG baseline across all domains"),
]

with PdfPages(OUT) as pdf:
    for filename, fig_num, caption in FIGURES:
        path = os.path.join(BASE, filename)
        img  = mpimg.imread(path)

        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("white")

        # Number badge
        fig.text(0.03, 0.965, fig_num,
                 ha="left", va="top",
                 fontsize=13, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.35", fc="#1565c0", ec="none"))

        # Caption
        fig.text(0.5, 0.968, caption,
                 ha="center", va="top",
                 fontsize=11, color="#1a237e")

        # Divider line
        ax_line = fig.add_axes([0.03, 0.933, 0.94, 0.002])
        ax_line.set_facecolor("#1565c0")
        ax_line.axis("off")

        # Image — full page
        ax_img = fig.add_axes([0.05, 0.04, 0.90, 0.88])
        ax_img.imshow(img)
        ax_img.axis("off")

        # Footer
        fig.text(0.5, 0.012,
                 "TruthfulRAG v5  |  Eesh Saxena 230101032  |  IIIT Manipur  |  April 2026",
                 ha="center", va="bottom", fontsize=8, color="#888888")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {fig_num} added")

    d = pdf.infodict()
    d["Title"]  = "TruthfulRAG v5 -- Figures 6.3, 6.9, 6.10, 6.11"
    d["Author"] = "Eesh Saxena 230101032 IIIT Manipur"

print(f"\n[DONE] {OUT}")
print("Pages: 4 (Figures 6.3, 6.9, 6.10, 6.11)")
