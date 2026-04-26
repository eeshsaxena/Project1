"""
clean_notebooks.py
Programmatically fixes both Jupyter notebooks for viva presentation:
  - TruthfulRAG_Evaluation.ipynb  : clears cell-20 error, fixes export path
  - metrics_comparison.ipynb      : removes 4 empty cells, adds markdown
                                    section headers, converts dark→white theme
Run once:  python clean_notebooks.py
"""

import json, copy, os, re

OUT_DIR = r"d:\Project-1\Report_Format___BTech_CSE_IIIT_Manipur__1___1_\report_file"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_nb(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return json.load(f)

def save_nb(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Saved: {path}")

def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text], "outputs": []}

def empty_code_cell():
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "source": [], "outputs": []}

def strip_dark_theme(source_lines):
    """Replace dark-background colour strings with white equivalents."""
    text = "".join(source_lines)
    replacements = {
        '"#0f1117"': '"white"',
        '"#1a1d27"': '"white"',
        '"#1e2130"': '"white"',
        '"#ffffff"': '"black"',   # text that was white-on-dark → black
        "'#0f1117'": "'white'",
        "'#1a1d27'": "'white'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return [text]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Fix TruthfulRAG_Evaluation.ipynb
# ─────────────────────────────────────────────────────────────────────────────

NB1 = r"d:\Project-1\TruthfulRAG_Evaluation.ipynb"
print("=== Fixing TruthfulRAG_Evaluation.ipynb ===")
nb1 = load_nb(NB1)

for i, cell in enumerate(nb1["cells"]):
    src = "".join(cell.get("source", []))

    # Fix cell 20: clear error output, fix export path
    if i == 20 and cell.get("cell_type") == "code":
        cell["outputs"] = []          # wipe error output
        cell["execution_count"] = None
        # Fix the OUT path to absolute
        new_src = src.replace(
            "Report_Format___BTech_CSE_IIIT_Manipur__1___1_/report_file",
            OUT_DIR.replace("\\", "\\\\")
        ).replace(
            r"Report_Format___BTech_CSE_IIIT_Manipur__1___1_\\report_file",
            OUT_DIR.replace("\\", "\\\\")
        )
        cell["source"] = [new_src]
        print(f"  [Cell {i}] Cleared error output, fixed export path.")

    # Strip dark theme colours from all code cells
    if cell.get("cell_type") == "code" and cell.get("source"):
        cell["source"] = strip_dark_theme(cell["source"])

save_nb(nb1, NB1)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Fix metrics_comparison.ipynb
# ─────────────────────────────────────────────────────────────────────────────

NB2 = r"d:\Project-1\metrics_comparison.ipynb"
print("\n=== Fixing metrics_comparison.ipynb ===")
nb2 = load_nb(NB2)

# Section titles for each code cell (by position)
section_headers = {
    0:  None,   # dependency install – no header needed
    1:  "## Setup — Imports & Plot Style",
    2:  "## Data — Benchmark Results (All Systems)",
    3:  "## Figure 1 — Answer Accuracy Grouped Bar Chart",
    4:  "## Figure 2 — Radar / Feature Comparison Chart",
    5:  "## Figure 3 — Score Gap (v4 vs v5)",
    6:  "## Figure 4 — Temporal Decay Curve",
    7:  "## Figure 5 — Confidence Score Decomposition",
    8:  "## Figure 6 — Corroboration Bonus Curve",
    9:  "## Figure 7 — Confidence Scenario Comparison",
    10: "## Figure 8 — Gains Over LangChain",
    11: "## Summary Table — All Metrics",
}

new_cells = []
code_idx = 0
for cell in nb2["cells"]:
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if not src.strip():               # skip empty cells
            print(f"  Removed empty code cell (index {code_idx})")
            code_idx += 1
            continue
        # Insert markdown header before this code cell if defined
        header = section_headers.get(code_idx)
        if header:
            new_cells.append(md_cell(header))
            print(f"  Added header '{header}' before code cell {code_idx}")
        # Fix dark theme
        cell["source"] = strip_dark_theme(cell["source"])
        new_cells.append(cell)
        code_idx += 1
    else:
        new_cells.append(cell)

nb2["cells"] = new_cells
save_nb(nb2, NB2)

print("\nDone. Both notebooks are clean and ready for presentation.")
print("Re-run both notebooks top-to-bottom in Jupyter to refresh all outputs.")
