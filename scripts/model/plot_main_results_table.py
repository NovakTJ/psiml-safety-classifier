#!/usr/bin/env python
"""Render the main test-set results as a styled table image ->
/home/mls01/main_results_table.png (repo root).

Same 5 systems/numbers as plot_main_results.py (regex baseline dropped, best-to-worst
by F1), realized as a table instead of a bar chart. Pure CPU (matplotlib only).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_PNG = "/home/mls01/main_results_table.png"

# (name, F1, recall) -- best to worst by F1. No row is highlighted: on pure F1/recall
# Qwen3Guard is #1, not the ensemble -- the ensemble's actual selling point (~half the
# inference cost at matching recall) belongs on the routing-rate chart, not here.
SYSTEMS = [
    ("Qwen3Guard",       0.9531, 0.9606),
    ("Ensemble",         0.9385, 0.9606),
    ("Gemma LoRA",       0.9163, 0.9055),
    ("Linear probe",     0.9091, 0.9843),
    ("Gemma zero-shot",  0.8084, 0.9134),
]

HEADER_BG = "#1f77b4"
HEADER_FG = "white"
ROW_BG_EVEN = "#f2f2f2"
ROW_BG_ODD = "white"

col_labels = ["System", "F1", "Harmful recall"]
n_rows = len(SYSTEMS) + 1  # + header
n_cols = len(col_labels)

fig, ax = plt.subplots(figsize=(7, 0.55 * n_rows), dpi=170)
ax.axis("off")

table = ax.table(cellText=[[name, f"{f1:.3f}", f"{rec:.3f}"] for name, f1, rec in SYSTEMS],
                 colLabels=col_labels, cellLoc="center", colLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.0)

# Header row styling.
for c in range(n_cols):
    cell = table[0, c]
    cell.set_facecolor(HEADER_BG)
    cell.get_text().set_color(HEADER_FG)
    cell.get_text().set_weight("bold")

# Body row styling: plain alternating banding only, no row is called out as "the winner".
for r, (name, f1, rec) in enumerate(SYSTEMS, start=1):
    bg = ROW_BG_EVEN if r % 2 == 0 else ROW_BG_ODD
    for c in range(n_cols):
        table[r, c].set_facecolor(bg)

fig.suptitle("PSIML safety classifier — test set comparison\n"
            "(227-row held-out test set, harmful = positive, end-to-end metrics)",
            fontsize=12, y=0.98)
fig.subplots_adjust(top=0.78)
fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.15)
print(f"Saved: {OUT_PNG}")
