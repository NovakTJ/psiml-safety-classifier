#!/usr/bin/env python
"""Plot the main test-set results comparison -> /home/mls01/main_results.png (repo root).

Grouped bar chart, F1 + recall, 5 systems (regex baseline dropped -- its F1 0.058 would
flatten the rest of the chart), sorted best-to-worst by F1. All numbers read from the
already-verified test-evaluation files (results/{gemma_demo_zeroshot_v2_no_refusal,
gemma_lora_v2_sweep_phase2,linear_probe_v3_locked/../test_evaluation,
ensemble_v2_final_maxf1,qwen3guard_native_v2_no_refusal}) via
gemma_lora_v2_lora_dora_ablation/test_evaluation/all_systems_test_comparison.csv for the
cross-checked end-to-end methodology. Pure CPU (matplotlib only, no GPU/model load).

Styled to match the teammate's plot_learning_curve_v2.py (figsize, dpi, colors, grid).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_PNG = "/home/mls01/main_results.png"

# (name, F1, recall) -- best to worst by F1. Regex baseline (F1 0.058) intentionally
# excluded (see docstring).
SYSTEMS = [
    ("Qwen3Guard",       0.9531, 0.9606),
    ("Ensemble",         0.9385, 0.9606),
    ("Gemma LoRA",       0.9163, 0.9055),
    ("Linear probe",     0.9091, 0.9843),
    ("Gemma zero-shot",  0.8084, 0.9134),
]

names = [s[0] for s in SYSTEMS]
f1s = [s[1] for s in SYSTEMS]
recalls = [s[2] for s in SYSTEMS]

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=170)

x = np.arange(len(names))
width = 0.35

bars_f1 = ax.bar(x - width / 2, f1s, width, label="F1", color="#2a78d6", zorder=3)
bars_rec = ax.bar(x + width / 2, recalls, width, label="Harmful recall", color="#1baf7a",
                  zorder=3)

for bars in (bars_f1, bars_rec):
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h),
                   xytext=(0, 3), textcoords="offset points",
                   ha="center", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0.6, 1.0)
ax.set_ylabel("score")
ax.set_title("Safety classifiers — test set comparison")
ax.grid(True, axis="y", alpha=0.3)
ax.set_axisbelow(True)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_PNG)
print(f"Saved: {OUT_PNG}")
