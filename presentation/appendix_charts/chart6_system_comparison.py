"""Appendix chart 6 — system comparison + validation/test selection noise.

Standalone and re-runnable. All numbers inline, transcribed from:
  scripts/model/results/gemma_lora_v2_lora_dora_ablation/test_evaluation/REPORT.md
  scripts/model/results/gemma_lora_v2_lora_dora_ablation/REPORT.md

Run with: /home/novaktj/aegis_env/bin/python chart6_system_comparison.py
"""

import os

import altair as alt
import pandas as pd
import vl_convert as vlc

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "chart6_system_comparison.svg")

# --- Panel A: unified 8-system test table (n=227: 127 harmful / 100 unharmful) ---
SYSTEMS = [
    # (system, f1, recall, fpr)
    ("Qwen3Guard (zero-shot)",   0.953125, 0.960630, 0.07),
    ("LoRA attention-only",      0.937500, 0.944882, 0.09),
    ("DoRA attention-only",      0.929688, 0.937008, 0.10),
    ("Gemma sweep LoRA",         0.916335, 0.905512, 0.09),
    ("Gemma initial LoRA",       0.916031, 0.944882, 0.15),
    ("DoRA all-linear",          0.906977, 0.921260, 0.14),
    ("Gemma zero-shot",          0.808362, 0.913386, 0.44),
    ("Regex baseline",           0.057554, 0.031496, 0.08),
]

order = [s[0] for s in SYSTEMS]  # already F1-descending
rows = []
for name, f1, rec, fpr in SYSTEMS:
    rows.append({"system": name, "metric": "F1 (higher is better)", "value": f1})
    rows.append({"system": name, "metric": "Harmful recall (higher is better)", "value": rec})
    rows.append({"system": name, "metric": "FPR (LOWER is better)", "value": fpr})
dfa = pd.DataFrame(rows)

metric_order = ["F1 (higher is better)",
                "Harmful recall (higher is better)",
                "FPR (LOWER is better)"]
metric_colors = ["#2f6f9f", "#6aa84f", "#c2513a"]

base_a = alt.Chart(dfa).encode(
    y=alt.Y("system:N", sort=order, title=None,
            axis=alt.Axis(labelFontSize=11, labelLimit=200)),
    yOffset=alt.YOffset("metric:N", sort=metric_order),
)

bars_a = base_a.mark_bar(height=8).encode(
    x=alt.X("value:Q", title="Rate (0–1)",
            scale=alt.Scale(domain=[0, 1.0], nice=False),
            axis=alt.Axis(format=".1f", values=[0, 0.2, 0.4, 0.6, 0.8, 1.0])),
    color=alt.Color("metric:N", sort=metric_order,
                    scale=alt.Scale(domain=metric_order, range=metric_colors),
                    legend=alt.Legend(title=None, orient="bottom",
                                      direction="vertical", labelFontSize=11)),
)

labels_a = base_a.mark_text(align="left", dx=3, fontSize=8.5, color="#333").encode(
    x=alt.X("value:Q"),
    text=alt.Text("value:Q", format=".3f"),
)

panel_a = (bars_a + labels_a).properties(
    width=470, height=330,
    title=alt.TitleParams(
        text="A. Test-set comparison of all 8 systems (n = 227; 127 harmful / 100 unharmful)",
        subtitle=[
            "Sorted by F1 descending. Blue/green: higher is better. Red (FPR): lower is better.",
            "Regex baseline is a train-derived keyword baseline: it mostly fires on character-spacing",
            "obfuscation, not on harm — recall 0.031, F1 0.058, i.e. near-total failure.",
        ],
        fontSize=13, subtitleFontSize=10, anchor="start", subtitleColor="#555",
    ),
)

# --- Panel B: validation (n=259) -> test (n=227) selection noise ---
ARMS = [
    # (arm, val_f1, test_f1, trainable_params, test_rank_of_8)
    ("DoRA all-linear",     0.953271, 0.906977, 6_982_144, 6),
    ("DoRA attention-only", 0.948718, 0.929688, 1_560_832, 3),
    ("LoRA attention-only", 0.943396, 0.937500, 1_490_944, 2),
    ("LoRA all-linear (Exp1)", 0.941538, 0.916031, 6_522_880, 5),
]

recs = []
for arm, vf1, tf1, params, rank in ARMS:
    for split, val in (("Validation\n(n = 259)", vf1), ("Test\n(n = 227)", tf1)):
        recs.append({"arm": arm, "split": split, "f1": val,
                     "params": params,
                     "params_m": f"{params/1e6:.1f}M",
                     "rank": rank})
dfb = pd.DataFrame(recs)
split_order = ["Validation\n(n = 259)", "Test\n(n = 227)"]
arm_order = [a[0] for a in ARMS]
arm_colors = ["#c2513a", "#7b6ca8", "#2f6f9f", "#999999"]

base_b = alt.Chart(dfb).encode(
    x=alt.X("split:N", sort=split_order, title=None,
            scale=alt.Scale(padding=0.42),
            axis=alt.Axis(labelFontSize=11, labelAngle=0)),
    y=alt.Y("f1:Q", title="F1 (higher is better)",
            scale=alt.Scale(domain=[0.89, 0.965], nice=False)),
    color=alt.Color("arm:N", sort=arm_order,
                    scale=alt.Scale(domain=arm_order, range=arm_colors),
                    legend=alt.Legend(title="Ablation arm", orient="bottom",
                                      labelFontSize=10, titleFontSize=10,
                                      direction="vertical")),
    detail="arm:N",
)

lines_b = base_b.mark_line(strokeWidth=2.2, point=False)
points_b = base_b.mark_point(filled=True, size=110, opacity=1)

left_lab = base_b.transform_filter(
    alt.datum.split == split_order[0]
).mark_text(align="right", dx=-9, fontSize=9.5).encode(
    text=alt.Text("label:N"),
    # nudge the two nearly-tied validation labels apart so they don't collide
    yOffset=alt.YOffset("nudge:Q", scale=None),
).transform_calculate(
    nudge="datum.arm == 'LoRA attention-only' ? -6 : (datum.arm == 'LoRA all-linear (Exp1)' ? 7 : 0)"
).transform_calculate(
    label="datum.arm + '  ' + format(datum.f1, '.4f') + '  (' + datum.params_m + ' params)'"
)

right_lab = base_b.transform_filter(
    alt.datum.split == split_order[1]
).mark_text(align="left", dx=9, fontSize=9.5).encode(
    text=alt.Text("label:N")
).transform_calculate(
    label="format(datum.f1, '.4f') + '  — test rank ' + datum.rank + '/8'"
)

panel_b = (lines_b + points_b + left_lab + right_lab).properties(
    width=210, height=330,
    title=alt.TitleParams(
        text="B. Selection noise: validation winner is not the test winner",
        subtitle=[
            "DoRA all-linear won on validation (0.9533) but fell to 6th of 8 on test (0.9070).",
            "LoRA attention-only did the reverse (0.9434 -> 0.9375, 2nd of 8) with ~4.7x fewer",
            "trainable parameters (1.49M vs 6.98M). Lines cross — validation ranking did not hold.",
        ],
        fontSize=13, subtitleFontSize=10, anchor="start", subtitleColor="#555",
    ),
)

chart = alt.hconcat(panel_a, panel_b, spacing=95).resolve_scale(
    color="independent"
).configure_view(strokeOpacity=0).configure_axis(
    labelFontSize=10, titleFontSize=11, grid=True, gridColor="#e8e8e8"
).properties(
    title=alt.TitleParams(
        text="",
        subtitle=[],
    )
)

svg = vlc.vegalite_to_svg(chart.to_json())
with open(OUT, "w") as f:
    f.write(svg)
print("wrote", OUT)
