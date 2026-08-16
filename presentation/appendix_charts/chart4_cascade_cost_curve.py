#!/usr/bin/env python
"""Appendix chart 4: cascade cost vs deployment prevalence, and the precision counterpoint.

Data inline; all curves derived in closed form from the measured, prevalence-independent
stage rates (r_H, r_B, b_H, b_B) reported on the v2 validation split (n=259):
  scripts/model/results/ensemble_v2_final/REPORT.md         (objective: constrained_cost)
  scripts/model/results/ensemble_v2_final_maxf1/REPORT.md   (objective: max_f1)

Run with the aegis venv:
  /home/novaktj/aegis_env/bin/python chart4_cascade_cost_curve.py
"""
import json
import os

import altair as alt
import numpy as np
import pandas as pd
import vl_convert as vlc

OUT_SVG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "chart4_cascade_cost_curve.svg")

# --- measured stage rates (validation, n=259) -------------------------------
CONFIGS = {
    "constrained_cost (recall>=0.95)": dict(
        r_H=0.9500, r_B=0.1212, b_H=1.0000, b_B=0.1667,
        recall=0.9500, fpr=0.0202,
    ),
    "max_f1": dict(
        r_H=0.9812, r_B=0.3636, b_H=1.0000, b_B=0.0556,
        recall=0.9812, fpr=0.0202,
    ),
}
TABULATED_PI = [0.001, 0.01, 0.05, 0.2]   # the pi values tabulated in both REPORTs
KAPPA = 1e-3                              # c_probe / c_clf, upper end of the ~1e-4..1e-3 range


def routing_rate(pi, c):
    return pi * c["r_H"] + (1.0 - pi) * c["r_B"]


def precision(pi, c):
    R, F = c["recall"], c["fpr"]
    return pi * R / (pi * R + (1.0 - pi) * F)


pi_grid = np.logspace(np.log10(0.001), np.log10(0.5), 400)

curve_rows, point_rows = [], []
for name, c in CONFIGS.items():
    for pi in pi_grid:
        curve_rows.append(dict(config=name, pi=float(pi),
                               cost=float(KAPPA + routing_rate(pi, c)),
                               precision=float(precision(pi, c))))
    for pi in TABULATED_PI:
        point_rows.append(dict(config=name, pi=float(pi),
                               cost=float(KAPPA + routing_rate(pi, c)),
                               precision=float(precision(pi, c))))

curves = pd.DataFrame(curve_rows)
points = pd.DataFrame(point_rows)

COLORS = ["#1f77b4", "#d62728"]
color = alt.Color("config:N", title="config (selected on validation)",
                  scale=alt.Scale(domain=list(CONFIGS), range=COLORS),
                  legend=alt.Legend(orient="bottom", direction="horizontal",
                                    titleLimit=300, labelLimit=300))
x = alt.X("pi:Q",
          scale=alt.Scale(type="log", domain=[0.001, 0.5], nice=False),
          axis=alt.Axis(title="deployment harmful prevalence  π  (log scale)",
                        values=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                        format=".3~f", grid=True))

W, H = 430, 330

# --- Panel A: relative cost -------------------------------------------------
a_line = alt.Chart(curves).mark_line(strokeWidth=2.5).encode(
    x=x,
    y=alt.Y("cost:Q", scale=alt.Scale(type="log", domain=[0.05, 1.6], nice=False),
            axis=alt.Axis(title="cost per exchange  (1.0 = always-on classifier)",
                          format=".2~f", values=[0.05, 0.1, 0.2, 0.5, 1.0])),
    color=color,
)
a_pts = alt.Chart(points).mark_point(filled=True, size=70, opacity=1).encode(
    x=x, y="cost:Q", color=color,
    tooltip=["config:N", alt.Tooltip("pi:Q", format=".3f"),
             alt.Tooltip("cost:Q", format=".4f")],
)
a_ref = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(
    strokeDash=[6, 4], color="#444", strokeWidth=1.5).encode(y="y:Q")
a_ref_txt = alt.Chart(pd.DataFrame(
    {"y": [1.0], "pi": [0.45], "t": ["always-on classifier = 1.0"]})).mark_text(
    align="right", dy=-7, fontSize=11, color="#444").encode(x=x, y="y:Q", text="t:N")
a_call = alt.Chart(pd.DataFrame({
    "pi": [0.01], "cost": [KAPPA + routing_rate(0.01, CONFIGS["constrained_cost (recall>=0.95)"])],
    "t": ["π=0.01: 0.131 → ~13% of always-on cost, recall 0.95"]})).mark_text(
    align="left", dx=9, dy=16, fontSize=11.5, fontWeight="bold", color=COLORS[0]).encode(
    x=x, y="cost:Q", text="t:N")

panelA = (a_ref + a_line + a_pts + a_ref_txt + a_call).properties(
    width=W, height=H, title=alt.TitleParams(
        "A. Cascade cost vs always-on classifier",
        subtitle=["C = κ + r(π),  r(π) = π·r_H + (1−π)·r_B,  κ = 1e-3",
                  "dots = the four π values tabulated in the REPORTs"],
        anchor="start", fontSize=14, subtitleFontSize=11, subtitleColor="#555"))

# --- Panel B: deployment precision -----------------------------------------
b_line = alt.Chart(curves).mark_line(strokeWidth=2.5).encode(
    x=x,
    y=alt.Y("precision:Q", scale=alt.Scale(domain=[0, 1], nice=False),
            axis=alt.Axis(title="deployment precision at prevalence π", format=".1f")),
    color=color,
)
b_pts = alt.Chart(points).mark_point(filled=True, size=70, opacity=1).encode(
    x=x, y="precision:Q", color=color,
    tooltip=["config:N", alt.Tooltip("pi:Q", format=".3f"),
             alt.Tooltip("precision:Q", format=".4f")],
)
b_ann = alt.Chart(pd.DataFrame({
    "pi": [0.001, 0.01],
    "precision": [precision(0.001, CONFIGS["constrained_cost (recall>=0.95)"]),
                  precision(0.01, CONFIGS["constrained_cost (recall>=0.95)"])],
    "t": ["0.045", "0.32"]})).mark_text(
    align="left", dx=8, dy=-9, fontSize=11.5, fontWeight="bold", color="#333").encode(
    x=x, y="precision:Q", text="t:N")

panelB = (b_line + b_pts + b_ann).properties(
    width=W, height=H, title=alt.TitleParams(
        "B. The counterpoint: precision collapses at low π",
        subtitle=["precision(π) = πR / (πR + (1−π)·FPR);  both configs FPR = 0.0202",
                  "same FPR ⇒ the two curves nearly coincide"],
        anchor="start", fontSize=14, subtitleFontSize=11, subtitleColor="#555"))

chart = alt.hconcat(panelA, panelB, spacing=42).properties(
    title=alt.TitleParams(
        "Two-stage cascade: cost saving and its precision cost",
        subtitle=[
            "constrained_cost: recall 0.9500, FPR 0.0202, r_H 0.9500, r_B 0.1212, b_B 0.1667   |   "
            "max_f1: recall 0.9812, FPR 0.0202, r_H 0.9812, r_B 0.3636, b_B 0.0556",
            "Curves are computed in closed form from the measured stage rates; those rates are "
            "validation-measured (n=259) and both calibration and config selection happened on that "
            "same validation split (mild selection optimism).",
        ],
        anchor="start", fontSize=17, subtitleFontSize=11, subtitleColor="#555",
        offset=8, subtitlePadding=5),
).configure_view(stroke=None).configure_axis(
    labelFontSize=11, titleFontSize=12, gridColor="#e8e8e8"
).configure_legend(labelFontSize=12, titleFontSize=12)

svg = vlc.vegalite_to_svg(chart.to_json())
with open(OUT_SVG, "w") as f:
    f.write(svg)
print("wrote", OUT_SVG)
print(json.dumps(points.to_dict(orient="records"), indent=1))
