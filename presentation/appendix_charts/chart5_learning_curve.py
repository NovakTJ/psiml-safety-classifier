"""Chart 5 (appendix): validation F1 vs training-set size.

Source of numbers: scripts/model/results/learning_curve_v2/REPORT.md
Locked LoRA config: lr=3e-4, r=8, alpha=16, dropout=0.0. Validation = 259 rows.

Run with:  /home/novaktj/aegis_env/bin/python chart5_learning_curve.py
"""

import math
from pathlib import Path

import altair as alt
import pandas as pd
import vl_convert as vlc

OUT = Path(__file__).with_suffix(".svg")

# ---- data (verbatim from REPORT.md results table) -------------------------
MAIN = pd.DataFrame(
    [
        # n_groups, n_rows, F1, precision, recall, best_epoch
        (100, 250, 0.9108, 0.8970, 0.9250, 6),
        (200, 495, 0.9270, 0.9419, 0.9125, 6),
        (400, 987, 0.9527, 0.9618, 0.9437, 6),
        (600, 1477, 0.9518, 0.9801, 0.9250, 3),
        (800, 1985, 0.9684, 0.9808, 0.9563, 6),
    ],
    columns=["n_groups", "n_rows", "f1", "precision", "recall", "best_epoch"],
)
MAIN["seed"] = "seed 42 (main curve)"

SEED23 = pd.DataFrame(
    [
        (200, 495, 0.9350),
        (800, 1985, 0.9657),
    ],
    columns=["n_groups", "n_rows", "f1"],
)
SEED23["seed"] = "seed 23 (repeat run)"

# Multiseed study at full data: +/- 0.0032 F1 seed std.
SEED_STD = 0.0032

# Log-linear fit over the 5 seed-42 points.
FIT_A, FIT_B = 0.7627, 0.0186

XS = [250, 495, 987, 1477, 1985]
FIT = pd.DataFrame({"n_rows": XS})
FIT["f1"] = FIT_A + FIT_B * FIT["n_rows"].map(math.log2)

MAIN["lo"] = MAIN["f1"] - SEED_STD
MAIN["hi"] = MAIN["f1"] + SEED_STD
MAIN["label"] = MAIN["f1"].map("{:.4f}".format)
# hand-placed label offsets so nothing collides at slide size
MAIN["align"] = ["left", "left", "right", "center", "right"]
MAIN["dx"] = [8, 10, -9, 0, -6]
MAIN["dy"] = [-16, 18, -13, 20, -14]

SEED23["label"] = SEED23["f1"].map("{:.4f}".format)
SEED23["align"] = ["left", "right"]
SEED23["dx"] = [9, -12]
SEED23["dy"] = [-10, 20]

BLUE = "#2C5FBF"
ORANGE = "#C2611F"
GRID = "#D8DBE0"
INK = "#22252A"
MUTED = "#6B7078"

X = alt.X(
    "n_rows:Q",
    scale=alt.Scale(type="log", base=2, domain=[220, 2300], nice=False),
    axis=alt.Axis(
        values=XS,
        format="d",
        title="Training rows (log2 scale)  —  100 / 200 / 400 / 600 / 800 label-pure groups",
        grid=True,
        gridColor=GRID,
        labelFontSize=13,
        titleFontSize=13,
    ),
)
YSCALE = alt.Scale(domain=[0.895, 0.985], zero=False, nice=False)
YAXIS = alt.Axis(
    title="Validation F1 (259-row split)",
    format=".3f",
    grid=True,
    gridColor=GRID,
    labelFontSize=13,
    titleFontSize=13,
)
Y = alt.Y("f1:Q", scale=YSCALE, axis=YAXIS)

COLOR = alt.Color(
    "seed:N",
    scale=alt.Scale(
        domain=["seed 42 (main curve)", "seed 23 (repeat run)"],
        range=[BLUE, ORANGE],
    ),
    legend=alt.Legend(title=None, orient="top-left", labelFontSize=12,
                      fillColor="white", padding=8, strokeColor=GRID),
)

# +/- seed-noise band around the main curve.
band = (
    alt.Chart(MAIN)
    .mark_area(color=BLUE, opacity=0.14)
    .encode(X, alt.Y("lo:Q", scale=YSCALE, axis=YAXIS), alt.Y2("hi:Q"))
)

fit_line = (
    alt.Chart(FIT)
    .mark_line(color=MUTED, strokeDash=[7, 5], strokeWidth=2)
    .encode(X, Y)
)

fit_label = (
    alt.Chart(pd.DataFrame({"n_rows": [640], "f1": [0.9155],
                            "t": ["log-linear fit: +0.0186 F1 per doubling of data"]}))
    .mark_text(align="left", color=MUTED, fontSize=12.5, fontStyle="italic")
    .encode(X, Y, alt.Text("t:N"))
)

line = alt.Chart(MAIN).mark_line(strokeWidth=2.5).encode(X, Y, COLOR)
pts = (
    alt.Chart(MAIN)
    .mark_point(filled=True, size=110, stroke="white", strokeWidth=2)
    .encode(X, Y, COLOR, tooltip=["n_groups", "n_rows", "f1", "precision",
                                  "recall", "best_epoch"])
)
def text_layer(df, color):
    """One text mark per row so dx/dy/align can be placed by hand."""
    return alt.layer(
        *[
            alt.Chart(df.iloc[[i]])
            .mark_text(fontSize=12, color=color, align=df.iloc[i]["align"],
                       dx=float(df.iloc[i]["dx"]), dy=float(df.iloc[i]["dy"]))
            .encode(X, Y, alt.Text("label:N"))
            for i in range(len(df))
        ]
    )


pt_labels = text_layer(MAIN, INK)

s23 = (
    alt.Chart(SEED23)
    .mark_point(shape="triangle-up", filled=True, size=150,
                stroke="white", strokeWidth=1.5)
    .encode(X, Y, COLOR, tooltip=["n_groups", "n_rows", "f1"])
)
s23_labels = text_layer(SEED23, ORANGE)

# Flat segment call-out (987 -> 1477 is -0.0009, inside seed noise).
flat = (
    alt.Chart(pd.DataFrame({"n_rows": [1210], "f1": [0.9035],
                            "t": ["987 -> 1477 rows: -0.0009 = flat, inside seed noise"]}))
    .mark_text(align="center", fontSize=11.5, color=MUTED)
    .encode(X, Y, alt.Text("t:N"))
)
flat_rule = (
    alt.Chart(pd.DataFrame({"n_rows": [1210], "f1": [0.9065], "f2": [0.9465]}))
    .mark_rule(color=MUTED, strokeWidth=1, strokeDash=[3, 3])
    .encode(X, alt.Y("f1:Q"), alt.Y2("f2:Q"))
)

chart = (
    alt.layer(band, fit_line, fit_label, flat_rule, flat,
              line, pts, pt_labels, s23, s23_labels)
    .properties(
        width=800,
        height=430,
        title=alt.Title(
            "Still data-limited: validation F1 keeps rising to the full 1985-row train split",
            subtitle=[
                "Locked LoRA config (lr=3e-4, r=8, alpha=16, dropout=0.0) on nested group-stratified subsets; "
                "same 259-row validation split throughout.",
                "Shaded band = +/-0.0032 F1 seed std measured in the full-data multiseed study. "
                "Last segment 1477 -> 1985 is +0.0166, well outside that noise.",
            ],
            fontSize=17,
            subtitleFontSize=12,
            subtitleColor=MUTED,
            anchor="start",
            offset=12,
        ),
    )
    .configure_view(stroke=None)
    .configure_axis(domainColor=GRID, tickColor=GRID, labelColor=INK, titleColor=INK)
)

OUT.write_text(vlc.vegalite_to_svg(chart.to_json()))
print("wrote", OUT)
