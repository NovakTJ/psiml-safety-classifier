"""Appendix chart 1 — linear probe operating curve (threshold sweep).

Numbers are the 7 tabulated rows from the "Recall-oriented threshold analysis"
section of
  scripts/model/results/linear_probe_v2_layer27_lasttoken_sweep/REPORT.md
(validation split, n=259). The underlying threshold_scan.csv is cluster-only, so
the data is inlined here verbatim.

Run:  /home/novaktj/aegis_env/bin/python chart1_probe_operating_curve.py
"""

from pathlib import Path

import altair as alt
import pandas as pd
import vl_convert as vlc

OUT = Path(__file__).with_suffix(".svg")

# threshold, precision, recall, F1, FPR, FN, routing_rate  (validation, n=259)
ROWS = [
    (0.010, 0.738, 0.9875, 0.845, 0.566, 2, 0.826),
    (0.040, 0.804, 0.9750, 0.881, 0.384, 4, 0.749),
    (0.100, 0.842, 0.9625, 0.898, 0.293, 6, 0.707),
    (0.150, 0.874, 0.9500, 0.910, 0.222, 8, 0.672),
    (0.300, 0.906, 0.9060, 0.906, 0.152, 15, 0.618),
    (0.500, 0.947, 0.9000, 0.9231, 0.081, 16, 0.587),
    (0.540, 0.960, 0.8940, 0.9256, 0.061, 17, 0.575),
]
df = pd.DataFrame(
    ROWS,
    columns=["threshold", "precision", "recall", "f1", "fpr", "fn", "routing_rate"],
)

SERIES = {
    "recall": "Harmful recall",
    "fpr": "False-positive rate",
    "routing_rate": "Routing rate (→ guard)",
}
ORDER = list(SERIES.values())
COLORS = ["#0072B2", "#D55E00", "#009E73"]  # CVD-validated categorical trio

long = (
    df.melt(
        id_vars="threshold",
        value_vars=list(SERIES),
        var_name="metric_key",
        value_name="value",
    )
    .assign(metric=lambda d: d.metric_key.map(SERIES))
)

W, H = 900, 470
# 0.540 is omitted as a tick (it would collide with 0.5); the dashed rule marks it.
X_VALUES = [0.01, 0.04, 0.1, 0.15, 0.3, 0.5]

X_SCALE = alt.Scale(type="log", domain=[0.008, 0.7], nice=False)
X_AXIS = alt.Axis(
    title="Decision threshold t on probe probability (log scale)",
    values=X_VALUES,
    format=".3~g",
    grid=True,
    gridColor="#e7e7e4",
    labelFontSize=13,
    titleFontSize=14,
    titlePadding=10,
)
x = alt.X("threshold:Q", scale=X_SCALE, axis=X_AXIS)
y = alt.Y(
    "value:Q",
    scale=alt.Scale(domain=[0, 1.0]),
    axis=alt.Axis(
        title="Rate (fraction of rows, 0–1)",
        format=".0%",
        grid=True,
        gridColor="#e7e7e4",
        labelFontSize=13,
        titleFontSize=14,
        titlePadding=10,
    ),
)
color = alt.Color(
    "metric:N",
    scale=alt.Scale(domain=ORDER, range=COLORS),
    legend=alt.Legend(
        title=None, orient="bottom", direction="horizontal",
        labelFontSize=13, symbolStrokeWidth=3, fillColor="#ffffff",
        padding=8, cornerRadius=3, strokeColor="#ddddda",
    ),
)

# --- knee band: recommended high-recall operating zone -----------------------
knee = alt.Chart(pd.DataFrame({"lo": [0.10], "hi": [0.15]})).mark_rect(
    color="#0072B2", opacity=0.07
).encode(
    # Same scale + axis spec as the other layers, otherwise Vega-Lite's layer
    # axis merge drops the shared x axis (or its title).
    x=alt.X("lo:Q", scale=X_SCALE, axis=X_AXIS),
    x2="hi:Q",
)

knee_label = alt.Chart(
    pd.DataFrame({"threshold": [0.1225], "value": [0.45],
                  "text": ["knee: t ≈ 0.10–0.15"]})
).mark_text(
    align="center", baseline="middle", fontSize=13, fontWeight="bold",
    color="#333333", lineBreak="\n",
).encode(x=x, y=y, text="text:N")

knee_sub = alt.Chart(
    pd.DataFrame({"threshold": [0.1225], "value": [0.39],
                  "text": ["recall 0.950–0.9625 · recommended"]})
).mark_text(
    align="center", baseline="middle", fontSize=11, color="#5a5a56",
).encode(x=x, y=y, text="text:N")

lines = alt.Chart(long).mark_line(strokeWidth=2).encode(x=x, y=y, color=color)
points = alt.Chart(long).mark_point(
    filled=True, size=70, stroke="#fcfcfb", strokeWidth=2
).encode(x=x, y=y, color=color)

# --- F1-optimal marker at t = 0.540 ------------------------------------------
f1_rule = alt.Chart(pd.DataFrame({"threshold": [0.540]})).mark_rule(
    strokeDash=[5, 4], strokeWidth=1.5, color="#77736c"
).encode(x=x)
f1_label = alt.Chart(
    pd.DataFrame({"threshold": [0.505], "value": [0.045],
                  "text": ["F1-optimal t = 0.540"]})
).mark_text(
    align="right", baseline="middle", fontSize=12, color="#4a4744",
).encode(x=x, y=y, text="text:N")

# --- selective direct labels at the low-t end --------------------------------
ends = long[long.threshold == 0.010].assign(
    label=lambda d: d.value.map(lambda v: f"{v:.3f}".rstrip("0"))
)
end_labels = alt.Chart(ends).mark_text(
    align="left", dx=8, dy=-8, fontSize=12, color="#3a3a37"
).encode(x=x, y=y, text="label:N")

chart = (
    # NOTE: keep a full-axis layer first — a layer with axis=None placed first
    # suppresses the shared x axis for the whole layered chart.
    alt.layer(knee_label, knee, knee_sub, f1_rule, f1_label,
              lines, points, end_labels)
    .properties(
        width=W,
        height=H,
        title=alt.TitleParams(
            text="Linear probe operating curve — threshold sweep",
            subtitle=[
                "Layer-27 last-token probe, validation split (n = 259 exchanges). "
                "Lowering t buys harmful recall at the cost of false positives and guard traffic.",
            ],
            fontSize=19,
            subtitleFontSize=13,
            subtitleColor="#5a5a56",
            anchor="start",
            offset=14,
        ),
    )
    .configure_view(stroke=None, fill="#fcfcfb")
    .configure_axis(labelColor="#3a3a37", titleColor="#3a3a37", domainColor="#c9c9c4",
                    tickColor="#c9c9c4")
)

svg = vlc.vegalite_to_svg(chart.to_json())
OUT.write_text(svg)
print(f"wrote {OUT} ({len(svg)} bytes)")
