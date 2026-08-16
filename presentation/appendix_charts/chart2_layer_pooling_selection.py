#!/usr/bin/env python
"""Appendix chart 2 -- "validation F1 silently selected a prompt-only probe".

Panel A: val F1 for the layer x pooling combos that are ACTUALLY REPORTED.
         The full 8x4 = 32-combo sweep CSV lives on the cluster only; the local
         REPORT.md tabulates just the top 10. The missing 22 combos are shown as
         an explicit "not reported locally" note -- no values are invented.
Panel B: val F1 (x) vs switched-prompt recall (y). The punchline: the val-F1
         winner (layer15_mean_prompt) catches 1/51 benign-prompt+harmful-response
         rows; the response-aware pick (layer15_mean_response) catches 51/51.

Sources (all read verbatim, nothing extrapolated):
  scripts/model/results/linear_probe_v3_multilayer_pooling_sweep/REPORT.md
  scripts/model/results/linear_probe_v3_multilayer_pooling_sweep/switched_prompt_check/REPORT.md
  scripts/model/results/linear_probe_v3_multilayer_pooling_sweep_response_aware/REPORT.md

Run:  /home/novaktj/aegis_env/bin/python chart2_layer_pooling_selection.py
"""

import os

import altair as alt
import pandas as pd
import vl_convert as vlc

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "chart2_layer_pooling_selection.svg")

C_PROMPT = "#D97706"    # prompt-only pooling (ignores the response)
C_RESPONSE = "#3B6FD4"  # response-aware pooling
C_OTHER = "#7C5CBF"     # last / mean_last16 (partial response coverage)
INK = "#1F2328"
MUTED = "#5C6470"

# ---------------------------------------------------------------- Panel A data
# Phase 1, fixed config C=0.03 cw=balanced. Top 10 of 32 combos -- the only rows
# tabulated in the local report.
PHASE1 = [
    # layer, pooling, val_f1
    (15, "mean_prompt",   0.9718),
    (19, "mean_prompt",   0.9430),
    (31, "mean_prompt",   0.9412),
    (23, "mean_prompt",   0.9393),
    (11, "last",          0.9359),
    (11, "mean_prompt",   0.9346),
    (11, "mean_response", 0.9320),
    (15, "mean_last16",   0.9274),
    (15, "mean_response", 0.9251),
    (19, "mean_response", 0.9251),
]

FAMILY = {
    "mean_prompt": "prompt-only (ignores response)",
    "mean_response": "response-aware",
    "last": "partial (last token / last 16)",
    "mean_last16": "partial (last token / last 16)",
}
FAM_ORDER = ["prompt-only (ignores response)", "response-aware",
             "partial (last token / last 16)"]
FAM_RANGE = [C_PROMPT, C_RESPONSE, C_OTHER]
# explicit descending-F1 order (layered charts ignore sort="-x")
SORT_A = [f"layer{l}_{p}" for l, p, _f in PHASE1]

dfA = pd.DataFrame(
    [{"layer": l, "pooling": p, "val_f1": f,
      "family": FAMILY[p], "label": f"layer{l}_{p}"} for l, p, f in PHASE1]
)

# ---------------------------------------------------------------- Panel B data
# (a) response-aware sweep top-10: val_f1 AND switched_recall for each config.
RESP_AWARE = [
    ("layer15_mean_response__C0.003_cw_harmful_x2", 15, 0.9329, 1.0000),
    ("layer27_mean_response__C0.003_cw_harmful_x2", 27, 0.9085, 1.0000),
    ("layer19_mean_response__C0.003_cw_harmful_x2", 19, 0.9321, 0.9608),
    ("layer23_mean_response__C0.003_cw_harmful_x2", 23, 0.9146, 0.9608),
    ("layer15_mean_response__C0.01_cw_harmful_x2",  15, 0.9317, 0.9412),
    ("layer31_mean_response__C0.003_cw_harmful_x2", 31, 0.8875, 0.9804),
    ("layer31_mean_response__C0.01_cw_harmful_x2",  31, 0.8931, 0.9608),
    ("layer19_mean_response__C0.01_cw_harmful_x2",  19, 0.9317, 0.9216),
    ("layer15_mean_response__C0.003_cw_none",       15, 0.9304, 0.9216),
    ("layer27_mean_response__C0.01_cw_harmful_x2",  27, 0.9108, 0.9412),
]

rowsB = [{"config": c, "val_f1": f, "switched_recall": r,
          "family": "response-aware", "hits": f"{round(r * 51)}/51",
          "note": ""} for c, _l, f, r in RESP_AWARE]

# (b) the three probes from the switched-prompt stress test. val F1 for
#     layer15_mean_prompt is its phase-2 winning config (0.9779); layer27_last is
#     the attempt-2 locked probe (0.9231). layer11_mean_response's switched recall
#     (0.8431) comes from that same file -- its val F1 under the fixed phase-1
#     config is 0.9320.
rowsB += [
    {"config": "layer15_mean_prompt  (val-F1 winner)", "val_f1": 0.9779,
     "switched_recall": 0.0196, "family": "prompt-only (ignores response)",
     "hits": "1/51", "note": "picked by val F1"},
    {"config": "layer27_last  (v2 locked probe)", "val_f1": 0.9231,
     "switched_recall": 0.4314, "family": "partial (last token / last 16)",
     "hits": "22/51", "note": ""},
    {"config": "layer11_mean_response  (stress-test run)", "val_f1": 0.9320,
     "switched_recall": 0.8431, "family": "response-aware", "hits": "43/51",
     "note": ""},
]
dfB = pd.DataFrame(rowsB)

ANNOT = pd.DataFrame([
    {"val_f1": 0.9779, "switched_recall": 0.0196,
     "text": "layer15_mean_prompt - best val F1 (0.978), catches 1/51",
     "dx": -18, "dy": -12, "align": "right"},
    {"val_f1": 0.9329, "switched_recall": 1.0000,
     "text": "layer15_mean_response - val F1 0.933, catches 51/51",
     "dx": 6, "dy": -20, "align": "right"},
])
HILITE = ANNOT[["val_f1", "switched_recall"]]

# ---------------------------------------------------------------------- Panel A
baseA = alt.Chart(dfA)
barsA = baseA.mark_bar(cornerRadiusEnd=4, height=15).encode(
    x=alt.X("val_f1:Q",
            scale=alt.Scale(domain=[0.90, 1.0], clamp=True),
            axis=alt.Axis(title="validation F1 (val n=259)", format=".2f",
                          grid=True, gridColor="#E6E8EB", tickCount=6)),
    y=alt.Y("label:N", sort=SORT_A, title=None,
            axis=alt.Axis(labelFontSize=11, labelColor=INK, domain=False,
                          ticks=False)),
    color=alt.Color("family:N",
                    scale=alt.Scale(domain=FAM_ORDER, range=FAM_RANGE),
                    legend=alt.Legend(title="pooling looks at",
                                      orient="bottom", direction="horizontal",
                                      columns=3, labelFontSize=11,
                                      titleFontSize=11, symbolType="square")),
)
labA = baseA.mark_text(align="left", dx=5, fontSize=10, color=MUTED).encode(
    x=alt.X("val_f1:Q", scale=alt.Scale(domain=[0.90, 1.0], clamp=True)),
    y=alt.Y("label:N", sort=SORT_A),
    text=alt.Text("val_f1:Q", format=".4f"),
)
panelA = (barsA + labA).properties(
    width=350, height=250,
    title=alt.TitleParams(
        "A. Layer x pooling screen - the 10 combos the report tabulates",
        subtitle=["Phase 1, fixed C=0.03 / class_weight=balanced.",
                  "The sweep ran 8 layers x 4 poolings = 32 combos; only the top 10 are",
                  "reported locally (full CSV is cluster-only), so 22 combos are not shown.",
                  "Top of the ranking is all prompt-only pooling."],
        anchor="start", fontSize=13, subtitleFontSize=10, subtitleColor=MUTED),
)

# Explicit marker for the 22 combos whose values are not available on this machine.
missing = alt.Chart(pd.DataFrame([{"t": ""}])).mark_text(
    align="left", x=0, y=8, fontSize=10.5, fontStyle="italic", color=MUTED,
    text="+ 22 further layer x pooling combos: NOT REPORTED (values unavailable locally)",
).properties(width=350, height=18)
panelA = alt.vconcat(panelA, missing, spacing=2)

# ---------------------------------------------------------------------- Panel B
baseB = alt.Chart(dfB)
xB = alt.X("val_f1:Q", scale=alt.Scale(domain=[0.875, 0.995]),
           axis=alt.Axis(title="validation F1 (val n=259)", format=".2f",
                         grid=True, gridColor="#E6E8EB", tickCount=5))
yB = alt.Y("switched_recall:Q", scale=alt.Scale(domain=[-0.05, 1.12]),
           axis=alt.Axis(title="switched-prompt recall (n=51)", format=".0%",
                         grid=True, gridColor="#E6E8EB", values=[0, .25, .5, .75, 1]))

ring = baseB.mark_point(filled=True, size=150, opacity=1,
                        color="#FFFFFF").encode(x=xB, y=yB)
dots = baseB.mark_point(filled=True, size=95, opacity=0.95).encode(
    x=xB, y=yB,
    color=alt.Color("family:N",
                    scale=alt.Scale(domain=FAM_ORDER, range=FAM_RANGE),
                    legend=None),
    shape=alt.Shape("family:N", scale=alt.Scale(
        domain=FAM_ORDER, range=["triangle-down", "circle", "square"]),
        legend=None),
    tooltip=["config:N", "val_f1:Q", "switched_recall:Q", "hits:N"],
)
hl = alt.Chart(HILITE).mark_point(size=340, shape="circle", filled=False,
                                  stroke=INK, strokeWidth=1.5,
                                  strokeDash=[3, 2]).encode(x=xB, y=yB)
ann = alt.layer(*[
    alt.Chart(ANNOT.iloc[[i]]).mark_text(
        fontSize=11, fontWeight="bold", color=INK,
        dx=int(ANNOT.iloc[i]["dx"]), dy=int(ANNOT.iloc[i]["dy"]),
        align=ANNOT.iloc[i]["align"],
    ).encode(x=xB, y=yB, text="text:N")
    for i in range(len(ANNOT))
])
arrow = alt.Chart(pd.DataFrame([{"x": 0.9740, "y": 0.055,
                                 "x2": 0.9370, "y2": 0.945}])).mark_line(
    color=MUTED, strokeWidth=1.5, strokeDash=[4, 3]).encode(
    x=alt.X("x:Q", scale=alt.Scale(domain=[0.875, 0.995])),
    y=alt.Y("y:Q", scale=alt.Scale(domain=[-0.05, 1.12])),
    x2="x2:Q", y2="y2:Q")

panelB = (arrow + ring + dots + hl + ann).properties(
    width=430, height=250,
    title=alt.TitleParams(
        "B. Validation F1 does not buy response detection",
        subtitle=["Each point is a probe config. Switched-prompt set = benign prompt +",
                  "harmful response (n=51, all harmful): pure recall, no prompt signal.",
                  "Down-triangle = prompt-only pooling, circle = response-aware,",
                  "square = last-token / last-16. Higher and righter is better."],
        anchor="start", fontSize=13, subtitleFontSize=10, subtitleColor=MUTED),
)

chart = (
    alt.hconcat(panelA, panelB, spacing=34)
    .properties(title=alt.TitleParams(
        "Selecting a linear probe on validation F1 picks a prompt-only classifier in disguise",
        subtitle=["Qwen3.5-9B activations, logistic probe. Sources: linear_probe_v3_multilayer_pooling_sweep "
                  "(+ switched_prompt_check) and ..._response_aware REPORT.md."],
        anchor="start", fontSize=16, subtitleFontSize=10.5, subtitleColor=MUTED,
        offset=6))
    .configure_view(stroke=None)
    .configure_axis(labelColor=MUTED, titleColor=MUTED, titleFontSize=11,
                    labelFontSize=10, domainColor="#C9CDD3", tickColor="#C9CDD3")
    .configure_title(color=INK)
    .configure_legend(labelColor=MUTED, titleColor=MUTED)
)

if __name__ == "__main__":
    svg = vlc.vegalite_to_svg(chart.to_json())
    with open(OUT, "w") as fh:
        fh.write(svg)
    print("wrote", OUT)
