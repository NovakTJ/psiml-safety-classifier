#!/usr/bin/env python
"""Appendix chart 3 — Gemma-3-1B-IT LoRA hyperparameter sweep on data/gemma_v2_no_refusal.

Three panels, one story:
  A  phase-1 grid: validation F1 over learning_rate x rank (alpha = 2*rank, dropout 0.05, seed 42)
  B  phase-2 dropout sweep on the top-3 phase-1 configs (seed 42)
  C  multiseed check of the two best configs (seeds 23/41/42), mean +/- std

Punchline: the phase-1 spread at the top of the grid is smaller than the
seed-to-seed std measured in panel C, so those cells are indistinguishable.

Numbers are transcribed from the REPORT.md tables in
  scripts/model/results/gemma_lora_v2_sweep/
  scripts/model/results/gemma_lora_v2_sweep_phase2/
  scripts/model/results/gemma_lora_v2_multiseed/

Run:  /home/novaktj/aegis_env/bin/python chart3_lora_hyperparam_sweep.py
"""

from pathlib import Path

import altair as alt
import pandas as pd
import vl_convert as vlc

OUT = Path(__file__).with_suffix(".svg")

VAL_N = 259
SEED_STD = 0.0032  # config A F1 std across seeds 23/41/42 (multiseed REPORT)

# ---------------------------------------------------------------- phase 1
# lr, rank, precision, recall, f1  (alpha = 2*rank, dropout = 0.05, seed = 42)
PHASE1 = [
    ("1e-4", 16, 0.9568, 0.9688, 0.9627),
    ("2e-4", 4, 0.9684, 0.9563, 0.9623),
    ("3e-4", 8, 0.9565, 0.9625, 0.9595),
    ("3e-4", 16, 0.9563, 0.9563, 0.9563),
    ("3e-4", 4, 0.9620, 0.9500, 0.9560),
    ("2e-4", 16, 0.9503, 0.9563, 0.9533),
    ("1e-4", 8, 0.9735, 0.9187, 0.9453),
    ("2e-4", 8, 0.9273, 0.9563, 0.9415),
    ("5e-5", 8, 0.9325, 0.9500, 0.9412),
    ("5e-5", 16, 0.9321, 0.9437, 0.9379),
    ("5e-5", 4, 0.9313, 0.9313, 0.9313),
    ("1e-4", 4, 0.9367, 0.9250, 0.9308),
]
p1 = pd.DataFrame(PHASE1, columns=["lr", "rank", "precision", "recall", "f1"])

BEST1 = p1["f1"].max()
p1["tie"] = p1["f1"] >= BEST1 - SEED_STD  # within one seed-sigma of the best cell
p1["label"] = p1["f1"].map(lambda v: f"{v:.4f}")

LR_ORDER = ["5e-5", "1e-4", "2e-4", "3e-4"]
RANK_ORDER = [16, 8, 4]

# ---------------------------------------------------------------- phase 2
# config, dropout, precision, recall, f1  (seed 42; dropout 0.05 rows are phase-1 refs)
PHASE2 = [
    ("lr3e-4 r8 a16", 0.0, 0.9808, 0.9563, 0.9684),
    ("lr3e-4 r8 a16", 0.05, 0.9565, 0.9625, 0.9595),
    ("lr3e-4 r8 a16", 0.1, 0.9682, 0.9500, 0.9590),
    ("lr1e-4 r16 a32", 0.0, 0.9745, 0.9563, 0.9653),
    ("lr1e-4 r16 a32", 0.05, 0.9568, 0.9688, 0.9627),
    ("lr1e-4 r16 a32", 0.1, 0.9568, 0.9688, 0.9627),
    ("lr2e-4 r4 a8", 0.0, 0.9804, 0.9375, 0.9585),
    ("lr2e-4 r4 a8", 0.05, 0.9684, 0.9563, 0.9623),
    ("lr2e-4 r4 a8", 0.1, 0.9563, 0.9563, 0.9563),
]
p2 = pd.DataFrame(PHASE2, columns=["config", "dropout", "precision", "recall", "f1"])
p2["dropout_s"] = p2["dropout"].map(lambda d: f"{d:g}")

# ---------------------------------------------------------------- multiseed
MULTISEED = [
    ("A: lr3e-4 r8 a16 do0.0", 0.9654, 0.0032, 0.9620, 0.9684),
    ("B: lr1e-4 r16 a32 do0.0", 0.9590, 0.0055, 0.9557, 0.9653),
]
p3 = pd.DataFrame(MULTISEED, columns=["config", "mean", "std", "min", "max"])
p3["lo"] = p3["mean"] - p3["std"]
p3["hi"] = p3["mean"] + p3["std"]
p3["label"] = p3.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)

SEEDS = [
    ("A: lr3e-4 r8 a16 do0.0", 23, 0.9657),
    ("A: lr3e-4 r8 a16 do0.0", 41, 0.9620),
    ("A: lr3e-4 r8 a16 do0.0", 42, 0.9684),
    ("B: lr1e-4 r16 a32 do0.0", 23, 0.9560),
    ("B: lr1e-4 r16 a32 do0.0", 41, 0.9557),
    ("B: lr1e-4 r16 a32 do0.0", 42, 0.9653),
]
seeds = pd.DataFrame(SEEDS, columns=["config", "seed", "f1"])

# ---------------------------------------------------------------- panels
BASE_W, BASE_H = 300, 300

heat = (
    alt.Chart(p1)
    .mark_rect(stroke="white", strokeWidth=1)
    .encode(
        x=alt.X("lr:N", sort=LR_ORDER, title="learning rate", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("rank:N", sort=RANK_ORDER, title="LoRA rank r  (alpha = 2r)"),
        color=alt.Color(
            "f1:Q",
            title="val F1",
            scale=alt.Scale(scheme="blues", domain=[0.930, 0.965]),
            legend=alt.Legend(format=".3f", gradientLength=140),
        ),
    )
)
heat_tie = (
    alt.Chart(p1[p1["tie"]])
    .mark_rect(fill=None, stroke="#d62728", strokeWidth=2.5)
    .encode(x=alt.X("lr:N", sort=LR_ORDER), y=alt.Y("rank:N", sort=RANK_ORDER))
)
heat_txt = (
    alt.Chart(p1)
    .mark_text(fontSize=13, fontWeight="bold")
    .encode(
        x=alt.X("lr:N", sort=LR_ORDER),
        y=alt.Y("rank:N", sort=RANK_ORDER),
        text="label:N",
        color=alt.condition(alt.datum.f1 > 0.9545, alt.value("white"), alt.value("#1a1a1a")),
    )
)
panel_a = (
    (heat + heat_tie + heat_txt)
    .properties(
        width=BASE_W,
        height=BASE_H,
        title=alt.TitleParams(
            "A — Phase 1 grid: validation F1",
            subtitle=[
                "dropout 0.05, seed 42, 12/12 cells.",
                "Red outline: within ±1 seed-σ (0.0032) of the best cell —",
                "3 cells statistically indistinguishable.",
            ],
            subtitleFontSize=10,
            anchor="start",
        ),
    )
)

band_b = (
    alt.Chart(pd.DataFrame({"lo": [BEST1 - SEED_STD], "hi": [BEST1 + SEED_STD]}))
    .mark_rect(color="#d62728", opacity=0.10)
    .encode(
        # explicit scale: without it this layer's default zero-based scale wins the merge
        y=alt.Y("lo:Q", title=None, scale=alt.Scale(domain=[0.950, 0.972], nice=False)),
        y2="hi:Q",
    )
)
line_b = (
    alt.Chart(p2)
    .mark_line(point=alt.OverlayMarkDef(size=70, filled=True), strokeWidth=2)
    .encode(
        x=alt.X("dropout_s:N", sort=["0", "0.05", "0.1"], title="LoRA dropout"),
        y=alt.Y(
            "f1:Q",
            title="validation F1",
            scale=alt.Scale(domain=[0.950, 0.972], nice=False),
            axis=alt.Axis(format=".3f"),
        ),
        color=alt.Color(
            "config:N",
            title="config (seed 42)",
            scale=alt.Scale(scheme="dark2"),
            legend=alt.Legend(orient="bottom", columns=1, labelFontSize=10),
        ),
    )
)
panel_b = (
    (band_b + line_b)
    .properties(
        width=BASE_W - 60,
        height=BASE_H,
        title=alt.TitleParams(
            "B — Phase 2: dropout on the top-3 configs",
            subtitle=[
                "dropout 0.0 wins on 2 of 3 configs, but the whole",
                "spread here is only ~4 seed-σ. Shaded band = ±σ",
                "around the best phase-1 cell.",
            ],
            subtitleFontSize=10,
            anchor="start",
        ),
    )
)

err_c = (
    alt.Chart(p3)
    .mark_rule(strokeWidth=2.5, color="#333")
    .encode(x=alt.X("config:N", title=None, axis=alt.Axis(labelAngle=-12, labelFontSize=10)),
            y=alt.Y("lo:Q", title="validation F1 (mean ± std, 3 seeds)",
                    scale=alt.Scale(domain=[0.950, 0.972], nice=False),
                    axis=alt.Axis(format=".3f")),
            y2="hi:Q")
)
caps_c = (
    alt.Chart(p3)
    .mark_tick(thickness=2.5, size=18, color="#333")
    .encode(x="config:N", y="lo:Q")
) + (
    alt.Chart(p3)
    .mark_tick(thickness=2.5, size=18, color="#333")
    .encode(x="config:N", y="hi:Q")
)
pts_c = (
    alt.Chart(seeds)
    .mark_point(size=45, opacity=0.55, color="#777", xOffset=14)
    .encode(x="config:N", y="f1:Q")
)
mean_c = (
    alt.Chart(p3)
    .mark_point(size=140, filled=True, color="#1f4e99")
    .encode(x="config:N", y="mean:Q")
)
txt_c = (
    alt.Chart(p3)
    .mark_text(dy=-22, fontSize=11, fontWeight="bold")
    .encode(x="config:N", y="mean:Q", text="label:N")
)
panel_c = (
    (err_c + caps_c + pts_c + mean_c + txt_c)
    .properties(
        width=BASE_W - 130,
        height=BASE_H,
        title=alt.TitleParams(
            "C — Multiseed (seeds 23/41/42)",
            subtitle=[
                "Error bars overlap: A's lead (+0.0064)",
                "is ~1σ. Grey dots = individual seeds.",
            ],
            subtitleFontSize=10,
            anchor="start",
        ),
    )
)

chart = (
    alt.hconcat(panel_a, panel_b, panel_c, spacing=34)
    .properties(
        title=alt.TitleParams(
            "Gemma-3-1B-IT LoRA hyperparameter sweep — validation set, n = 259 rows (harmful = positive class)",
            subtitle=[
                "alpha = 2 × rank throughout. Phase 1 = full 4×3 lr×rank grid at dropout 0.05, seed 42. "
                "Phase 2 = dropout {0, 0.05, 0.1} on the phase-1 top 3, seed 42.",
                "Phase-1 #1 vs #2 differ by 0.0004 F1 — on 259 rows that is a fraction of one example. "
                "test.jsonl (227 rows) untouched; all numbers here are validation.",
            ],
            subtitleFontSize=11,
            anchor="start",
            fontSize=15,
        )
    )
    .configure_view(strokeWidth=0)
    .configure_axis(labelFontSize=11, titleFontSize=11)
)

OUT.write_text(vlc.vegalite_to_svg(chart.to_json()))
print(f"wrote {OUT}")
