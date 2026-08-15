#!/usr/bin/env python
"""Presentation-simplified learning-curve chart -> presentation/learning_curve_v2.png.

A separate script from the teammate's plot_learning_curve_v2.py (not edited,
not overwritten) -- reads the exact same source data and reproduces the exact
same curve/fit/points, only with simplified title/legend/axis text for the
slide (per Lazar's request: short title like the results chart, shorter
legend labels, x-axis says "training rows" only -- log2 scaling is kept on
the axis itself, just not spelled out in the label -- and y-axis drops the
"(harmful = positive)" qualifier).

Source data (unchanged from the original script):
  scripts/model/results/learning_curve_v2/learning_curve_summary.csv
  scripts/model/results/train_expanded_v3/*/run_summary.json (if present)
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SUMMARY_CSV = Path("/home/mls01/scripts/model/results/learning_curve_v2/learning_curve_summary.csv")
EXPANDED_DIR = Path("/home/mls01/scripts/model/results/train_expanded_v3")
OUT_PNG = Path("/home/mls01/presentation/learning_curve_v2.png")


def main():
    df = pd.read_csv(SUMMARY_CSV)
    done = df[df["status"] == "completed"]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=170)

    c42 = done[done["seed"] == 42].sort_values("n_rows")
    xs = c42["n_rows"].tolist()
    ys = c42["f1"].tolist()
    ax.plot(xs, ys, "o-", color="#2a78d6", lw=2, ms=7, zorder=3, label="F1")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color="#2a78d6")

    lx = [math.log2(x) for x in xs]
    xbar, ybar = sum(lx) / len(lx), sum(ys) / len(ys)
    b = sum((x - xbar) * (y - ybar) for x, y in zip(lx, ys)) / sum((x - xbar) ** 2 for x in lx)
    a = ybar - b * xbar
    x_lo, x_hi = min(xs) * 0.9, max(xs) * 2.6
    fx = [x_lo, x_hi]
    fy = [a + b * math.log2(x) for x in fx]
    ax.plot(fx, fy, "--", color="#2a78d6", alpha=0.45, lw=1.5, zorder=2,
           label=f"trend: {b:+.3f} F1/doubling")

    c23 = done[done["seed"] == 23]
    if len(c23):
        ax.scatter(c23["n_rows"], c23["f1"], marker="s", s=55, color="#4a3aa7",
                  zorder=3, label="seed 23 check")
        for x, y in zip(c23["n_rows"], c23["f1"]):
            off, ha = ((14, 4), "left") if x < 1000 else ((0, -16), "center")
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=off,
                        ha=ha, fontsize=8, color="#4a3aa7")

    exp_rows = []
    for summary_path in sorted(EXPANDED_DIR.glob("*/run_summary.json")):
        s = json.loads(summary_path.read_text())
        exp_rows.append({"seed": int(s["config_id"].rsplit("seed", 1)[1]),
                         "f1": s["best_metrics"]["f1"]})
    if exp_rows:
        exp_df = pd.DataFrame(exp_rows)
        n_expanded = 3985
        ax.scatter([n_expanded] * len(exp_df), exp_df["f1"], marker="*", s=220,
                  color="#1baf7a", zorder=4, label="2× data")
        for y in exp_df["f1"]:
            ax.annotate(f"{y:.4f}", (n_expanded, y), textcoords="offset points",
                        xytext=(10, -3), ha="left", fontsize=8, color="#1baf7a")

    ax.set_xscale("log", base=2)
    ticks = sorted(set(xs) | {3985})
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("training rows")
    ax.set_ylabel("validation F1")
    ax.set_title("Learning curve — validation F1")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
