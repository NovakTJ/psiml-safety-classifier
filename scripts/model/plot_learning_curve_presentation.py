#!/usr/bin/env python
"""Presentation-simplified learning-curve chart -> presentation/learning_curve_v2.png.

A separate script from the teammate's plot_learning_curve_v2.py (not edited,
not overwritten) -- reads the exact same source data and reproduces the same
curve/fit, only with simplified title/legend/axis text for the slide.

2026-08-16: dropped the seed-23 stability-check points and the expanded-data
("2x data") star markers per Lazar's request -- just the seed-42 curve plus
its trend line now. Bigger figure/fonts throughout for readability from the
back of the room.

Source data (unchanged from the original script):
  scripts/model/results/learning_curve_v2/learning_curve_summary.csv
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SUMMARY_CSV = Path("/home/mls01/scripts/model/results/learning_curve_v2/learning_curve_summary.csv")
OUT_PNG = Path("/home/mls01/presentation/learning_curve_v2.png")


def main():
    df = pd.read_csv(SUMMARY_CSV)
    done = df[df["status"] == "completed"]

    fig, ax = plt.subplots(figsize=(13, 6.4), dpi=170)

    c42 = done[done["seed"] == 42].sort_values("n_rows")
    xs = c42["n_rows"].tolist()
    ys = c42["f1"].tolist()
    ax.plot(xs, ys, "o-", color="#2a78d6", lw=3, ms=12, zorder=3, label="F1")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=15, color="#2a78d6")

    lx = [math.log2(x) for x in xs]
    xbar, ybar = sum(lx) / len(lx), sum(ys) / len(ys)
    b = sum((x - xbar) * (y - ybar) for x, y in zip(lx, ys)) / sum((x - xbar) ** 2 for x in lx)
    a = ybar - b * xbar
    x_lo, x_hi = min(xs) * 0.9, max(xs) * 1.3
    fx = [x_lo, x_hi]
    fy = [a + b * math.log2(x) for x in fx]
    ax.plot(fx, fy, "--", color="#2a78d6", alpha=0.45, lw=2, zorder=2,
           label=f"trend: {b:+.3f} F1/doubling")

    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.tick_params(axis="both", labelsize=15)
    ax.set_xlabel("training rows", fontsize=18)
    ax.set_ylabel("validation F1 harmful", fontsize=18)
    ax.set_title("Learning curve — validation F1", fontsize=20)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
