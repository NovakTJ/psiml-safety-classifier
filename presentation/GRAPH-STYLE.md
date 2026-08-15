# Graph style guide — PSIML 11 presentation

Style derived from `learning_curve_v2.png` (Novak's original) plus the fixes made while
building `main_results.png`/`main_results_table.png` (Lazar). Follow this so every chart
in the deck looks like it belongs to the same deck.

## Tooling

- **matplotlib only**, `matplotlib.use("Agg")` backend (headless, no display needed on the
  cluster). No seaborn, no plotly — keep the dependency footprint the same as what's
  already used.
- Pure CPU. Never load a model to make a plot — read already-saved metrics/CSVs.
- One script per chart, named `plot_<thing>.py`, living in `scripts/model/`. Output PNG
  saved to the repo root (`/home/mls01/<name>.png`), not inside `results/` (results/ is
  gitignored except `REPORT.md`; the deck's images need to actually be committed).

## Figure setup

```python
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=170)
```
`figsize=(9, 5.5), dpi=170` is the deck default (from `plot_learning_curve_v2.py`). Use it
unless a chart genuinely needs to be wider/taller (e.g. a table with many rows) — then
scale proportionally, don't invent a new aspect ratio per chart.

## Title

Two lines, always:
```python
ax.set_title("<Short claim, Title Case>\n"
             "(<dataset/config detail in parens, lowercase>)")
```
Example: `"Gemma 3 1B LoRA safety classifier — learning curve\n(locked config lr=3e-4 r=8 a=16 dropout=0, 259-row validation, test untouched)"`.
Line 1 is what the chart shows; line 2 is enough provenance that the slide is
self-explanatory without the spoken script (locked config, split, row count, n).

## Colors

**Do not use matplotlib's default tab10 blue+orange pair (`#1f77b4`/`#ff7f0e`) — it was
tried and rejected (looks dated / clashes with the brand deck).** Use the validated
categorical palette from the `dataviz` skill instead
(`references/palette.md`), and always validate a new pair/set before using it:

```
node <dataviz-skill-dir>/scripts/validate_palette.js "<hex,hex,...>" --mode light
```

Fixed slot order (only skip forward, never invent a new hex):

| Slot | Hue | Hex |
|---|---|---|
| 1 | blue | `#2a78d6` |
| 2 | orange | `#eb6834` |
| 3 | aqua | `#1baf7a` |
| 4 | yellow | `#eda100` |
| 5 | magenta | `#e87ba4` |
| 6 | green | `#008300` |
| 7 | violet | `#4a3aa7` |
| 8 | red | `#e34948` |

**2-series default: slot 1 + slot 3 (blue `#2a78d6` + aqua `#1baf7a`)** — this is what
`main_results.png` uses. Validated: CVD ΔE 23.1, normal-vision ΔE 24.0 (both well above
the 8/15 floors). Skips slot 2 (orange) deliberately — nicer-looking pair, still passes.

If a chart ever needs 3+ series, use slots 1-2-3 in order (don't cherry-pick), and
re-run the validator — a pair that passes doesn't guarantee a triple passes.

Single-series charts (e.g. the learning curve) can stay on slot 1 blue alone, as already
shipped.

## Axes

- **Y-axis grid only** (`ax.grid(True, axis="y", alpha=0.3)`, `ax.set_axisbelow(True)`) —
  no vertical gridlines on a bar chart; a line/scatter chart with two dimensions of real
  variation can use `which="both"` like the learning curve does.
- **Truncated y-axis is allowed** when every value in the chart sits in a narrow high
  range (e.g. 0.8-0.98) and a 0-start axis would flatten the real differences into a
  sliver — this is what `main_results.png` does (`ax.set_ylim(0.6, 1.0)`). When you
  truncate, the axis ticks must stay visible and labeled (never hide them) and every bar
  must carry a value label — the truncation must never be the only way a reader could
  misread a value. Don't truncate a chart whose whole point is "these are basically the
  same" — truncation exaggerates differences, only use it when the differences are real
  and the point of the chart.
- Log-scale x-axis (`ax.set_xscale("log", base=2)` style) for anything spanning an order
  of magnitude or more (dataset size, prevalence sweep) — see the learning curve.

## Bars

- Grouped bars: `width=0.35`, one bar pair per category, `zorder=3` (so bars draw above
  the gridlines).
- **Every bar gets a value label**, 3pt above the bar top, `fontsize=8`, centered:
  ```python
  ax.annotate(f"{h:.3f}", xy=(b.get_x()+b.get_width()/2, h), xytext=(0,3),
             textcoords="offset points", ha="center", fontsize=8)
  ```
- Sort categories deliberately, not alphabetically — pick best→worst or the narrative
  order the slide is spoken in, and say which one you picked in the script notes (see
  the open question in `NOVAK-PLAN.md` about slide #17's ordering).

## Lines (learning-curve style)

- `"o-"` marker+line, `lw=2, ms=7, zorder=3`.
- Per-point value labels: `fontsize=8`, colored to match the series, offset above/below
  to avoid overlapping the marker.
- A fitted/reference line (e.g. the log-linear trend) is `"--"`, `alpha=0.45`, `lw=1.5`,
  `zorder=2` — visually behind the real data line.

## Legend

`ax.legend(loc="lower right", fontsize=9)` unless the data itself sits in that corner —
then `"lower left"` or `"upper left"`, whichever is empty. Never `"best"` (matplotlib's
auto-placement can overlap a bar/line unpredictably between reruns).

## Tables (as images)

When a table communicates better than a chart (e.g. a plain metrics comparison), render
it with `ax.table(...)`, not a screenshot of a dataframe:
- Header row: filled with slot-1 blue (`#2a78d6`), white bold text.
- Body rows: plain alternating banding only (`#f2f2f2` / white) — **do not highlight or
  bold a specific row to imply "the winner"** unless the chart's entire point is that
  system and the caveat needed to justify it (e.g. cost, not raw score) is on the same
  slide. A bolded row reads as a ranking claim; don't make one the numbers alone don't
  support.
- `table.scale(1, 2.0)` for readable row height; `fig.subplots_adjust(top=0.78)` (or
  similar) to kill excess whitespace between the title and the table — check the render,
  don't assume `tight_layout()` alone gets it right for a table figure.

## What NOT to do

- Don't use matplotlib's raw default colors (`C0`/`C1`/tab10) unvalidated.
- Don't dual-axis a chart (two y-scales) — split into two charts or index to a common
  base instead (this is the dataviz skill's #1 anti-pattern).
- Don't put a value label on every single point of a dense line/scan (only on the
  points that matter — see the learning curve, which labels its 5-7 real points, not
  every candidate it fit against).
- Don't invent a new figsize/dpi/font per chart — reuse the deck default unless there's
  a concrete reason (e.g. a wide table) and note the reason in the script's docstring.
