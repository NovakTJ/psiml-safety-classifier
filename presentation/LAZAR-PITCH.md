# Lazar's spoken pitch — slides 12-15

Confirmed-yours sections only (§3 and §6 from `CLAUDE-PLAN.md`). Deliberately kept
non-technical — no hyperparameter names, no loss functions, no code. Every number here
is already verified (see `CLAUDE-PLAN.md` "Sources for every number").

Not committed for review purposes — content is now finalized in `index.html`, update
this file if the spoken text changes further.

---

## Slide 12 — "Architectures we tried" (text-free list; you're planning to talk more
here — this section only covers the on-screen list, not your extra ad-libbed content)

Five items, in build order, Qwen3Guard kept at the end as the external SOTA
reference point (not something we built, but the benchmark to compare against):

1. Gemma-3-1B zero-shot
2. Gemma-3-1B + LoRA
3. Linear probe
4. Ensemble — probe + LoRA
5. Qwen3Guard-8B — SOTA reference

> Here's everything we actually built and compared, in the order we built it: a
> zero-shot Gemma classifier, that same model fine-tuned with LoRA, a linear probe
> reading Qwen's own activations, an ensemble of the probe and the LoRA classifier
> together, and finally Qwen3Guard — an 8-billion-parameter production guard — as
> the SOTA reference point we're measuring ourselves against.

**~35 words, ~14s** as a bare minimum transition — you said you'll expand on this live,
so treat this as the floor, not the target.

---

## Slide 13 — "Results: five systems on the held-out test set" (§6, ~50s of the 1:45)

⚠️ **Title change from the plan's "six systems" — see the note below the pitch.**

> Everything here is on our held-out test set — 227 rows, touched exactly once, after
> every configuration was already locked in.
>
> A quick floor to set the scale: a keyword baseline with no learning gets F1 0.06 —
> it's not detecting harm, it's detecting how obfuscated text happens to look. Zero-shot
> Gemma does much better, F1 0.81, but flags 44% of benign exchanges — unusable in
> production. Fine-tuning that same model with LoRA takes it to F1 0.92 and drops false
> alarms from 44% down to 9%. That's the headline: fine-tuning buys you precision, not
> recall. Our probe-gated cascade reaches F1 0.94, matching Qwen3Guard's recall at a
> fraction of its size and cost.

**~110 words, ~44s.**

---

## Slide 14 — "We are data-limited, not architecture-limited" (§6, ~15s)

> And the learning curve is still climbing at full data — about **+0.02 F1 every time
> we double the training set**. We didn't run out of signal to learn from; we ran out
> of data.

**~30 words, ~12s.** Note: the chart's x-axis just says "training rows" now (no
"log2 scale" label) — the pitch text above already covers it verbally ("every time
we double the training set"), so nothing extra needed, but don't drop that line.

---

## Slide 15 — "The higher-F1 probe was a prompt classifier in disguise" (§6, ~25s)

> One result worth pausing on. Our best probe by validation score only looked at the
> **prompt** — F1 0.98. Give it the 51 cases where the prompt is innocent but the
> response is harmful, and it catches **one**. It was a prompt classifier wearing an
> exchange classifier's numbers. The version that looks at the **response** instead
> scores lower on paper, F1 0.93 — and catches all 51. The validation metric we were
> optimizing couldn't tell those two apart.

**~85 words, ~34s.**

---

## Total (fixed slides only, excluding slide 12's open-ended ad-lib): ~260 words, ≈1:30.

The "LoRA config is a real sweep, not one run" content (12-config grid, dropout
finding, 3-seed stability check 0.965 ± 0.003) had its own slide, tried as both a
19-point chart and a grouped/generalized chart — neither worked visually, so the
slide was cut. Fold whatever of that you still want to say into slide 12's ad-lib
time instead of a dedicated slide.

## ⚠️ Open decision: "six systems" vs "five"

`main_results.png` shows **5** systems — the regex baseline was dropped from the chart
itself because F1 0.058 flattens the other bars into a sliver. The script's "floor"
beat still needs the regex number, so slide 13's pitch mentions it **as a spoken
number, not a chart bar**. The slide's `<h1>` was changed from the plan's "six
systems" to "five systems" to match what's actually shown — tell me if you'd rather
keep "six" and word the h1 differently instead (e.g. drop the count: "Results on the
held-out test set").
