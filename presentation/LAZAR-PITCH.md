# Lazar's spoken pitch — slides 12-16

Confirmed-yours sections only (§3 and §6 from `CLAUDE-PLAN.md`). Budget: §3 = 45 s
(~110 words), §6 = 1:45 (~260 words) → **~370 words total, ~2:30**. Deliberately kept
non-technical — no hyperparameter names, no loss functions, no code. Every number here
is already verified (see `CLAUDE-PLAN.md` "Sources for every number").

Not committed yet — review first, then tell me what to adjust.

---

## Slide 12 — "Architectures we tried" (moved: now a text-free recap right
before Results, not the original §3 intro slot)

The "exchange classifier" explanation that used to live on this slide is cut — that
concept is already covered on slide 6 ("Why classify the exchange, not the prompt"),
so repeating it here would be redundant. This slide is now just the five-item list
on screen (no text at all, per your instruction) and a short spoken transition. Regex
is dropped entirely, not mentioned here or anywhere else.

> Here's everything we actually built and compared, in the order we built it: a
> zero-shot Gemma classifier, that same model fine-tuned with LoRA, Qwen3Guard as an
> outside reference point, a linear probe reading Qwen's own activations, and a
> cascade of the probe gating the LoRA classifier.

**~45 words, ~18s.** Much shorter than the old §3 budget (45s) since the concept
explanation moved elsewhere — **that frees ~27s you can spend on Results instead**,
where the four-slide budget (1:45 for slides 13-16) was the tightest part of the
whole pitch. Suggest putting the extra time into slide 13 (the money slide) or
slide 16 (the probe-pooling story) — both currently written right at the word-count
edge of their allotted time.

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

## Slide 14 — "The LoRA config is a real sweep, not one run" (§6, ~15s)

> This config isn't one lucky run — it's the winner of a learning-rate-and-rank sweep,
> then a dropout sweep, then a 3-seed stability check: validation F1 **0.965 ± 0.003**.
> Consistently, less dropout won.

**~35 words, ~14s.**

---

## Slide 15 — "We are data-limited, not architecture-limited" (§6, ~15s)

> And the learning curve is still climbing at full data — about **+0.02 F1 every time
> we double the training set**. We didn't run out of signal to learn from; we ran out
> of data.

**~30 words, ~12s.**

---

## Slide 16 — "The higher-F1 probe was a prompt classifier in disguise" (§6, ~25s)

> One result worth pausing on. Our best probe by validation score only looked at the
> **prompt** — F1 0.98. Give it the 51 cases where the prompt is innocent but the
> response is harmful, and it catches **one**. It was a prompt classifier wearing an
> exchange classifier's numbers. The version that looks at the **response** instead
> scores lower on paper, F1 0.93 — and catches all 51. The validation metric we were
> optimizing couldn't tell those two apart.

**~85 words, ~34s.**

---

## Total: ~375 words, ≈2:30 — matches the §3 (0:45) + §6 (1:45) budget almost exactly.

## ⚠️ Open decision: "six systems" vs "five"

`main_results.png` (already built, already matches `GRAPH-STYLE.md`) shows **5** systems
— we dropped the regex baseline from the chart itself because F1 0.058 flattens the
other bars into a sliver. But the script's "floor" beat needs the regex number to land,
so slide 13's pitch above mentions it **as a spoken number, not a chart bar** ("F1
0.06... not detecting harm, detecting obfuscation"). I changed the slide's `<h1>` from
the plan's "six systems" to "five systems" to match what's actually in the chart —
tell me if you'd rather keep "six" and I'll word the h1 differently (e.g. drop the
count from the title entirely: "Results on the held-out test set").
