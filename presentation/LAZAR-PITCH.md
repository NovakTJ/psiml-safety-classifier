# Lazar's spoken pitch — slides 12-14, 17-20

Confirmed-yours sections only (§3, §6, and — as of the ownership swap on 2026-08-15 —
§8 "Red teaming" from `CLAUDE-PLAN.md`; Novak now covers §7 "Inference system" incl.
AEGIS and the ensemble-price finding). Deliberately kept non-technical — no
hyperparameter names, no loss functions, no code. Every number here is already
verified (see `CLAUDE-PLAN.md` "Sources for every number", plus the primary sources
`REDTEAM-FIRST-TRY/{OLD-CLAUDE,CLAUDE}.md` and `psiml_data/gemma_knowledge_probe/REPORT.md`
read directly for this section).

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

## Slide 17 — "Red teaming with a frontier coding agent" (§8, ~20s)

Section-opener, title only on screen + the task-brief image (`REDTEAM-FIRST-TRY/OLD-CLAUDE.md`).
Proposed on-screen bullets (3, matching the "architectures" slide's bare-list style —
apply to `index.html` once you've reviewed them):

On-screen (current, matches `index.html` — **corrected 2026-08-15, was stale**):
1. Kimi K3 — open-weight coding agent
2. Real attack, not a static eval
3. No restrictions, no holding back

> Every classifier needs someone actively trying to break it — Anthropic pays
> professional red teamers for this. We're students, so we used a frontier
> open-weight coding agent instead: Kimi K3. This wasn't another run through our
> test set — it had internet access, could look up known jailbreak techniques, and
> adapt as it went. And we told it explicitly not to hold back — no restrictions,
> don't go easy on us, or we learn nothing. The whole session cost $2.30 in API
> spend.

**~85 words, ~25s.** (the $2.30 figure isn't on-screen anymore, but it's a good
closing detail to say out loud)

---

## Slide 18 — "Example of an attack that worked" (§8, ~45s)

**Superseded 2026-08-15** — the plain-bullet version below was replaced by merging
in Novak's `two_stage_classifier_bypass.svg` diagram instead (conflict resolved in
his favor, see the merge commit). The diagram shows two side-by-side examples, each
passing through three stages, but its three yellow explanation annotations were
deliberately stripped from the image (Lazar's call — "ja cu to sve usmeno da
kazem") so the pitch below carries that content verbally instead of on-screen.

**The three stages, for your own reference (say this first if the audience needs
the frame — same guard model does both classifier stages, just called twice):**
- Input classifier stage — checks the prompt alone, before the chat model runs
  (this is the token-0 pre-check)
- Chat model stage — Qwen3.5-9B actually generates the answer
- Output classifier stage — the same guard re-checks prompt + response together,
  mid-stream

**The exact annotation text that was on the image (say these three, they're the
payload of the slide):**
1. Input stage, Example 2 — "Classifier does not identify the chemical"
2. Chat model stage, Example 2 — "Chat model fooled by academic register"
3. Output stage, Example 2 — "Looks like a normal recipe to classifier — knowledge gap"

> Here's exactly how it gets through — two examples, same three stages. An input
> classifier that checks the prompt, the chat model that answers, and an output
> classifier that checks prompt and response together — the same guard, called
> twice. Example one: a plain "write a recipe for [dangerous chemical]" — blocked
> immediately, the input classifier catches it. Example two: same request, but the
> chemical is a technical description instead of a name. The input classifier
> doesn't recognize the chemical, so it passes. The chat model gets fooled by the
> academic register and gives the full harmful answer. And when the output
> classifier checks that answer, it just looks like a normal recipe — that's not a
> detection failure, it's a knowledge gap.

**~135 words, ~45s.**

---

## Cut: "Fine-tuning traded recall for precision" (was slide 19)

Removed from the deck (2026-08-15) — same "ne treba nam" pattern as the probe-pooling
slide. The finding itself (zero-shot Gemma blocks pipe-bomb/ANFO/nitroglycerin prompts
that the fine-tuned LoRA guard passes — the tuned model over-corrected on
"exam"/"report"/"literature" framing as a signal for *unharmful*, a regression
invisible to the results-table metrics) is still real and still in
`REDTEAM-FIRST-TRY/CLAUDE.md` — say it as an ad-lib during slide 18 or 19 if there's
time, there's no dedicated slide for it anymore.

---

## Slide 19 — "Unguardable at 1B: the gap is knowledge, not capacity" (§8, ~30s)

Table slide (3 model columns: 1B base / 1B+LoRA / 4B, on the same aliases) — this is
the knowledge-probe follow-up, not from `REDTEAM-FIRST-TRY/` directly but from
`psiml_data/gemma_knowledge_probe/REPORT.md` (built to answer a question the
red-team findings raised).

> So why does register beat the guard? We asked the base 1B model, in plain chat, if
> it knows these substances. It doesn't — it calls the ricin precursor "a generally
> low-risk protein" and mustard gas "a flame retardant," confidently and wrong. It
> knows ricin and mustard gas fine by their famous names — it just can't bridge the
> technical alias to the famous name to the harm. We checked: the LoRA adapter
> doesn't touch this, so it's not training damage. And a 4B model fails the exact
> same aliases — so scaling the guard doesn't fix it either. This is structural. To
> block this class of attack at 1B, you'd have to block chemistry as a topic
> entirely — which isn't a usable classifier.

**~130 words, ~40s** (slightly over the 30s guideline — trim the last sentence live if
running long).

---

## Section total (slides 17-19): ~350 words, ≈1:50 — a touch over the plan's
7:15-9:00 (1:45) budget; the diagram-explanation on slide 18 is the easiest trim
if you're running long.

---

## Cut: "The higher-F1 probe was a prompt classifier in disguise"

Removed entirely, not just simplified — you said "ne treba nam" (don't need it), so
this slide and its content are gone from the deck (was slide 15: prompt-pool vs
response-pool F1/switched-recall comparison). If you still want to make this point
verbally somewhere (it's one of the stronger findings — a probe selected on
validation F1 alone was blind to the exact failure mode it needed to catch), it'd
now have to happen as an ad-lib on slide 12 or 13, there's no dedicated slide for it
anymore.

---

## Grand total (all sections, fixed slides only, excluding slide 12's open-ended
ad-lib): ~525 words, ≈2:50 — slides 12-14 (~175 words, ≈1:00) + slides 17-19
(~350 words, ≈1:50). Two separate speaking blocks in the deck, not back-to-back.

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
