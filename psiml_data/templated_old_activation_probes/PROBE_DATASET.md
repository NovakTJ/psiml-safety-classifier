# Activation Probe Dataset — three buckets

> **⚠️ DEPRECATED / OLD (renamed 2026-08-14):** this folder was renamed from
> `psiml_data/probes/` to `psiml_data/templated_old_activation_probes/` to avoid confusion
> with the upcoming probe-training dataset. **It was never actually used to train a probe.**
> It is considered a *bad* dataset for the real probe-training goal: the harmful class is
> entirely prompt-driven (no benign-prompt + harmful-response rows), the activations are
> self-generation captures (responses do NOT match the v2 dataset's stored responses — 0/288),
> and 3 rows' `.npz` files were silently overwritten by a second template run. Kept for
> reference/reuse of capture tooling only; do not train the real linear probe on this corpus.
> See `scripts/model/results/linear_probe_preflight/REPORT.md` for the full audit.

> **Known selection caveat (no action needed):** the bucket split was decided partly by
> mistake — for the plain-harmful bucket the agent filtered on `adversarial=False`, which
> happened due to a misunderstanding (it was meant to be "any harmful prompt from
> WildGuardMix"). Impact on the dataset is small, and we can still train the exchange probe
> (the label only cares about `prompt_harmful_label` anyway).
>
> **Missing example type:** there is no **benign prompt + harmful response** example in this
> corpus, and that's not an oversight — it's hard to produce one. Doing so would require either
> building a multi-turn conversation attack, fooling the prompt classifier (hard), or — hardest
> of all — inducing a genuinely harmful response from a truly benign prompt. This branch of the
> OR rule is therefore untrained; worth revisiting when the mid-stream goal needs it.


Source of truth for training the linear probe on **local Qwen3.5-9B residual-stream activations**.
All activations share one capture protocol: fp16 `[8 layers, n_tokens, 4096]` at layers
**3, 7, 11, 15, 19, 23, 27, 31** (the full-attention layers of the hybrid 32-layer stack),
span = last 64 prompt tokens + full locally-generated response. See the two per-folder
`REPORT.md` files for capture details.

**This is NOT the LM-finetuning dataset.** The probe dataset is the combined corpus below;
the LM-finetuning data (`data/complete_dataset.jsonl`, `data/gemma_v2_no_refusal/`) is separate.

## The three buckets (combined = 288 exchanges, 163 harmful / 125 safe)

| # | Bucket | Folder | Rows | `exchange_label` (our rule = prompt_harm OR response_harm) |
|---|---|---|---|---|
| 1 | **Jailbreaks** | `local_replay/` | 38 | harmful (all 38) |
| 2 | **Harmful, non-jailbreaking** | `plain_prompts/` | 125 | harmful (all 125) |
| 3 | **Benign** | `plain_prompts/` | 125 | safe (all 125) |

**Jailbreak bucket (38):** 38 prompts that had succeeded as jailbreaks on OpenRouter
(`psiml_data/jailbreak_v1/successful_jailbreaks.jsonl`), replayed against local `/data/models/Qwen3.5-9B`
with their G0DM0D3 templates (GODMODE FAST ×12, GROK 4.20 ×26; the only two templates that ever worked),
temp 1.0, top_p 0.95. Because sampling is stochastic, most did not re-succeed locally:
Qwen3Guard judged the local response **Unsafe 8 / Controversial 4 / Safe 26**. The correct
"local successes" number is **8/38** (8 rows with `guard_safety="Unsafe"` in
`labeled_local.jsonl`). All 38 are still positive exchanges (harmful prompt), and the judge
fields are metadata for slicing, not labels.

**Harmful non-jailbreaking bucket (125):** plain (no-template) harmful prompts, selected
from `complete_dataset.jsonl` with `adversarial=False` — i.e. direct harmful requests with no
jailbreak wrapper. The local model refuses/deflects nearly all of these, so they are the
"guarded model resists" distribution the deployed probe sees: the exchange is **harmful
because the prompt is harmful**, even though the response is usually a refusal.

**Response-label caveat for bucket 2:** we believe these very likely did NOT jailbreak even a
single time (plain direct harmful requests, consistent with the 0/30 OpenRouter probe), but we
**did not verify that** — there is **no response label** here (`plain_prompts/` has no
Qwen3Guard judgment; guard fields are null). This is fine for training the exchange classifier
(the exchange is harmful because the prompt is harmful, regardless of the response), but if we
ever pivot to a **response-only** classifier we will need to re-label these responses with
Qwen3Guard (or similar) — see "How to use for training".

**Benign bucket (125):** plain unharmful prompts, `exchange_label="safe"` (unharmful prompt
AND unharmful response). Mix of `adversarial` True/False (73/52) — this does not matter, see
below.

## What the labels actually mean (read this before training)

- `exchange_label` in the `.jsonl` files **equals our own label machinery** exactly:
  `prompt_harm_label == "harmful"` **OR** `response_harm_label == "harmful"`.
  Verified 0/288 mismatches against `data/complete_dataset.jsonl`. This is the binary target:
  **"should the guard fire here" = harmful, "should not fire" = safe.**
- For the three buckets this means: positive class = ALL harmful prompts (bucket 1 + 2,
  including refusal responses); negative class = unharmful prompt + unharmful response.
- **Do not use the `adversarial` column to describe the buckets.** It is orthogonal to our
  label (it marks whether the *source wildguard* prompt came from the WildJailbreak adversarial
  pool, independent of harm). The benign bucket is majority `adversarial=True` (73/52) and the
  jailbreak bucket is mostly `adversarial=False` (33/5). The 3-way split is derivable instead
  from **our label + template presence**: jailbreaks have `template_id`/`template_codename` set,
  the two plain buckets have `template_id=None` and differ only by `exchange_label`.

## How to use for training

Recommended train/val split: **by bucket / by row**, stratified so both classes and all three
buckets appear in train and val (e.g. 80/20 row split, ensuring positive=163 and negative=125
both represented; positive is over-represented, 57% vs 43%). Do not leak template or language
info across the split.

Suggested recipes, in order of increasing fidelity:

1. **Exchange-level probe** (simplest): average/max-pool hidden states over the captured span
   per row → one probe score per exchange. Labels = `exchange_label == "harmful"`.
   Note: because the harmful class is dominated by harmful-prompt rows, this effectively learns
   "prompt is harmful" — fine for a deployment guard, but it is NOT yet response-stream harm
   detection.
2. **Token-level probe** (for the mid-stream stopping goal): slice activations per token
   (16-token sliding window), label from the exchanged label (all tokens of a harmful exchange
   inherit `harmful`). This is where prefix truncation (Idea 1/2) becomes relevant: tokens before
   the harmful content appear are noisy positives — the known, accepted tradeoff. The 8 local
   jailbreak successes (bucket 1, `guard_safety=Unsafe`) are the cleanest source of
   response-driven harm for this.

For all of the above: **no Qwen3Guard judgment of the local responses exists in
`plain_prompts/`** (guard fields are null — bucket 2/3). The binary target does not need it,
but if you want to slice "prompt-driven vs response-driven harm" or add a response-harm axis to
the probe, re-run `scripts/redteam/label_jailbreak_dataset.py` over `plain_prompts/` to backfill
`guard_*` fields. **If we pivot to a response-only classifier, this re-label is mandatory for
bucket 2** (and bucket 3): the current `exchange_label` is prompt-driven, so a response-only
target cannot be derived from it without a judge pass.

Metrics to report (per the project plan): unsafe recall, benign false-positive rate, precision,
F1, and — for routing — the probe score threshold.

## Files

- `local_replay/labeled_local.jsonl` — jailbreak bucket (38), with local Qwen3Guard judge fields. **Source for bucket 1.**
- `plain_prompts/raw_plain.jsonl` — plain harmful (125) + benign (125), guard fields null. **Source for buckets 2 & 3.**
- `local_replay/activations/{row_id}.npz`, `plain_prompts/activations/{row_id}.npz` — the activations (**gitignored**, 3.7 + 11 GB).
- `local_replay/REPORT.md`, `plain_prompts/REPORT.md` — capture methodology and pitfalls.
