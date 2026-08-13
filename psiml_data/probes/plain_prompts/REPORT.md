# Plain-prompt activation capture (harmful-refusals + benign) for probe training

Date: 2026-08-13. Script: `scripts/redteam/capture_plain_activations.py` (qwen35_env).
Complement to `psiml_data/probes/local_replay/` (jailbreak captures) — same npz
schema, same 8 layers (3,7,11,15,19,23,27,31), same span (last 64 prompt tokens +
full response), same sampling params (temp 1.0, top_p 0.95, max_new_tokens 2048,
`enable_thinking=False`, per-batch seeds = 20260814 + batch_idx).

## What

250 plain (NO jailbreak template) prompts from `data/complete_dataset.jsonl`,
single user turn, generated + captured on local `/data/models/Qwen3.5-9B`:

- **125 harmful prompts, `adversarial=False` only** → near-certain refusals
  (Aug-12 OpenRouter probe: 0/30 non-template adversarial prompts succeeded).
  Exchange label still **harmful** (harmful prompt ⇒ harmful exchange) — this is
  the deployment-time "guarded model resists" distribution for the probe.
- **125 benign prompts** (both adversarial flags, matching pool mix: 73 T / 52 F)
  → exchange label **safe**.

Both classes stratified ~50/50 EN/non-EN across 31 languages (EN proportional to
pool share, non-EN round-robin; pure round-robin would have shrunk EN to ~4%).
Row-ids used in jailbreak rounds 1–3 excluded (already captured with templates).
Selection seed 42.

**Selection pitfall (corrected during review):** row_id prefix
(`orig`/`trans`/`weird`) is the augmentation TYPE, not adversarialness —
`trans-*` = translation (~half adversarial=False), even `weird-*` has
adversarial=False rows, and benign prompts are ~half non-EN. Use the explicit
`adversarial` boolean, not the prefix.

## Result

- 250/250 captured, 0 npz shape mismatches (verified against token-count metadata).
- finish_reason: 240 stop / 10 length (the 10 length rows incl. benign long
  answers and a few harmful prompts the model engaged with at length — still
  correctly harmful-labeled at the exchange level).
- Harmful responses: median 209 tokens (refusal + explanation); spot-checks show
  real refusals incl. non-EN (es/ur/hi...). English refusal-regex only matches
  50/125 — expected regex language bias, NOT non-refusals; no Qwen3Guard run
  (per user instruction), guard fields are null in the jsonl.
- Benign responses: median 801 tokens, on-topic helpful answers.
- Disk: 11 GB activations (gitignored), ~44 MB/row avg.

## Files

- `raw_plain.jsonl` — 250 rows, source of truth: prompt/response, token counts,
  `exchange_label`, `adversarial`, language/subcategory, null guard fields,
  activation-file metadata. Schema matches `local_replay/raw_local.jsonl`
  (plus `adversarial`, minus `openrouter_*`; `template_id` = null).
- `activations/{row_id}.npz` — `hidden` fp16 [8, n_tokens, 4096], `token_ids` int32.
- `run.log`.

## Combined probe corpus so far

| source | rows | exchange labels |
|---|---|---|
| `local_replay/` (jailbreaks, temp 1.0 replay of OR successes) | 38 | all harmful |
| `plain_prompts/` (this run) | 250 | 125 harmful / 125 safe |

Still missing for high-recall probe training: more safe-adversarial coverage is
now decent (73 benign adversarial rows here), but harmful exchanges are 163 vs
125 safe — consider a second plain-prompt pass for more safe rows, or weight the
loss. Prefix truncation of responses into token-level examples comes next.
