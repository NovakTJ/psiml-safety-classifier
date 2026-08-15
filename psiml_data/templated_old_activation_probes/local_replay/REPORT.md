# Local replay of the 38 successful OpenRouter jailbreaks + activation capture

Date: 2026-08-13. Script: `scripts/redteam/replay_jailbreaks_local.py` (qwen35_env).
Labeling: `scripts/redteam/label_jailbreak_dataset.py` (ccpp_env, Qwen3Guard-Gen-8B,
native response-moderation chat template, greedy).

## Purpose

Train a high-recall **linear probe on the LOCAL Qwen3.5-9B's activations**, so the
38 jailbreaks that succeeded on OpenRouter (`psiml_data/jailbreak_v1/successful_jailbreaks.jsonl`)
were replayed — same harmful prompts, same G0DM0D3 templates (`build_messages`),
same sampling params (temp 1.0, top_p 0.95) — against `/data/models/Qwen3.5-9B`,
capturing residual-stream hidden states.

**Label semantics (exchange classifier):** every row is `exchange_label="harmful"`
because the PROMPT is harmful, regardless of how the local model responded. The
Qwen3Guard fields are metadata for later scenario slicing (prompt-driven vs
response-driven harm, probe-as-response-classifier), NOT training labels.

## Outcome vs OpenRouter

| | OpenRouter | Local replay |
|---|---|---|
| Unsafe (jailbreak succeeded) | 38/38 | **8/38** |
| Controversial | 0 | 4 |
| Safe (refused / deflected) | 0 | 26 |
| Refusal: Yes | 11/38 | 28/38 |

30/38 rows flipped vs the OpenRouter labels. Expected: temp 1.0 sampling makes
divider-style jailbreaks stochastic, and OpenRouter's served weights need not bit-match
the local checkpoint. Local successes: grok-420 5, hermes-fast 3 — same two templates
as every prior round (nothing else has ever worked). For probe training this is fine:
all 38 rows are positive (harmful-prompt) exchanges, and 30 of them are now
adversarial-prompt + refusal/deflection rows — exactly the distribution the deployed
probe sees when the guarded model resists.

## Generation notes / pitfalls hit

- **`max_new_tokens` fidelity:** script fixed 2048 for all rows, but 32/38 source
  rows were generated with 3072 (rounds 2–3). Kept as-is: the cap does not change
  the prompt distribution or early-response content, and probe prefix training
  truncates anyway. Flagged here so nobody diff-compares response tails.
- **eos/pad bug (fixed in script, data verified clean):** Qwen3.5's chat eos is
  `<|im_end|>` (248046) but pad is `<|endoftext|>` (248044), and the model ships
  NO `generation_config.json` — so `generate()` did not stop at `<|im_end|>` and the
  first-pass `finish_reason` detection (post-strip eos check) mislabeled everything
  "length". Scan of all 38 captures: every early stop is `<|im_end|>` + exactly ONE
  trailing token (newline) then `<|endoftext|>` — **zero rows have post-eos content
  contamination**. `finish_reason` was recomputed from the saved token ids
  (27 stop / 11 length) and the script now passes both eos ids to `generate()`.
- Local responses ramble to the token cap more often than OpenRouter's did
  (11/38 length vs 2/6 truncations in round 1) — sampling variance, not a bug.

## Files

- `raw_local.jsonl` — 38 rows: source metadata, local response, generation params
  (per-batch seed = 20260813 + batch_idx; batched sampling shares one RNG, so
  reproduce with the same batching), token counts, `openrouter_guard_*` fields,
  activation-file metadata.
- `labeled_local.jsonl` — same + `guard_raw/guard_safety/guard_categories/
  guard_refusal/is_refusal` from the local Qwen3Guard run. **Use this as the source.**
- `activations/{row_id}.npz` (**gitignored**, 3.7 GB total) — per row:
  - `hidden`: float16 `[8, n_tokens, 4096]` — residual stream at layers
    **3, 7, 11, 15, 19, 23, 27, 31** (the 8 full-attention layers of the hybrid
    32-layer stack, incl. the final layer; `hidden_states[l+1]`).
  - `token_ids`: int32 `[n_tokens]` — span = last **64 prompt tokens** + full
    response (`activations_response_offset` = 64 in every row here), so the
    16-token probe window can cross the prompt/response boundary and prefix
    truncation (Idea 1/2) needs no model re-run.
- Disk: 64 KB/token (8 layers × 4096 × fp16), ~97 MB/row avg, 3.7 GB total.
  Re-capturing more layers later costs ~1–2 s/row (single teacher-forcing pass).

## Next

- Benign-prompt exchanges + activations for the SAFE class (the v1 plan's
  "add benign prompts" step) — replay benign prompts through the same capture.
- Prefix truncation of these rows into token-level training examples.
- Probe layer sweep over the 8 captured layers.
- If more response-driven-harm positives are needed: re-replay with different
  seeds (K>1) — 8/38 per single sample at temp 1.0.
