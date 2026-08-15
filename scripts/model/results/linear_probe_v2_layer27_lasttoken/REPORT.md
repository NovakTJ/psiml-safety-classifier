# Linear probe v2 — attempt 1: last-token, layer 27, teacher-forced

**Date**: 2026-08-14 · **Scripts**: `scripts/model/probe_v2_common.py`,
`train_probe_v2.py`, `eval_probe_v2.py` (qwen35_env, transformers 5.15.0) ·
**Status**: DONE for attempt 1 (train + validation). **`test.jsonl` activations
NOT yet captured, test NOT evaluated** — one flag away
(`eval_probe_v2.py --split test`), deliberately deferred until probe
hyperparameters are locked (attempt-1 hyperparameters are guessed).

## Locked decisions for attempt 1

- **Task**: reproduce the v2 exchange label (`final_label`, harmful=1) from
  Qwen3.5-9B's residual stream — the same (prompt, response) exchanges Gemma /
  Qwen3Guard were evaluated on. Teacher-forced: prefill only, **no decoding**
  (the response text comes from the dataset, never generated).
- **Input format** (per `linear_probe_preflight` REPORT §7): chat template,
  `add_generation_prompt=True, enable_thinking=False`; response rows get
  response tokens + `<|im_end|>` appended (Format A, separate tokenization +
  concatenate); empty-response rows are the bare prompt prefix (Format B
  option 1 — identical prefix boundary).
- **Feature**: activation at the **last token** (`<|im_end|>` for response
  rows, think-block-closing newline for empty-response rows), **decoder layer
  27 = second-to-last full-attention layer** (full-attention layers
  3,7,11,15,19,23,27,31 — verified at runtime from `config.layer_types`;
  layer 27 is a raw pre-norm residual value, deliberately avoiding the
  post-final-RMSNorm layer 31 scale mismatch flagged in the preflight).
  4096-dim, stored fp16.
- **Probe** (guessed, explicitly not tuned): `StandardScaler →
  LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, lbfgs,
  seed 42)`.

## Kill-safety / resume design (as requested)

- Per-row activations written to `activations/{split}/{row_id}.npy` via
  temp-file + `os.replace` (atomic) immediately after each batch; one
  fsync'd line per row in `progress_{split}.jsonl`; heartbeat per batch in
  `run.log` (rows done, tok/s, ETA).
- Re-running either script skips rows whose `.npy` already exists → a kill
  loses at most the in-flight batch. **Proven live**: the first eval run
  OOM'd after 240/259 validation rows; the re-run did only the remaining 19.
- Activations are kept precisely so probe hyperparameters can be tweaked
  later without re-extraction (extraction is the expensive part; fitting is
  14 s).

## Runtime (estimate vs actual)

Pre-code estimate: ~10–20 min total (model load ~1–2 min + prefill 2–11 min
at 2k–10k tok/s for 1.28M tokens + seconds to fit).

Actual: **~6,100 tok/s sustained** (bf16, A100, batches of ≤16 rows /
≤24,576 padded tokens, sorted by length). Train split: 1,085,866 tokens in
**178 s**; full train run (load → capture → fit → save) **~3.5 min**.
Validation: 259 rows in ~1 min once resumed. Fit: 14 s (140 lbfgs iters).

## Results (attempt 1, threshold 0.5)

| split | n | P | R (harmful) | F1 | FPR | acc | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---:|---:|---|
| train (sanity only) | 1985 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1198/0/0/787 |
| **validation** | 259 | **0.940** | **0.881** | **0.910** | 0.091 | 0.892 | 141/9/19/90 |

Validation context (same 259-row split, harmful=positive): Gemma zero-shot
F1 0.838 · Qwen3Guard native F1 0.926 · Gemma locked LoRA F1 0.968
(validation number from the sweep). **The untuned single-layer last-token
probe (F1 0.910) already beats Gemma zero-shot and sits just below the
purpose-built 8B guard** — a strong signal that layer-27 residual features
carry the exchange-harm information. Train F1 = 1.0 is expected for a
4096-dim linear model on 1985 rows and means nothing by itself.

## Bugs hit during the run (both fixed in `probe_v2_common.py`)

1. `np.save(path_without_.npy_suffix)` silently appends `.npy` → the atomic
   temp-write target didn't exist for `os.replace`. Fixed by saving into an
   open file handle.
2. Eval OOM'd on a large validation batch (32,768-token budget + a foreign
   ~4 GiB orphan process on the GPU + fragmentation). Fixed with
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and budget
   32,768 → 24,576 (transient hidden-states 8.9 → 6.6 GB). The orphan
   process (`nvidia-smi` shows 4 GiB with no PID table entry — the known
   PID-namespace artifact) was left alone per the team-confirmation rule.

## Files

```
activations/{train,validation}/{row_id}.npy   4096-dim fp16 last-token layer-27 vectors
progress_{train,validation}.jsonl             per-row fsync'd capture log
run.log                                       heartbeat log (batch, tok/s, ETA)
probe.joblib / probe_config.json              trained probe + locked config
metrics_train.json / metrics_validation.json
predictions_validation.csv                    per-row probs + preds
nohup_{train,eval}.out                        raw stdout (incl. tracebacks)
```

## Next steps (not done)

- **Hyperparameter tweaking is now free** (cache hit): C, class_weight,
  solver, threshold tuning on validation — no GPU needed.
- **Layer sweep needs new captures** — only layer 27 was saved (by design;
  all 8 full-attention layers would be 8× storage, still only ~130 MB, worth
  considering for attempt 2).
- Known data caveats from the preflight still apply: `final_label` is
  prompt-driven on every v2 row (no benign-prompt+harmful-response example),
  so this probe — like every v2 classifier so far — is evaluated on what is
  effectively prompt-harm classification.
- Test eval intentionally deferred until hyperparameters are locked.
