# aegis-chat — usage

Guarded chat with Qwen3.5-9B (OpenRouter) watched by the Gemma-3-1B LoRA
exchange classifier, with mid-stream blocking. Spec: `PLAN.md`. Engine: `core.py`.

## Setup (off-cluster: laptop / GCP VM)

On the **cluster** everything just works (`ccpp_env`, `/data/models`, adapter in
`scripts/model/results/`). Off-cluster:

```bash
python3 -m venv ~/aegis_env
~/aegis_env/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/aegis_env/bin/pip install "transformers==4.57.6" peft accelerate

# base model: google/gemma-3-1b-it is license-gated; the unsloth mirror is
# identical and ungated — core.py uses it automatically when /data/models
# is absent (override with AEGIS_BASE_MODEL).
~/aegis_env/bin/hf download unsloth/gemma-3-1b-it

# adapter (25 MB): copy from the cluster into this exact repo-relative path
# (the dir is gitignored, so it must be synced out-of-band):
#   scripts/model/results/gemma_lora_v2_sweep_phase2/lr3e-4_r8_alpha16_dropout0.0_seed42/best_adapter/

# OpenRouter key: put OPENROUTER_API_KEY=... in the repo-root .env
# (cli.py loads it) or export it.
```

## Run

```bash
# human REPL
~/aegis_env/bin/python scripts/redteam/aegis/cli.py --device cpu

# machine mode (one user message per stdin line, JSON events on stdout)
echo "hi" | ~/aegis_env/bin/python scripts/redteam/aegis/cli.py --jsonl --device cpu
```

REPL slash commands: `/reset /thinking on|off /threshold F /check-every N
/stats /config /help /quit`.

Key flags (full list: `--help`): `--adapter PATH|none` (none = bare zero-shot
Gemma), `--guard-prompt '...'|@file`, `--model OPENROUTER_ID` (default
`qwen/qwen3.5-9b`), `--threshold`, `--check-every`, `--thinking on|off`,
`--log-dir`. See PLAN.md "Expected-to-change pieces" for why these are knobs.

Session logs: one JSONL per session in `psiml_data/redteam_sessions/`
(config header incl. target model + guard prompt, per-turn checks with
latencies, verdicts; the REAL blocked partial is preserved in the log while the
model-visible history gets a block notice). These are red-team evidence and v3
dataset raw material — keep them.

## CPU performance (this laptop, WSL2, 8 cores)

Guard check ≈ 3–5 s (grows slightly with exchange length past the 512-token
sliding window). The A100 does this in single-digit ms. For a snappier local
demo, raise `--check-every` (e.g. 150) at the cost of later blocks.

## Tests

```bash
PY=~/aegis_env/bin/python   # or /home/mls01/ccpp_env/bin/python on the cluster
$PY scripts/redteam/aegis/tests/test_core_logic.py         # no model/network
$PY scripts/redteam/aegis/tests/test_text_construction.py  # no model/network
$PY scripts/redteam/aegis/tests/test_guard_live.py --device cpu [--kv-only|--logit-only] [--dtype fp32]
$PY scripts/redteam/aegis/tests/debug_kv.py                # fp32 exactness proof, crosses the 512 window
```

- `--kv-only` = incremental-vs-full scoring check. bf16 tolerance is 1e-2
  (chunked-vs-full matmul noise, worst 5.7e-3 seen short / 1.8e-2 past the
  512-token window); `--dtype fp32` proves the logic exact (~1e-6).
- `--logit-only` needs the sweep's `validation_predictions_epoch_6.csv`, which
  is cluster-only (gitignored results dir) — run it there.

## Implementation notes (why it looks like this)

- **Score = softmax over first-token logits** (`harm` vs `un`) — identical
  decision to the sweep's greedy generation, thresholdable from one forward.
- **KV reuse:** the real cache holds exactly prefix + response-so-far; each
  check forwards only the BPE-realized delta. The 5 template-suffix tokens are
  forwarded over a **throwaway deep copy** of the cache — never the real one —
  because gemma-3-1b's `DynamicSlidingWindowLayer` refuses to `crop()` once the
  exchange is longer than its 512-token window (the original crop-back scheme
  crashed mid-stream there, 2026-08-13). The rare BPE-realign crop falls back
  to a one-shot re-prime forward if it hits that limit.
- **Blocking:** stream aborted, partial withheld from model history (attacker
  can't read the leak and adapt), fixed block notice substituted; real partial
  in the session log only.
- Blocked-benign turns are false-positive evidence (PLAN.md
  "Expected-to-change" #4) — the log keeps `p_harmful` + token offset for them.
