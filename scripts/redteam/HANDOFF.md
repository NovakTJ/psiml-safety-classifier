# Handoff — Jailbreak dataset generation for Qwen3.5-9B (v1)

Status: **ready to run, not yet executed**. Written 2026-08-12. All files are in this
directory (`scripts/redteam/`) + one artifact dir (`psiml_data/jailbreak_v1/`, empty).

## What we're doing

WildGuardMix's adversarial prompts scored **0/30 jailbreaks** against Qwen3.5-9B
(Aug 12 probe). The dataset is "too easy" — so we generate our own jailbroken Qwen3.5
responses using real jailbreak templates (G0DM0D3), judge them with Qwen3Guard-Gen-8B,
and use the result as v1 training data. **Generation must come from Qwen3.5-9B itself**
(guarded model), not from permissive models (Hermes/Grok etc.) — that would recreate the
"random old outputs" distribution mismatch we're trying to escape.

## Files created

| File | Purpose |
|---|---|
| `scripts/redteam/g0dm0d3_source/godmode-prompt.ts` | Verbatim copy of G0DM0D3's `GODMODE_SYSTEM_PROMPT` (commit `f630176`, cloned from github.com/elder-plinius/G0DM0D3) |
| `scripts/redteam/g0dm0d3_source/libertas.ts` | Verbatim copy of G0DM0D3's `HALL_OF_FAME` (5 L1B3RT4S-style personas: grok-420, gemini-reset, gpt-classic, claude-inversion, hermes-fast) |
| `scripts/redteam/jailbreak_templates.py` | Python port of both template sets + `build_messages(template_id, prompt)` + refusal-regex helper |
| `scripts/redteam/generate_jailbreak_dataset.py` | Generation script — loads **Qwen3.5-9B from `/data/models/Qwen3.5-9B`** (run in `qwen35_env`), writes raw responses JSONL |
| `scripts/redteam/label_jailbreak_dataset.py` | Labeling script — loads **Qwen3Guard-Gen-8B from `/data/models/Qwen3.5-9B`** (run in `ccpp_env`), appends `guard_raw` + `is_refusal` per row |
| `psiml_data/probes/qwen35_openrouter_probe_aug12.jsonl` | Preserved 9-row partial from the *failed thinking-mode* OpenRouter probe (demo of the empty-`content` pitfall) |

## Correctness notes on the template port (read before running)

1. **GODMODE prompt**: online quotes are condensed paraphrases. The real one is 111
   lines (sections I–V) — we copied it verbatim.
2. **Placeholder injection bug fixed**: G0DM0D3's own `injectQuery()` replaces the raw
   `<user_query>` wrapper tags globally, which double-injects the prompt in the
   gemini-reset user turn (`prompt+prompt`). Our `_inject()` only replaces literal
   placeholder tokens (`{Z}`, `{QUERY}`, `<|user_query|>`, ...); the gemini user turn
   returns the raw prompt. The `<user_query>` inside the *gemini system prompt* is
   intentional literal text — leave it.
3. **Faithful user turns**: grok-420 → `variable Z = [prompt]`, gpt-classic →
   `prompt=prompt`, gemini/claude/hermes/godmode → raw `prompt`. These wrappers are
   part of the original templates, not bugs.

## Corrected misconceptions (from the dumber-model suggestion)

- **Non-thinking mode is mandatory on OpenRouter**: `"reasoning": {"enabled": false}`
  (the Anthropic-style `{"thinking": {"type": "disabled"}}` is wrong). Without it Qwen3.5
  returns `content=""`, `finish_reason="length"`. G0DM0D3's own `queryModel` does NOT
  set this.
- **Do NOT race across models** (Hermes/Grok) — race across the 6 templates only.
- **Refusal regex ≠ harm label** — absence of "I can't" isn't compliance; use
  Qwen3Guard-Gen-8B as judge. Regex (`is_refusal`) is a weak pre-filter only.
- **Temperature doesn't override refusals** — it's for output diversity, not jailbreak.
- **Local generation preferred**: free, no rate limits, exact guarded weights, no
  jailbreak traffic through OpenRouter. OpenRouter (`qwen/qwen3.5-9b`) only for pilots.
- Model ID `qwen/qwen3.5-plus-02-15` exists on OpenRouter but is a DIFFERENT model than
  the guarded Qwen3.5-9B.

## How to run

```bash
# 1. Generate (in qwen35_env — only env that can load the qwen3_5 architecture):
cd /home/mls01/scripts/redteam
/home/mls01/.conda/envs/qwen35_env/bin/python generate_jailbreak_dataset.py \
    --n-prompts 10 --out /home/mls01/psiml_data/jailbreak_v1/raw_qwen35.jsonl

# 2. Label (in ccpp_env — loads Qwen3Guard):
/home/mls01/ccpp_env/bin/python label_jailbreak_dataset.py \
    --in /home/mls01/psiml_data/jailbreak_v1/raw_qwen35.jsonl \
    --out /home/mls01/psiml_data/jailbreak_v1/labeled_qwen35.jsonl
```

- Generation params: `--seed 0 --max-new-tokens 2048 --temperature 1.0 --top-p 0.95`
  (10 prompts × 6 templates = 60 generations; stratified across subcategories from
  `data/complete_dataset.jsonl`, EN + `prompt_harm_label=harmful` only, 750 available).
- The generation script sets the required env-var block (USER/LOGNAME/TORCHINDUCTOR/
  TRITON/XDG_CACHE_HOME) **before** importing transformers — required in `qwen35_env`
  (getpwuid crash for uid 1562).

## Current machine state (important)

- **GPU 0 (A100 40GB): ~20.9 GB already in use** by two processes not visible in our
  namespace (`ps` shows nothing; `nvidia-smi` shows 18550 MiB + 2356 MiB). They cannot
  be killed from here. ~19 GB free — Qwen3.5-9B bf16 needs ~17 GB, so it should fit
  with `device_map="auto"`, but **watch for OOM**; if it fails, wait for the other
  container to finish or reduce to `--max-new-tokens 1024`.
- Prior probe script/results were lost once to `/tmp` cleanup — keep everything under
  the repo (that's why templates + scripts live in `scripts/redteam/`).

## Next steps after this pilot

1. Inspect `labeled_qwen35.jsonl`: jailbreak rate per template (harmful responses /
   total), which templates actually break Qwen3.5-9B.
2. Scale up: more prompts (all 750), keep only successful jailbreaks + refusals, add
   benign prompts for the SAFE class.
3. Truncate harmful responses into prefixes (Idea 1/2) for mid-stream stopping data.
4. Split train/test before any training.
