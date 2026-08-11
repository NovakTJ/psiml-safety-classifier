# psiml-safety-classifier

Multilingual + obfuscated-text augmentation pipeline for the
[allenai/wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix) safety dataset.

## Pipeline

```
wildguardmix (train, 86,759 rows)
        │
        ▼
scripts/run_pipeline.py          ← orchestrates both steps below (subprocess)
        │
        ├──► scripts/translate_wildguard.py        ← each example translated ONCE
        │        │                                    into one seeded-random language
        │        ▼
        │      data/translated.jsonl                ← labels preserved
        │
        └──► harmful prompts (English originals + translated)
                │
                ▼
        scripts/augment_with_parseltongue.py       ← P4RS3LT0NGV3 weird-char
                │                                     transforms (Node bridge),
                │                                     in two modes:
                │                                     (a) same transform on prompt
                │                                     AND response; (b) transform on
                ▼                                     prompt ONLY, response kept plain
        data/train_weird_en.jsonl                  ← one row per (harmful row,
        data/train_weird_tr.jsonl                    transform); labeled harmful
        data/train_weird_promptonly_en.jsonl       ← obfuscated prompt + plain
        data/train_weird_promptonly_tr.jsonl         response (the common real-life
                                                     case: models answer parseltongue
                                                     prompts in plain text,
                                                     especially when refusing)

        scripts/combine_dataset.py                 ← merges the artifacts into one
                │                                     training file, sampling the
                │                                     parseltongue pools down to
                │                                     ~20% of the final dataset
                ▼                                     (half of that prompt-only)
        data/augmented.jsonl                       ← final dataset
                │
                ▼
        scripts/truncate_responses.py              ← shorten ~60% of the rows with
                                                     responses to a random prefix, so
                                                     the classifier learns to fire on
                                                     half-finished generations
                ▼
        data/augmented.jsonl                       ← final dataset (truncated responses)
```

## Output schema (translated.jsonl)

The translated text overwrites the original `prompt` / `response` fields in
place (no parallel column), plus:

| column | meaning |
|---|---|
| `prompt` / `response` | in-place model translation (response = `""` if the original had none) |
| `original_idx` | row index in the wildguardtrain split |
| `source_split` | `train` |
| `language` | ISO 639-1 code (**one seeded-random language per example**; pool of 30) |
| `encoding_type` | `none` for translation rows, transform key for parseltongue rows |
| `translation_model` | OpenRouter model id |
| `prompt_template_version` / `augmentation_pipeline_version` | template + pipeline versioning |
| `timestamp` | UTC ISO |
| `verified_accurate_description` | always `false` — translations need a human spot-check before use |
| `notes` | flags e.g. `no_response`, `empty_translation` |

Failures (refusals, unparseable JSON) go to `data/translation_failures.jsonl`, never the main output.

## Output schema (augmented.jsonl)

The final training file is schema-consistent: every row — original, translation,
obfuscation_en, obfuscation_tr, truncated or not — carries the same columns.
Where a column does not apply to a row type it keeps an *empty value* (``None`` /
``""`` / ``False``) instead of dropping the key, so a loader reads `row[col]`
directly, no `row.get()` chains or per-row-type branches.

`scripts/dataset_schema.py` is the single source of truth (`FINAL_COLUMNS`,
`INTERMEDIATE_COLUMNS`, `normalize()`); every step that writes a dataset passes
its rows through it.

| column | on every row | notes |
|---|---|---|
| `row_id`, `augmentation_type`, `original_idx`, `source_split`, `language`, `encoding_type`, `prompt_template_version`, `augmentation_pipeline_version`, `timestamp`, `notes` | yes | `source_split`=`train`; `prompt_template_version`/`augmentation_pipeline_version`=`v1`; `encoding_type`=`none` or the transform key |
| `translation_model` | yes | OpenRouter model id, or `null` for rows that were never machine-translated (originals + their obfuscations) |
| `verified_accurate_description` | yes | `true` for raw / obfuscated-raw rows; `false` for LLM translations and their obfuscations |
| `response`, `response_harm_label`, `response_refusal_label` | yes | `""` / `null` when the source row had no response |
| `response_truncated` | yes | `false`, or `true` when the response was shortened |
| `response_truncation_c`, `response_truncated_words` | yes | meaningful only when `response_truncated` is `true`; `null` otherwise |

The parseltongue debug columns (`__original_prompt__`, `__original_response__`,
`__transform__`) exist only on the intermediate pools (`train_weird_*.jsonl` —
the combine step reads `__original_prompt__` to recover the original row index)
and are dropped from the final file: the transform key is already in
`encoding_type`, and the transformed text is in `prompt`/`response`.

## Usage

```bash
# full pipeline: 1000 examples (each -> 1 random language), then obfuscate harmful rows
.venv/bin/python scripts/run_pipeline.py --n-examples 1000

# reuse an existing translated.jsonl (crash recovery / tuning obfuscation only)
.venv/bin/python scripts/run_pipeline.py --n-examples 1000 --skip-translate

# smoke test
.venv/bin/python scripts/run_pipeline.py --n-examples 4 --languages es,hi

# cheaper/faster model
.venv/bin/python scripts/run_pipeline.py --n-examples 1000 --model deepseek/deepseek-v4-flash-0731

# merge everything into one clean training file (data/augmented.jsonl)
#   parseltongue rows = --parseltongue-frac (default 0.20) of the final dataset,
#   half of them prompt-only (--prompt-only-share, default 0.5); then ~60% of
#   the rows with responses get them truncated to a random prefix
#   (min 10 words kept; --truncate-frac / --truncate-min-length to tune)
.venv/bin/python scripts/combine_dataset.py
.venv/bin/python scripts/combine_dataset.py --parseltongue-frac 0.1   # 10% instead

# truncation step alone (reuse a pre-built augmented.jsonl, no LLM calls)
.venv/bin/python scripts/truncate_responses.py
.venv/bin/python scripts/truncate_responses.py --frac 0.5 --min-length 8  # tune

# the steps, run individually:
.venv/bin/python scripts/translate_wildguard.py --n-examples 1000
.venv/bin/python scripts/augment_with_parseltongue.py --input data/harmful_translated.jsonl --output data/train_weird_tr.jsonl
.venv/bin/python scripts/augment_with_parseltongue.py --input data/harmful_en.jsonl --output data/train_weird_en.jsonl
# prompt-only variants (plain responses, the common real-life case):
.venv/bin/python scripts/augment_with_parseltongue.py --input data/harmful_en.jsonl --output data/train_weird_promptonly_en.jsonl --fields prompt --note parseltongue_prompt_only
.venv/bin/python scripts/augment_with_parseltongue.py --input data/harmful_translated.jsonl --output data/train_weird_promptonly_tr.jsonl --fields prompt --note parseltongue_prompt_only

API key: `OPENROUTER_API_KEY` env var or `.env` file (both gitignored).

## Notes / decisions

- **Language assignment:** each example is translated exactly once, into a
  seeded-random language drawn from the 30-language pool (~33 examples per
  language for a 1k run). Pass `--languages es,hi` to restrict the pool.
- **Labels:** translated rows keep the original labels. Parseltongue pools are
  generated from the harmful rows only (English + translated combined), in two
  modes: each row's prompt AND response get the same transform applied
  (`train_weird_*.jsonl`), or the prompt ONLY is transformed and the response is
  kept verbatim with its original response labels (`train_weird_promptonly_*.jsonl`,
  marked `parseltongue_prompt_only` in `notes`). The prompt-only mode mirrors
  production: models almost never mirror zalgo/leetspeak in their output —
  especially refusals, which come out plain — so obfuscated-prompt/plain-response
  is the dominant real-life case. All parseltongue rows are prompt-labeled
  harmful by design (obfuscation is treated as an adversarial signal in
  production). The final dataset keeps only a sampled subset of the parseltongue
  pools (default ~20% of the final file, `--parseltongue-frac`), split between
  the two modes via `--prompt-only-share` (default 0.5); nothing is hardcoded
  to a specific run size, so the same pipeline works for >1k prompts.
  `verified_accurate_description` is `false` for LLM translations (need
  spot-checking). Parseltongue rows inherit the flag from their source: transforms
  of unverified translations stay unverified; transforms of raw English rows are
  `true`.
- **A caveat on response-only training:** if the classifier is trained ONLY on
  `response_harm_label`, the forced harmful prompt labels on parseltongue rows
  have no effect — and prompt-only rows with benign/refusal responses would
  teach the model to ALLOW obfuscated prompts, the opposite of the "users must
  speak normally" policy. That policy is a surface property of the prompt, not
  expressible as response harm. Prefer enforcing it with a deterministic
  pre-filter (combining-mark density for zalgo, NFKC-confusables for fancy
  Unicode, substitution ratio for leetspeak — most transforms are even
  reversible via NFKC normalization, so deobfuscate-then-classify works), or
  train on the deployment decision (`block = obfuscated OR harmful(prompt) OR
  harmful(response)`) instead of the raw response label.
- **Empty responses:** ~56% of wildguardtrain has no response; those rows get
  `response = ""` and we never let the model invent one.
- **Schema consistency:** every dataset the pipeline writes is
  schema-consistent (every row has every column; empty cells where a column
  does not apply), enforced by `scripts/dataset_schema.py` and documented
  under "Output schema (augmented.jsonl)" above.
- **Silent refusals:** DeepSeek sometimes returns the source text unchanged
  instead of translating harmful content. The translator detects no-op
  translations (exact match or >85% token overlap) and routes them to
  `translation_failures.jsonl` instead of the dataset.
- **Cost/latency (measured 2026):** 1k examples = 1k API calls. DeepSeek V3
  ~15s/call; V4 Flash ~9s/call and ~5-7x cheaper. See the translation script
  header for the full model list.