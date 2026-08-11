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
                │                                     transforms (Node bridge);
                │                                     same transform applied to
                ▼                                     prompt AND response
        data/train_weird_en.jsonl                  ← one row per (harmful English
        data/train_weird_tr.jsonl                    row, transform); labeled
                                                     harmful

        scripts/combine_dataset.py                 ← merges the artifacts into one
                │                                     training file, sampling the
                │                                     parseltongue pools down to
                ▼                                     ~20% of the final dataset
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
#   parseltongue rows = --parseltongue-frac (default 0.20) of the final dataset
#   then ~60% of the rows with responses get them truncated to a random prefix
#   (min 10 words kept; --truncate-frac / --truncate-min-length to tune)
.venv/bin/python scripts/combine_dataset.py
.venv/bin/python scripts/combine_dataset.py --parseltongue-frac 0.1   # 10% instead

# truncation step alone (reuse a pre-built augmented.jsonl, no LLM calls)
.venv/bin/python scripts/truncate_responses.py
.venv/bin/python scripts/truncate_responses.py --frac 0.5 --min-length 8  # tune

# the two steps, run individually:
.venv/bin/python scripts/translate_wildguard.py --n-examples 1000
.venv/bin/python scripts/augment_with_parseltongue.py --input data/harmful_translated.jsonl --output data/train_weird_tr.jsonl
.venv/bin/python scripts/augment_with_parseltongue.py --input data/harmful_en.jsonl --output data/train_weird_en.jsonl

API key: `OPENROUTER_API_KEY` env var or `.env` file (both gitignored).

## Notes / decisions

- **Language assignment:** each example is translated exactly once, into a
  seeded-random language drawn from the 30-language pool (~33 examples per
  language for a 1k run). Pass `--languages es,hi` to restrict the pool.
- **Labels:** translated rows keep the original labels. Parseltongue pools are
  generated from the harmful rows only (English + translated combined); each
  row's prompt AND response get the same transform applied. They are labeled
  harmful by design (obfuscation is treated as an adversarial signal in
  production). The final dataset keeps only a sampled subset of each parseltongue
  pool (default ~20% of the final file, `--parseltongue-frac`); nothing is
  hardcoded to a specific run size, so the same pipeline works for >1k prompts.
  `verified_accurate_description` is `false` for LLM translations (need
  spot-checking). Parseltongue rows inherit the flag from their source: transforms
  of unverified translations stay unverified; transforms of raw English rows are
  `true`.
- **Empty responses:** ~56% of wildguardtrain has no response; those rows get
  `response = ""` and we never let the model invent one.
- **Silent refusals:** DeepSeek sometimes returns the source text unchanged
  instead of translating harmful content. The translator detects no-op
  translations (exact match or >85% token overlap) and routes them to
  `translation_failures.jsonl` instead of the dataset.
- **Cost/latency (measured 2026):** 1k examples = 1k API calls. DeepSeek V3
  ~15s/call; V4 Flash ~9s/call and ~5-7x cheaper. See the translation script
  header for the full model list.