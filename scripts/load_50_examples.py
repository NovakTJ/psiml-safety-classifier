#!/usr/bin/env python
"""Load the first 50 examples from allenai/wildguardmix without pulling the whole dataset.

Note: wildguardmix is a *gated* dataset. You must be logged in to the Hugging
Face Hub. If you see a 401/Not Found error, run once:

    huggingface-cli login

...and make sure you've accepted the dataset terms on
https://huggingface.co/datasets/allenai/wildguardmix

Alternatively, export your token before running (same scope as `login`):

    export HF_TOKEN=hf_...

The `datasets` library caches downloads under ~/.cache/huggingface/datasets, so
re-runs hit local cache instead of the network. We also use `streaming=True` and
slice `train[:N]` so we never materialize / download the full dataset.

Usage:
    .venv/bin/python scripts/load_50_examples.py
"""

from itertools import islice

from datasets import load_dataset

N = 50
CONFIG = "wildguardtrain"  # the only other config is "wildguardtest"
SPLIT = "train"

# stream so we don't download the full dataset just to peek at 50 rows.
ds = load_dataset("allenai/wildguardmix", CONFIG, split=SPLIT, streaming=True)

subset = list(islice(ds, N))
print(f"Selected {len(subset)} examples from split '{SPLIT}'.")
print(f"Columns: {list(subset[0].keys())}")

# Pretty-print the first example so you can see the schema.
first = subset[0]
for key, value in first.items():
    val = str(value)
    if len(val) > 300:
        val = val[:300] + "...[truncated]"
    print(f"\n--- {key} ---")
    print(val)
