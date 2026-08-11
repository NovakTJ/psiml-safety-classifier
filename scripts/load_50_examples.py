#!/usr/bin/env python
"""One-time script: load the first 50 examples from allenai/wildguardmix.

Note: wildguardmix is a *gated* dataset. You must be logged in to the Hugging
Face Hub. If you see a 401/Not Found error, run once:

    huggingface-cli login

...and make sure you've accepted the dataset terms on https://huggingface.co/datasets/allenai/wildguardmix

Usage:
    .venv/bin/python scripts/load_50_examples.py
"""

from datasets import load_dataset

N = 50
CONFIG = "wildguardtrain"  # the only other config is "wildguardtest"
SPLIT = "train"

ds = load_dataset("allenai/wildguardmix", CONFIG, split=SPLIT)
print(f"Dataset has {ds.num_rows:,} rows in split '{SPLIT}'")
print(f"Columns: {ds.column_names}")

subset = ds.select(range(N))
print(f"\nSelected {len(subset)} examples.")
print(f"Selected subset columns: {subset.column_names}")

# Pretty-print the first example so you can see the schema.
first = subset[0]
for key, value in first.items():
    val = str(value)
    if len(val) > 300:
        val = val[:300] + "...[truncated]"
    print(f"\n--- {key} ---")
    print(val)
