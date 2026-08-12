#!/usr/bin/env python
"""Peek at the first 50 examples of allenai/wildguardmix (gated dataset;
requires `huggingface-cli login`). Streams, so the full dataset is never
downloaded.

Usage:
    .venv/bin/python scripts/load_50_examples.py
"""

from itertools import islice

from datasets import load_dataset

N = 50
CONFIG = "wildguardtrain"  # the only other config is "wildguardtest"
SPLIT = "train"

ds = load_dataset("allenai/wildguardmix", CONFIG, split=SPLIT, streaming=True)

subset = list(islice(ds, N))
print(f"Selected {len(subset)} examples from split '{SPLIT}'.")
print(f"Columns: {list(subset[0].keys())}")

first = subset[0]
for key, value in first.items():
    val = str(value)
    if len(val) > 300:
        val = val[:300] + "...[truncated]"
    print(f"\n--- {key} ---")
    print(val)
