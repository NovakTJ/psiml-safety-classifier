"""Build `data/switched_prompt_dataset.jsonl`.

Motivation
----------
We train *exchange* classifiers (prompt + response -> one label).  In
deployment a prompt is often not obviously harmful to the classifier while the
response IS (multi-turn / context-dependence / the "$x example" in CLAUDE.md).
Every "harmful" exchange in `complete_dataset.jsonl` pairs a harmful prompt
with a harmful response, so without synthetic counterexamples the model can
trivially score by just reading the prompt and ignore the response.

This script fabricates exactly those counterexamples: it takes half of the
rows that have a *labeled harmful response with no response truncation*, and
swaps each row's (harmful) prompt for an *unharmful* prompt sampled from the
same `complete_dataset.jsonl`.  The response is untouched and is still
harmful, so the exchange label remains HARMFUL despite the benign prompt --
forcing the classifier to look at the response.

Selection rules (per the request):
  * rows with `response_harm_label == "harmful"` AND
    `response_truncated == False`  (no response cutting occurred)
  * take 50% of those rows (fixed seed, reproducible)
  * the `adversarial` column is deliberately NOT consulted
  * the replacement prompt is sampled (without replacement) from rows whose
    `prompt_harm_label == "unharmful"` in the same file

Output
------
Every row is a copy of the original row with:
  * `prompt` replaced by the sampled benign prompt
  * `prompt_harm_label` set to "unharmful" (it now describes the new prompt)
  * the original (harmful) prompt preserved in `original_prompt`
  * `original_prompt_harm_label` = "harmful"
  * `benign_prompt_source_row_id` = row_id of the benign prompt source
  * `switched` = true
The response + its labels are untouched, so the exchange is still harmful.

Run from the repo root with any python (pandas not required):
    python3 scripts/build_switched_prompt_dataset.py
"""

import json
import os
import random

DATA_PATH = "data/complete_dataset.jsonl"
OUT_PATH = "data/switched_prompt_dataset.jsonl"
SEED = 42
FRACTION = 0.5


def load_rows(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def build():
    rows = load_rows(DATA_PATH)

    # 1) candidates: labeled harmful response, no response truncation.
    candidates = [
        r
        for r in rows
        if r["response_harm_label"] == "harmful" and r["response_truncated"] is False
    ]
    print(f"complete_dataset rows: {len(rows)}")
    print(f"candidates (harmful response, no truncation): {len(candidates)}")

    # 2) take 50% deterministically.
    rng = random.Random(SEED)
    n_switch = int(len(candidates) * FRACTION)
    to_switch = rng.sample(candidates, n_switch)
    print(f"switching {len(to_switch)} rows ({FRACTION:.0%} of candidates)")

    # 3) benign prompt pool: unharmful prompts, sampled without replacement.
    benign_pool = [r for r in rows if r["prompt_harm_label"] == "unharmful"]
    if len(benign_pool) < n_switch:
        raise ValueError(
            f"need {n_switch} distinct benign prompts but only {len(benign_pool)} available"
        )
    benign_sources = rng.sample(benign_pool, n_switch)

    out_rows = []
    for src, benign in zip(to_switch, benign_sources):
        out = dict(src)
        out["original_prompt"] = src["prompt"]
        out["original_prompt_harm_label"] = src["prompt_harm_label"]
        out["prompt"] = benign["prompt"]
        out["prompt_harm_label"] = "unharmful"  # now describes the benign prompt
        out["benign_prompt_source_row_id"] = benign["row_id"]
        out["switched"] = True
        out_rows.append(out)

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(out_rows)} rows -> {OUT_PATH}")

    # sanity report
    nh = sum(1 for r in out_rows if r["response_harm_label"] == "harmful")
    npb = sum(1 for r in out_rows if r["prompt_harm_label"] == "unharmful")
    print(f"  rows with harmful response: {nh}")
    print(f"  rows with unharmful prompt: {npb}")
    print(f"  exchange harmful (prompt OR response): "
          f"{sum(1 for r in out_rows if r['prompt_harm_label']=='harmful' or r['response_harm_label']=='harmful')}")


if __name__ == "__main__":
    build()
