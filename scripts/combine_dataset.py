#!/usr/bin/env python
"""Combine the augmentation artifacts into one clean training JSONL.

Includes:
    original   English examples from the sampled 1k (data/sample.jsonl equiv.)
    translation  the successfully translated rows (data/translated.jsonl)
    obfuscation  parseltongue weird-char rows (data/train_weird.jsonl)

Excludes anything that failed translation (data/translation_failures.jsonl) —
the 23 rows that did not produce a usable translation are not in
translated.jsonl, so they simply never appear here.

Each row gets:
    row_id              unique id ("orig-*", "trans-*", "weird-*")
    augmentation_type   original | translation | obfuscation
    verified_accurate_description  True for originals; False for translations
                                 (and inherited False for their obfuscations)

Usage:
    .venv/bin/python scripts/combine_dataset.py                 # -> data/augmented.jsonl
    .venv/bin/python scripts/combine_dataset.py --n-examples 1000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from translate_wildguard import load_and_sample  # noqa: E402

PIPELINE_VERSION = "v1"
TEMPLATE_VERSION = "v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSONL (default: data/augmented.jsonl).")
    parser.add_argument("--n-examples", type=int, default=1000)
    parser.add_argument("--harmful-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Keep rows grouped by augmentation_type instead of shuffling.")
    return parser.parse_args(argv)


def enrich_original(idx: int, rec: dict[str, Any], now: str) -> dict[str, Any]:
    out = dict(rec)
    out["row_id"] = f"orig-{idx}"
    out["augmentation_type"] = "original"
    out["original_idx"] = idx
    out["source_split"] = "train"
    out["language"] = "en"
    out["encoding_type"] = "none"
    out["translation_model"] = None
    out["prompt_template_version"] = TEMPLATE_VERSION
    out["augmentation_pipeline_version"] = PIPELINE_VERSION
    out["timestamp"] = now
    out["verified_accurate_description"] = True  # raw dataset text/labels, nothing transformed
    out["notes"] = ""
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir
    output = args.output or data_dir / "augmented.jsonl"
    now = datetime.now(timezone.utc).isoformat()

    # Re-sample deterministically to get original_idx for the raw English rows.
    selected, records = load_and_sample(args.n_examples, args.harmful_frac, args.seed)
    orig_by_idx = {idx: rec for idx, rec in zip(selected, records)}

    translated: list[dict[str, Any]] = []
    for line in (data_dir / "translated.jsonl").open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec["original_idx"] not in orig_by_idx:
            print(f"warning: translated row idx={rec['original_idx']} not in sample; skipping", file=sys.stderr)
            continue
        rec["augmentation_type"] = "translation"
        rec["row_id"] = f"trans-{rec['original_idx']}-{rec['language']}"
        translated.append(rec)

    weird: list[dict[str, Any]] = []
    for line in (data_dir / "train_weird.jsonl").open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rec["augmentation_type"] = "obfuscation"
        rec["row_id"] = f"weird-{rec['original_idx']}-{rec['language']}-{rec['encoding_type']}"
        weird.append(rec)

    originals = [enrich_original(idx, rec, now) for idx, rec in zip(selected, records)]

    rows = originals + translated + weird
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(rows)

    with output.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    from collections import Counter
    by_type = Counter(r["augmentation_type"] for r in rows)
    by_label = Counter(r["prompt_harm_label"] for r in rows)
    n_langs = len({r["language"] for r in rows})
    print(f"wrote {len(rows)} rows -> {output}", file=sys.stderr)
    print(f"  by type:     {dict(by_type)}", file=sys.stderr)
    print(f"  by label:    {dict(by_label)}", file=sys.stderr)
    print(f"  languages:   {n_langs}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())