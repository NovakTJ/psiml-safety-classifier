#!/usr/bin/env python
"""Combine the augmentation artifacts into one training JSONL.

Categories: original (sampled English rows), translation (translated.jsonl),
obfuscation_en / obfuscation_tr (parseltongue pools). The parseltongue pools
are sampled down to --parseltongue-frac of the final dataset, split
proportionally to pool sizes. Failed translations simply never appear.

Usage:
    .venv/bin/python scripts/combine_dataset.py --parseltongue-frac 0.1
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

from dataset_schema import FINAL_COLUMNS, normalize  # noqa: E402
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
    parser.add_argument("--parseltongue-frac", type=float, default=0.20,
                        help="Target share of obfuscation rows in the final dataset "
                             "(default: 0.20). Exact count is computed from the actual "
                             "base size: P = B * frac / (1 - frac).")
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

    # Re-sample deterministically to recover original_idx for the raw rows.
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

    def load_pool(name: str, aug_type: str, prefix: str) -> list[dict[str, Any]]:
        path = data_dir / name
        pool: list[dict[str, Any]] = []
        if not path.exists():
            print(f"info: {path.name} missing; empty {aug_type} pool", file=sys.stderr)
            return pool
        for line in path.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            idx = rec.get("original_idx")
            if idx is None:
                # English rows carry no index; recover via the original prompt.
                match_text = rec.get("__original_prompt__") or rec.get("prompt")
                idx = orig_by_prompt.get(match_text)
                if idx is None:
                    print(f"warning: {name} row without matching original, dropping", file=sys.stderr)
                    continue
                rec["original_idx"] = idx
                rec["language"] = "en"
            rec["augmentation_type"] = aug_type
            rec["row_id"] = f"weird-{prefix}-{idx}-{rec['encoding_type']}"
            # Every pool row comes from the same source split / pipeline generation.
            rec["source_split"] = "train"
            rec["prompt_template_version"] = TEMPLATE_VERSION
            rec["augmentation_pipeline_version"] = PIPELINE_VERSION
            pool.append(rec)
        return pool

    originals = [enrich_original(idx, rec, now) for idx, rec in zip(selected, records)]
    orig_by_prompt = {rec["prompt"]: idx for idx, rec in zip(selected, records)}

    pools = [
        ("train_weird_en.jsonl", "obfuscation_en", "en", load_pool("train_weird_en.jsonl", "obfuscation_en", "en")),
        ("train_weird_tr.jsonl", "obfuscation_tr", "tr", load_pool("train_weird_tr.jsonl", "obfuscation_tr", "tr")),
    ]

    rng = random.Random(args.seed)
    n_base = len(originals) + len(translated)
    target = int(round(n_base * args.parseltongue_frac / (1.0 - args.parseltongue_frac)))
    avail = sum(len(pool) for _, _, _, pool in pools)
    if target >= avail:
        print(f"info: target {target} parseltongue rows exceeds pool ({avail}); keeping all", file=sys.stderr)
        weird = [rec for _, _, _, pool in pools for rec in pool]
    elif target <= 0:
        weird = []
    else:
        weird = []
        for name, aug_type, prefix, pool in pools:
            share = int(round(target * len(pool) / avail)) if avail else 0
            share = min(share, len(pool))
            weird.extend(rng.sample(pool, share))
        drift = target - len(weird)
        if drift > 0:
            leftovers = [rec for _, _, _, pool in pools for rec in pool if rec not in weird]
            weird.extend(rng.sample(leftovers, min(drift, len(leftovers))))
        elif drift < 0:
            weird = rng.sample(weird, len(weird) + drift)

    rows = originals + translated + weird
    if not args.no_shuffle:
        rng.shuffle(rows)

    # Canonical schema for the training file: every row has every column
    # (missing ones are filled with empty values) and the parseltongue debug
    # columns (__original_prompt__ etc.) are dropped.  See dataset_schema.py.
    rows = [normalize(rec, FINAL_COLUMNS) for rec in rows]

    with output.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    from collections import Counter
    by_type = Counter(r["augmentation_type"] for r in rows)
    by_label = Counter(r["prompt_harm_label"] for r in rows)
    n_langs = len({r["language"] for r in rows})
    n_weird = by_type["obfuscation_en"] + by_type["obfuscation_tr"]
    print(f"wrote {len(rows)} rows -> {output}", file=sys.stderr)
    print(f"  by type:     {dict(by_type)}", file=sys.stderr)
    print(f"  by label:    {dict(by_label)}", file=sys.stderr)
    print(f"  languages:   {n_langs}", file=sys.stderr)
    print(f"  parseltongue share: {n_weird}/{len(rows)} = {n_weird/len(rows):.1%}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())