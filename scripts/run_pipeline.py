#!/usr/bin/env python
"""Run the full augmentation pipeline end-to-end.

Step 1  sample N examples from wildguardmix, translate each ONCE into a random
        language  (scripts/translate_wildguard.py)
Step 2  collect the harmful prompts from BOTH sources: the English originals
        (data/sample.jsonl) and the translated rows (data/translated.jsonl)
Step 3  obfuscate all of those harmful rows with P4RS3LT0NGV3 weird-character
        transforms (scripts/augment_with_parseltongue.py); every selected row
        gets the same transform applied to BOTH its prompt and its response, and
        is labeled harmful by design. The English rows and the translated rows
        are obfuscated separately into two pool files, which is equivalent to
        running on the combined set.
Step 4  combine everything into data/augmented.jsonl, sampling the parseltongue
        pool down to --parseltongue-frac of the final dataset
        (scripts/combine_dataset.py).
Step 5  truncate the responses of a random ~60% of the rows that have one, so
        the classifier learns to fire on half-finished generations
        (scripts/truncate_responses.py; redraws c when the kept prefix would
        be < --truncate-min-length words).

Artifacts (all in data/, gitignored):
    sample.jsonl               sampled original rows
    translated.jsonl           one row per example (labels preserved)
    harmful_en.jsonl           harmful subset of sample.jsonl (English)
    harmful_translated.jsonl   harmful subset of translated.jsonl
    train_weird_en.jsonl       parseltongue obfuscations of harmful English prompts
    train_weird_tr.jsonl       parseltongue obfuscations of harmful translated prompts
    translation_failures.jsonl
    augmented.jsonl            final combined dataset (with truncated responses)

Usage
-----
    # full run
    .venv/bin/python scripts/run_pipeline.py --n-examples 1000

    # reuse an existing translated.jsonl (e.g. after a crash or a new run)
    .venv/bin/python scripts/run_pipeline.py --n-examples 1000 --skip-translate

    # quick smoke test: 4 examples, languages drawn from {es, hi}
    .venv/bin/python scripts/run_pipeline.py --n-examples 4 --languages es,hi
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-examples", type=int, default=1000)
    parser.add_argument("--harmful-frac", type=float, default=0.5,
                        help="Fraction of the sample that is harmful (default: 0.5).")
    parser.add_argument("--languages", default="all",
                        help="Comma-separated ISO codes to draw from (default: all 30).")
    parser.add_argument("--model", default="deepseek/deepseek-chat-v3-0324")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-translate", action="store_true",
                        help="Reuse the existing translated.jsonl instead of translating.")
    parser.add_argument("--skip-obfuscate", action="store_true",
                        help="Skip the parseltongue step.")
    parser.add_argument("--parseltongue-frac", type=float, default=0.20,
                        help="Target share of parseltongue rows in the final dataset "
                             "(default: 0.20; capped by the pool size).")
    parser.add_argument("--skip-combine", action="store_true",
                        help="Skip the final combine step (run combine_dataset.py yourself).")
    parser.add_argument("--skip-truncate", action="store_true",
                        help="Skip the response-truncation step (keeps full responses).")
    parser.add_argument("--truncate-frac", type=float, default=0.6,
                        help="Share of rows with a response to truncate (default: 0.6).")
    parser.add_argument("--truncate-min-length", type=int, default=10,
                        help="Min words a kept prefix must have (default: 10).")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    return parser.parse_args(argv)


def run(cmd: list[str]) -> None:
    print(f">> {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"step failed with exit code {proc.returncode}")


def step_translate(args: argparse.Namespace, data_dir: Path) -> None:
    cmd = [
        sys.executable, "scripts/translate_wildguard.py",
        "--n-examples", str(args.n_examples),
        "--harmful-frac", str(args.harmful_frac),
        "--model", args.model,
        "--workers", str(args.workers),
        "--seed", str(args.seed),
        "--output", str(data_dir / "translated.jsonl"),
        "--failures", str(data_dir / "translation_failures.jsonl"),
        "--sample-jsonl", str(data_dir / "sample.jsonl"),
    ]
    if args.languages != "all":
        cmd += ["--languages", args.languages]
    run(cmd)


def step_filter_harmful(src: Path, dst: Path) -> int:
    """Keep harmful rows from src -> dst (English or translated)."""
    kept = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("prompt_harm_label") == "harmful":
                fout.write(line + "\n")
                kept += 1
    print(f">> filtered: {kept} harmful rows -> {dst}", file=sys.stderr)


def step_obfuscate(data_dir: Path, input_name: str, output_name: str) -> None:
    cmd = [
        sys.executable, "scripts/augment_with_parseltongue.py",
        "--input", str(data_dir / input_name),
        "--output", str(data_dir / output_name),
        "--force-harmful",
    ]
    run(cmd)


def step_truncate(args: argparse.Namespace, data_dir: Path) -> None:
    cmd = [
        sys.executable, "scripts/truncate_responses.py",
        "--input", str(data_dir / "augmented.jsonl"),
        "--output", str(data_dir / "augmented.jsonl"),
        "--frac", str(args.truncate_frac),
        "--min-length", str(args.truncate_min_length),
        "--seed", str(args.seed),
    ]
    run(cmd)


def summarize(data_dir: Path) -> None:
    for pool in ("train_weird_en.jsonl", "train_weird_tr.jsonl"):
        weird = data_dir / pool
        if not weird.exists():
            continue
        n = 0
        langs: dict[str, int] = {}
        transforms: dict[str, int] = {}
        for line in weird.open("r", encoding="utf-8"):
            rec = json.loads(line)
            n += 1
            langs[rec.get("language", "?")] = langs.get(rec.get("language", "?"), 0) + 1
            transforms[rec["encoding_type"]] = transforms.get(rec["encoding_type"], 0) + 1
        print(f"\nsummary: {n} obfuscated rows in {weird}", file=sys.stderr)
        print("  languages:", dict(sorted(langs.items())), file=sys.stderr)
        print(f"  distinct transforms: {len(transforms)} used", file=sys.stderr)


def step_combine(args: argparse.Namespace, data_dir: Path) -> None:
    cmd = [
        sys.executable, "scripts/combine_dataset.py",
        "--data-dir", str(data_dir),
        "--n-examples", str(args.n_examples),
        "--harmful-frac", str(args.harmful_frac),
        "--seed", str(args.seed),
        "--parseltongue-frac", str(args.parseltongue_frac),
    ]
    run(cmd)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_translate:
        step_translate(args, data_dir)

    if not args.skip_obfuscate:
        # Combined harmful set = harmful English originals + harmful translations.
        step_filter_harmful(data_dir / "sample.jsonl", data_dir / "harmful_en.jsonl")
        step_filter_harmful(data_dir / "translated.jsonl", data_dir / "harmful_translated.jsonl")
        step_obfuscate(data_dir, "harmful_en.jsonl", "train_weird_en.jsonl")
        step_obfuscate(data_dir, "harmful_translated.jsonl", "train_weird_tr.jsonl")
        summarize(data_dir)

    if not args.skip_combine:
        step_combine(args, data_dir)

    if not args.skip_truncate:
        step_truncate(args, data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())