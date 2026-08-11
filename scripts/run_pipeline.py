#!/usr/bin/env python
"""Run the full augmentation pipeline end-to-end.

Step 1  sample N examples from wildguardmix, translate each ONCE into a random
        language  (scripts/translate_wildguard.py)
Step 2  keep only the harmful rows from the translation output
Step 3  obfuscate the harmful *translated* prompts with P4RS3LT0NGV3
        weird-character transforms (scripts/augment_with_parseltongue.py);
        those rows are labeled harmful by design.

Artifacts (all in data/, gitignored):
    sample.jsonl            sampled original rows
    translated.jsonl        one row per example (labels preserved)
    harmful_translated.jsonl  filtered harmful subset of translated.jsonl
    train_weird.jsonl       parseltongue obfuscations of harmful translated prompts
    translation_failures.jsonl

Usage
-----
    # full run
    .venv/bin/python scripts/run_pipeline.py --n-examples 1000

    # reuse an existing translated.jsonl (e.g. after a crash or a new run)
    .venv/bin/python scripts/run_pipeline.py --n-examples 1000 --skip-translate

    # quick smoke test: 4 examples, languages drawn from {es, hi}
    .venv/bin/python scripts/run_pipeline.py --n-examples 4 --languages es,hi

    # cheaper/faster model
    .venv/bin/python scripts/run_pipeline.py --n-examples 1000 --model deepseek/deepseek-v4-flash-0731
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


def step_filter_harmful(data_dir: Path) -> int:
    """Keep harmful rows from translated.jsonl -> harmful_translated.jsonl."""
    src = data_dir / "translated.jsonl"
    dst = data_dir / "harmful_translated.jsonl"
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
    return kept


def step_obfuscate(args: argparse.Namespace, data_dir: Path) -> None:
    cmd = [
        sys.executable, "scripts/augment_with_parseltongue.py",
        "--input", str(data_dir / "harmful_translated.jsonl"),
        "--output", str(data_dir / "train_weird.jsonl"),
        "--text-col", "translated_prompt",
        "--force-harmful",
    ]
    run(cmd)


def summarize(data_dir: Path) -> None:
    weird = data_dir / "train_weird.jsonl"
    if not weird.exists():
        return
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_translate:
        step_translate(args, data_dir)

    if not args.skip_obfuscate:
        n_harmful = step_filter_harmful(data_dir)
        if n_harmful == 0:
            print("warning: no harmful rows in translated.jsonl; skipping obfuscation", file=sys.stderr)
        else:
            step_obfuscate(args, data_dir)
            summarize(data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())