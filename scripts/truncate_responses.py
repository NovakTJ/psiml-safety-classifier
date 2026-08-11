#!/usr/bin/env python
"""Shorten the responses of a share of the rows that have one.

The classifier has to fire on half-finished generations, so a random --frac of
rows with a response gets its response replaced by a random prefix: draw
c in [0, 1), keep the first round(c * N) words; redraw while the prefix is
shorter than --min-length words. Responses with N <= min-length words (and
whitespace-less responses, e.g. CJK text or some parseltongue transforms) are
kept whole.

The drawn c is deterministic per (seed, row_id); already-truncated rows are
never truncated again.

Usage:
    .venv/bin/python scripts/truncate_responses.py \
        --input data/augmented.jsonl --output data/augmented.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# A "word" is any maximal run of non-whitespace.
_WORD_RE = re.compile(r"\S+")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "augmented.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "augmented.jsonl")
    parser.add_argument("--frac", type=float, default=0.6,
                        help="Share of rows with a response to truncate (default: 0.6).")
    parser.add_argument("--min-length", type=int, default=10,
                        help="Minimum number of words a kept prefix must have; "
                             "redraw c until satisfied (default: 10).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def word_spans(text: str) -> list[Any]:
    return list(_WORD_RE.finditer(text))


def prefix_at(text: str, c: float, min_length: int) -> str | None:
    """Keep the first round(c*N) words of text (N = word count).

    Returns the prefix, or None when it cannot satisfy the guard
    (round(c*N) < min_length or >= N), in which case the caller keeps the
    text whole.
    """
    spans = word_spans(text)
    n_words = len(spans)
    n = round(c * n_words)
    if n < min_length or n >= n_words:
        return None
    return text[: spans[n - 1].end()]


def truncate_once(text: str, rng: random.Random, min_length: int) -> tuple[str | None, float | None]:
    """Draw c until round(c*N) keeps between min_length and N-1 words.

    Returns (new_text, c), or (None, None) if the text is too short to ever
    satisfy the guard (then the caller keeps it whole).
    """
    n_words = len(word_spans(text))
    if n_words <= min_length:
        return None, None
    while True:
        c = rng.random()
        new_text = prefix_at(text, c, min_length)
        if new_text is not None:
            return new_text, c


def count_space_less(rows: list[dict[str, Any]]) -> int:
    """Rows whose response has no whitespace (kept whole by the truncator)."""
    n = 0
    for rec in rows:
        resp = rec.get("response")
        if resp and len(word_spans(resp)) < 2:
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    rows: list[dict[str, Any]] = []
    with args.input.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"loaded {len(rows)} rows from {args.input}", file=sys.stderr)

    # --- verification: does every response have spaces between words? ---
    n_space_less = count_space_less(rows)
    if n_space_less:
        print(f"note: {n_space_less} rows have a `response` with no whitespace "
              f"(e.g. some parseltongue transforms); they will be kept whole", file=sys.stderr)
    else:
        print("verified: every row with a response has spaces between words", file=sys.stderr)

    # --- selection: exactly --frac of the rows with a response ---
    sel_rng = random.Random(args.seed)
    resp_ix = [i for i, rec in enumerate(rows) if rec.get("response")]
    n_sel = int(round(args.frac * len(resp_ix)))
    selected = set(sel_rng.sample(resp_ix, n_sel))
    print(f"{len(resp_ix)} rows have a response; truncating {len(selected)} "
          f"({len(selected) / len(resp_ix):.1%}, min {args.min_length} words)",
          file=sys.stderr)

    n_truncated = 0
    n_too_short = 0
    n_already = 0
    for i in selected:
        rec = rows[i]
        # Never truncate twice: re-runs with --skip-combine operate on an
        # already-truncated file.
        if rec.get("response_truncated"):
            n_already += 1
            continue
        row_rng = random.Random(f"{args.seed}:{rec.get('row_id')}")
        new_resp, c = truncate_once(rec["response"], row_rng, args.min_length)
        if c is None:
            n_too_short += 1
            continue
        rec["response"] = new_resp
        rec["response_truncated"] = True
        rec["response_truncation_c"] = c
        rec["response_truncated_words"] = len(word_spans(new_resp))
        n_truncated += 1

    with args.output.open("w", encoding="utf-8") as fout:
        for rec in rows:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows -> {args.output}", file=sys.stderr)
    print(f"  truncated responses:       {n_truncated} "
          f"(skipped, response < min words: {n_too_short}; "
          f"already truncated: {n_already})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())