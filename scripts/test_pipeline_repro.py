#!/usr/bin/env python
"""Reproducibility test: re-running the pipeline (--skip-translate) on the
checked-in artifacts must reproduce (approximately) data/complete_dataset.jsonl.

What it does:
    1. Copies the translation artifacts (translated.jsonl, sample.jsonl) into a
       temp dir.
    2. Runs the full pipeline: run_pipeline.py --skip-translate --data-dir <tmp>
       (filter harmful -> parseltongue obfuscate -> combine -> truncate).
    3. Compares <tmp>/complete_dataset.jsonl against the committed data/complete_dataset.jsonl:
       same set of row_ids, and field-by-field equality ignoring volatile
       metadata (timestamps).

The combine step re-samples wildguardmix deterministically (seed=42) via the
HuggingFace cache, so it works offline. The obfuscation step shells out to Node
(deterministic mechanical transforms).

Usage:
    .venv/bin/python scripts/test_pipeline_repro.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Fields that legitimately differ between runs.
VOLATILE_FIELDS = {"timestamp"}

# The zalgo transform injects a random number of random combining marks per
# character (Math.random in the Node transform), so its textual output is
# non-deterministic by design. For zalgo rows we compare text fields after
# stripping exactly the mark set the transform can add.
ZALGO_MARKS = set("\u0300\u0301\u0302\u0303\u0304\u0305\u0306\u0307\u0308"
                  "\u0309\u030a\u030b\u030c\u030d\u030e\u030f\u0310\u0311"
                  "\u0312\u0313\u0314\u0315\u031a\u031b\u033d\u033e\u033f")
TEXT_FIELDS = {"prompt", "response"}


def strip_zalgo_marks(text: object) -> object:
    if not isinstance(text, str):
        return text
    return "".join(c for c in text if c not in ZALGO_MARKS)


def run_pipeline(tmp: Path) -> None:
    env = dict(os.environ)
    # Use the HF cache only; never hit the network in a test.
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    cmd = [
        sys.executable, "scripts/run_pipeline.py",
        "--skip-translate",
        "--data-dir", str(tmp),
    ]
    print(f">> {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"pipeline failed with exit code {proc.returncode}")


def load_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec.get("row_id")
            if rid in rows:
                raise AssertionError(f"duplicate row_id {rid!r} in {path}")
            rows[rid] = rec
    return rows


def strip_volatile(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k not in VOLATILE_FIELDS}


def main() -> int:
    for artifact in ("translated.jsonl", "sample.jsonl", "complete_dataset.jsonl"):
        if not (DATA_DIR / artifact).exists():
            raise SystemExit(f"missing required artifact: data/{artifact}")

    with tempfile.TemporaryDirectory(prefix="pipeline-repro-") as tmp_str:
        tmp = Path(tmp_str)
        for artifact in ("translated.jsonl", "sample.jsonl"):
            shutil.copy(DATA_DIR / artifact, tmp / artifact)

        run_pipeline(tmp)

        expected = load_rows(DATA_DIR / "complete_dataset.jsonl")
        actual = load_rows(tmp / "complete_dataset.jsonl")

    # --- row-set comparison -------------------------------------------------
    missing = set(expected) - set(actual)
    extra = set(actual) - set(expected)
    if missing or extra:
        print(f"FAIL: row_id sets differ: {len(missing)} missing, {len(extra)} extra",
              file=sys.stderr)
        for rid in sorted(missing)[:5]:
            print(f"  missing: {rid}", file=sys.stderr)
        for rid in sorted(extra)[:5]:
            print(f"  extra:   {rid}", file=sys.stderr)
        return 1

    # --- field-by-field comparison -------------------------------------------
    diffs: list[tuple[str, str, object, object]] = []
    for rid in expected:
        e, a = strip_volatile(expected[rid]), strip_volatile(actual[rid])
        zalgo = expected[rid].get("encoding_type") == "zalgo"
        keys = set(e) | set(a)
        for k in sorted(keys):
            ev, av = e.get(k), a.get(k)
            if zalgo and k in TEXT_FIELDS:
                ev, av = strip_zalgo_marks(ev), strip_zalgo_marks(av)
            if ev != av:
                diffs.append((rid, k, e.get(k), a.get(k)))

    n_rows = len(expected)
    if diffs:
        print(f"FAIL: {len(diffs)} field mismatches across {n_rows} rows:", file=sys.stderr)
        for rid, k, ev, av in diffs[:10]:
            print(f"  {rid} . {k}:\n    expected: {str(ev)[:120]!r}\n    actual:   {str(av)[:120]!r}",
                  file=sys.stderr)
        return 1

    print(f"OK: regenerated dataset matches data/complete_dataset.jsonl exactly "
          f"({n_rows} rows, ignoring {sorted(VOLATILE_FIELDS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
