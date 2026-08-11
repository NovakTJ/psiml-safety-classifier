#!/usr/bin/env python
"""Augment a text dataset with "weird character" transforms from P4RS3LT0NGV3.

This script calls the P4RS3LT0NGV3 transforms through a small Node.js bridge.
For each input example it produces one variant per selected transform, which is
useful for training/adversarially evaluating a safety classifier on obfuscated
or stylized text. Every selected column (default: ``prompt`` AND ``response``)
gets the SAME transform applied.

Prerequisites
-------------
- Node.js installed and on PATH.
- The P4RS3LT0NGV3 repo cloned at vendor/P4RS3LT0NGV3 (already done).

Usage
-----
    # See all 222 available transforms
    .venv/bin/python scripts/augment_with_parseltongue.py --list-transforms

    # Augment a JSONL file: obfuscates the "prompt" AND "response" columns
    .venv/bin/python scripts/augment_with_parseltongue.py \
        --input data/harmful_en.jsonl \
        --output data/train_weird_en.jsonl

    # Use only specific transforms
    .venv/bin/python scripts/augment_with_parseltongue.py \
        --transforms leetspeak,zalgo,circled,bold,upside_down \
        --input data/harmful_en.jsonl \
        --output data/train_weird_en.jsonl

    # Restrict which columns are transformed
    .venv/bin/python scripts/augment_with_parseltongue.py \
        --fields prompt,response --input data/train.jsonl --output data/weird.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TRANSFORMS = [
    # Leet / visual substitutions
    "leetspeak",
    # Unicode “fancy text” blocks
    "bold",
    "italic",
    "bold_italic",
    "circled",
    "squared",
    "negative_squared",
    "parenthesized",
    "small_caps",
    "cursive",
    "fraktur",
    "doubleStruck",
    "monospace",
    "fullwidth",
    "vaporwave",
    "wide_spacing",
    "mathematical",
    # Decorated / glitchy
    "zalgo",
    "strikethrough",
    "underline",
    "dashed_underline",
    "dotted_underline",
    "wavy_underline",
    "overline",
    # Mirrored / rotated
    "upside_down",
    "mirror",
    # Script / symbol alphabets
    "greek",
    "cyrillic_stylized",
    "regional_indicator",
    "braille",
    "wingdings",
]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def bridge_path() -> Path:
    return project_root() / "scripts" / "_parseltongue_node_bridge.js"


def run_node_bridge(payload: dict[str, Any]) -> dict[str, Any]:
    if not bridge_path().exists():
        raise FileNotFoundError(f"Node bridge not found: {bridge_path()}")

    proc = subprocess.run(
        ["node", str(bridge_path())],
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        capture_output=True,
        cwd=project_root(),
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"Node bridge failed: {proc.stderr.strip()}")

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Node bridge returned invalid JSON: {exc}") from exc


def list_transforms() -> list[str]:
    # The bridge supports a list command via an empty examples payload.
    all_transforms = run_node_bridge({"transforms": [], "examples": []})
    # If we want a real list we have to load them; fall back to reading the repo.
    repo = project_root() / "vendor" / "P4RS3LT0NGV3"
    transforms_dir = repo / "src" / "transformers"
    keys: list[str] = []
    for category_dir in sorted(transforms_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        for file in sorted(category_dir.glob("*.js")):
            if file.name in {"BaseTransformer.js", "index.js", "loader-node.js"}:
                continue
            key = file.stem.replace("-", "_")
            keys.append(key)
    return sorted(keys)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path | None, records: list[dict[str, Any]]) -> None:
    out = sys.stdout if path is None else path.open("w", encoding="utf-8")
    try:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if path is not None:
            out.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment text with P4RS3LT0NGV3 weird-character transforms."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input JSONL file (stdin if omitted).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file (stdout if omitted).",
    )
    parser.add_argument(
        "--fields",
        default="prompt,response",
        help='Comma-separated columns to transform; each gets the same transform '
             '(default: "prompt,response"). Columns missing from a record are '
             'treated as empty strings.',
    )
    parser.add_argument(
        "--id-col",
        default=None,
        help='Column to use as example id. If omitted, rows are numbered.',
    )
    parser.add_argument(
        "--transforms",
        default=",".join(DEFAULT_TRANSFORMS),
        help="Comma-separated list of transform keys (default: a curated weird-character set).",
    )
    parser.add_argument(
        "--list-transforms",
        action="store_true",
        help="Print all available transform keys and exit.",
    )
    parser.add_argument(
        "--force-harmful",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set prompt_harm_label='harmful' on every output row (default: on).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of examples to send to Node per batch (default: 100).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_transforms:
        for key in list_transforms():
            print(key)
        return 0

    transform_keys = [k.strip() for k in args.transforms.split(",") if k.strip()]
    if not transform_keys:
        print("error: no transforms specified", file=sys.stderr)
        return 1

    if args.input is None:
        source = sys.stdin
    else:
        source = args.input.open("r", encoding="utf-8")

    records: list[dict[str, Any]] = []
    try:
        for line in source:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    finally:
        if args.input is not None:
            source.close()

    if not records:
        print("warning: no input records", file=sys.stderr)
        return 0

    field_names = [f.strip() for f in args.fields.split(",") if f.strip()]
    if not field_names:
        print("error: no fields specified", file=sys.stderr)
        return 1

    # Prepare examples with stable IDs; every field is sent to the bridge and
    # transformed with the same transform.
    examples = []
    for idx, rec in enumerate(records):
        ex_id = rec.get(args.id_col) if args.id_col else idx
        fields = {name: rec.get(name, "") for name in field_names}
        examples.append({"id": ex_id, "fields": fields})

    # Process in batches to keep Node payloads reasonable.
    out_records: list[dict[str, Any]] = []
    errors_seen: list[str] = []

    for start in range(0, len(examples), args.batch_size):
        batch = examples[start : start + args.batch_size]
        payload = {"transforms": transform_keys, "examples": batch}
        result = run_node_bridge(payload)

        # Map id -> record index. Records are matched positionally via this map
        # (the old batch.index(...) scan was O(n^2) and matched on dict equality,
        # which silently aliased duplicate rows). Duplicate ids within a batch
        # fall back to the first occurrence, matching the bridge's behaviour.
        id_to_pos: dict[Any, int] = {}
        for pos, ex in enumerate(batch):
            id_to_pos.setdefault(ex["id"], pos)

        for err in result.get("errors", []):
            msg = f"transform={err.get('transform')} id={err.get('id')} error={err.get('message')}"
            if msg not in errors_seen:
                errors_seen.append(msg)
                print(f"warning: {msg}", file=sys.stderr)

        for item in result.get("results", []):
            pos = id_to_pos.get(item["id"])
            if pos is None:
                print(f"warning: bridge returned unknown id {item['id']!r}; skipping", file=sys.stderr)
                continue
            base = records[start + pos]
            outputs = item.get("outputs") or {}
            augmented = dict(base)
            augmented["__original_prompt__"] = base.get("prompt", "")
            augmented["__original_response__"] = base.get("response", "")
            augmented["__transform__"] = item["transform"]
            for name in field_names:
                augmented[name] = outputs.get(name, "")
            # --- metadata columns (translation file passes most through) ---
            augmented["encoding_type"] = item["transform"]
            # Faithful by construction (mechanical transform), BUT inherit the
            # verification status of the source row: a transform of an unverified
            # translation is still unverified content.
            augmented["verified_accurate_description"] = base.get("verified_accurate_description", True)
            augmented["augmentation_pipeline_version"] = "v1"
            augmented["timestamp"] = datetime.now(timezone.utc).isoformat()
            if args.force_harmful:
                augmented["prompt_harm_label"] = "harmful"
            base_notes = base.get("notes") or ""
            augmented["notes"] = ";".join(filter(None, [base_notes, "parseltongue"]))
            out_records.append(augmented)

    write_jsonl(args.output, out_records)

    if args.output:
        print(
            f"Wrote {len(out_records)} augmented records to {args.output}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
