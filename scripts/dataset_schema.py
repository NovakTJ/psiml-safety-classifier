"""Canonical column schema shared by every JSONL artifact the pipeline writes.

Rule: every row in a dataset carries *every* column.  Where a column does not
apply to a row type -- ``translation_model`` on a raw English row, or
``response_truncation_c`` on a row whose response was never truncated -- the
row keeps the column's empty value (``None`` / ``""`` / ``False``) instead of
dropping the key, so downstream loaders can use ``row[col]`` without
``KeyError`` and without a chain of ``row.get(...)`` fallbacks.

Two schemas:

* :data:`FINAL_COLUMNS` -- schema of ``data/complete_dataset.jsonl`` (the training
  file).  The parseltongue debug columns (``__original_prompt__``,
  ``__original_response__``, ``__transform__``) are intentionally *not* part
  of it: they are only useful while building the intermediate pools (the
  combine step reads ``__original_prompt__`` to recover the original row
  index), so they are dropped at combine time.
* :data:`INTERMEDIATE_COLUMNS`` -- ``FINAL_COLUMNS`` + the debug columns;
  used by the intermediate pool files (``train_weird_*.jsonl``) that feed
  ``combine_dataset.py``.

Usage::

    from dataset_schema import FINAL_COLUMNS, INTERMEDIATE_COLUMNS, normalize

    out_row = normalize(row, FINAL_COLUMNS)          # fill empties, drop debug
    pool_row = normalize(row, INTERMEDIATE_COLUMNS)  # fill empties, keep debug
"""

from __future__ import annotations

from typing import Any

# Every row of every final dataset must carry these keys, in this order.
FINAL_COLUMNS: list[str] = [
    "adversarial",
    "augmentation_pipeline_version",
    "augmentation_type",
    "encoding_type",
    "language",
    "notes",
    "original_idx",
    "prompt",
    "prompt_harm_label",
    "prompt_template_version",
    "response",
    "response_harm_label",
    "response_refusal_label",
    "response_truncated",
    "response_truncation_c",
    "response_truncated_words",
    "row_id",
    "source_split",
    "subcategory",
    "timestamp",
    "translation_model",
    "verified_accurate_description",
]

# Empty value per column, used when the column does not apply to a row.
# Types stay stable per column across the whole dataset (e.g. the truncation
# columns are always bool / float-or-None / int-or-None).
EMPTY: dict[str, Any] = {
    "adversarial": False,
    "augmentation_pipeline_version": "",
    "augmentation_type": "",
    "encoding_type": "",
    "language": "",
    "notes": "",
    "original_idx": None,
    "prompt": "",
    "prompt_harm_label": "",
    "prompt_template_version": "",
    "response": "",
    "response_harm_label": "",
    "response_refusal_label": None,
    "response_truncated": False,
    "response_truncation_c": None,
    "response_truncated_words": None,
    "row_id": "",
    "source_split": "",
    "subcategory": "",
    "timestamp": "",
    "translation_model": None,
    "verified_accurate_description": False,
}

# Debug columns written by augment_with_parseltongue.py.  combine_dataset.py
# needs __original_prompt__ to recover the original row index of English
# rows, but the columns are dropped from the final training file.
DEBUG_COLUMNS: list[str] = [
    "__original_prompt__",
    "__original_response__",
    "__transform__",
]

INTERMEDIATE_COLUMNS: list[str] = FINAL_COLUMNS + DEBUG_COLUMNS


def normalize(row: dict[str, Any], columns: list[str] | None = None) -> dict[str, Any]:
    """Return a copy of ``row`` with every schema column present.

    Missing columns are filled with their empty value from :data:`EMPTY`; any
    extra columns (e.g. the debug columns when ``columns=FINAL_COLUMNS``) are
    dropped.  Values that are already present are never overwritten.
    """
    columns = list(columns) if columns is not None else list(FINAL_COLUMNS)
    out: dict[str, Any] = {}
    for col in columns:
        out[col] = row[col] if col in row else EMPTY.get(col)
    return out