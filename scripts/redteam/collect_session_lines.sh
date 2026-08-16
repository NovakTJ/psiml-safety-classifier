#!/usr/bin/env bash
# Collect specific lines (plus surrounding context) from red-team session JSONL logs
# into a single text file.
#
# Usage: collect_session_lines.sh [OUT_FILE] [CONTEXT_LINES]
#   OUT_FILE       default: psiml_data/redteam_sessions/collected_lines.txt
#   CONTEXT_LINES  default: 1   (lines of context before/after each target line)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESS_DIR="$REPO_ROOT/psiml_data/redteam_sessions"
OUT="${1:-$SESS_DIR/collected_lines.txt}"
CTX="${2:-1}"

# "<filename> <line> [line ...]"
TARGETS=(
  "aegis_20260814T100337_2766.jsonl 2"
  "aegis_20260814T100933_3035.jsonl 2"
  "aegis_20260814T101233_3147.jsonl 2"
  "aegis_20260814T101814_3277.jsonl 2"
  "aegis_20260814T102337_3335.jsonl 2"
  "aegis_20260814T102756_3463.jsonl 2 3 4"
  "aegis_20260814T103319_3526.jsonl 2"
  "aegis_20260814T104228_3755.jsonl 2 4"
  "aegis_20260814T104823_3814.jsonl 2"
  "aegis_20260814T104912_4290.jsonl 2"
  "aegis_20260815T122128_40849.jsonl 3"
  "aegis_20260815T122448_41016.jsonl 2"
  "aegis_20260815T122559_41139.jsonl 2"
  "aegis_20260815T122613_41158.jsonl 2"
  "aegis_20260815T122628_41177.jsonl 2"
  "aegis_20260815T122643_41196.jsonl 2"
  "aegis_20260815T122711_41258.jsonl 2"
  "aegis_20260815T123221_41544.jsonl 2"
  "aegis_20260815T123524_41866.jsonl 2"
  "aegis_20260815T124054_41995.jsonl 2"
  "aegis_20260815T135605_43144.jsonl 2"
  "aegis_20260815T140609_43595.jsonl 2"
  "aegis_20260815T140651_43654.jsonl 2"
  "aegis_20260815T140714_43716.jsonl 2"
)

: > "$OUT"

for entry in "${TARGETS[@]}"; do
  read -r fname lines <<< "$entry"
  path="$SESS_DIR/$fname"

  {
    printf '===============================================================\n'
    printf 'FILE: %s\n' "$fname"
  } >> "$OUT"

  if [[ ! -f "$path" ]]; then
    printf 'MISSING FILE\n\n' >> "$OUT"
    continue
  fi

  for ln in $lines; do
    start=$(( ln - CTX )); (( start < 1 )) && start=1
    end=$(( ln + CTX ))
    {
      printf -- '--- target line %s (context %s-%s) ---\n' "$ln" "$start" "$end"
      awk -v s="$start" -v e="$end" -v t="$ln" \
        'NR>=s && NR<=e { printf "%s%d: %s\n", (NR==t ? ">>" : "  "), NR, $0 }
         NR>e { exit }' "$path"
      printf '\n'
    } >> "$OUT"
  done
done

printf 'Wrote %s (%s lines)\n' "$OUT" "$(wc -l < "$OUT")"
