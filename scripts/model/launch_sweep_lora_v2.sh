#!/usr/bin/env bash
# Launcher for the Gemma v2 LoRA Phase 1 sweep. Run from anywhere:
#   bash launch_sweep_lora_v2.sh
#
# Starts scripts/model/sweep_lora_v2.py detached (nohup + setsid) so it
# survives an SSH disconnect, refuses to start a second instance while one
# is already running, and prints exactly the commands needed to monitor or
# stop it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="$HOME/ccpp_env/bin/python"
SWEEP_SCRIPT="$SCRIPT_DIR/sweep_lora_v2.py"
OUTPUT_DIR="$SCRIPT_DIR/results/gemma_lora_v2_sweep"
LOCK_FILE="$OUTPUT_DIR/sweep.pid"
OUT_LOG="$OUTPUT_DIR/sweep.out"
ERR_LOG="$OUTPUT_DIR/sweep.err"
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-340}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "[FATAL] Python interpreter nije nađen ili nije izvršan: $PYTHON_BIN" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Isti PID-fajl proverava i sweep_lora_v2.py samostalno (odbrana u dubinu) —
# ova provera ovde sprečava čak i pokretanje python procesa.
if [ -f "$LOCK_FILE" ]; then
    EXISTING_PID="$(cat "$LOCK_FILE" 2>/dev/null || true)"
    if [ -n "$EXISTING_PID" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "[FATAL] Sweep je već aktivan (PID $EXISTING_PID, lock: $LOCK_FILE)." >&2
        echo "        Proveri: ps -p $EXISTING_PID -o pid,etime,cmd" >&2
        echo "        Ili prati postojeći log: tail -f $OUT_LOG" >&2
        exit 1
    fi
    echo "[WARN] Zastareo lock fajl (PID $EXISTING_PID nije živ) — uklanjam ga." >&2
    rm -f "$LOCK_FILE"
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USER="${USER:-mls01}"
export LOGNAME="${LOGNAME:-mls01}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"

nohup setsid "$PYTHON_BIN" "$SWEEP_SCRIPT" --time-budget-minutes "$TIME_BUDGET_MINUTES" \
    >> "$OUT_LOG" 2>> "$ERR_LOG" &
SWEEP_PID=$!

# sweep_lora_v2.py piše svoj sopstveni sweep.pid odmah po pokretanju
# (acquire_lock()); ovaj zapis ovde je dodatna, trenutna potvrda da je
# launcher zaista pokrenuo taj tačan proces (ista PID vrednost).
echo "$SWEEP_PID" > "$LOCK_FILE"

sleep 1
if ! kill -0 "$SWEEP_PID" 2>/dev/null; then
    echo "[FATAL] Proces se ugasio odmah po pokretanju — proveri $ERR_LOG" >&2
    tail -n 40 "$ERR_LOG" >&2 || true
    exit 1
fi

echo "Sweep pokrenut."
echo "  PID:              $SWEEP_PID"
echo "  Output folder:     $OUTPUT_DIR"
echo "  Vremenski budžet:  ${TIME_BUDGET_MINUTES} min"
echo ""
echo "Provera da li proces radi:"
echo "  ps -p $SWEEP_PID -o pid,etime,%cpu,%mem,cmd"
echo ""
echo "Praćenje stdout log-a:"
echo "  tail -f $OUT_LOG"
echo ""
echo "Praćenje stderr log-a:"
echo "  tail -f $ERR_LOG"
echo ""
echo "Bezbedno zaustavljanje (SIGTERM, oslobađa lock; trenutna konfiguracija"
echo "ostaje 'running' i biće bezbedno restartovana od nule pri sledećem pokretanju):"
echo "  kill $SWEEP_PID"
