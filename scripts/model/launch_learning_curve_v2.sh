#!/usr/bin/env bash
# Launcher for the learning-curve experiment (scripts/model/learning_curve_v2.py).
# No conda activation needed — the ccpp_env interpreter is hardcoded (same as
# launch_lora_dora_ablation_v2.sh). Run from anywhere:
#   bash launch_learning_curve_v2.sh
#
# Starts the script detached (nohup + setsid) so it survives a disconnect,
# refuses to start a second instance while one is running, and prints the
# commands needed to monitor or stop it.
#
# PID-lock contract (same fix as launch_sweep_lora_v2_phase2.sh):
#   Only learning_curve_v2.py's acquire_lock() creates/checks/removes
#   learning_curve.pid. This launcher NEVER pre-writes a PID — "nohup setsid
#   python ..." forks internally, so $! here is a short-lived wrapper, not
#   the real python PID. We launch, wait, then read back the PID python wrote.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# LC_LAUNCHER_PYTHON_OVERRIDE exists only so this launcher's process
# lifecycle logic can be tested with a stub interpreter (no GPU needed).
PYTHON_BIN="${LC_LAUNCHER_PYTHON_OVERRIDE:-/home/mls01/ccpp_env/bin/python}"
LC_SCRIPT="$SCRIPT_DIR/learning_curve_v2.py"
OUTPUT_DIR="$SCRIPT_DIR/results/learning_curve_v2"
LOCK_FILE="$OUTPUT_DIR/learning_curve.pid"
OUT_LOG="$OUTPUT_DIR/learning_curve.out"
ERR_LOG="$OUTPUT_DIR/learning_curve.err"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "[FATAL] Interpreter nije izvršiv: $PYTHON_BIN" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[CHECK] Proveravam da $PYTHON_BIN može da importuje torch/transformers/peft/pandas/sklearn ..."
if ! "$PYTHON_BIN" -c "import torch, transformers, peft, pandas, sklearn" 2>/tmp/lc_import_check.$$.log; then
    echo "[FATAL] Import provera nije prošla za $PYTHON_BIN:" >&2
    cat /tmp/lc_import_check.$$.log >&2
    rm -f /tmp/lc_import_check.$$.log
    exit 1
fi
rm -f /tmp/lc_import_check.$$.log
echo "[OK] Sve potrebne biblioteke su dostupne."

# Refuse to start only if a lock file exists AND its PID is alive. We never
# create/touch the lock file — a stale lock is left for python's own
# acquire_lock() to clean up (exactly one writer of this file).
if [ -f "$LOCK_FILE" ]; then
    EXISTING_PID="$(cat "$LOCK_FILE" 2>/dev/null || true)"
    if [ -n "$EXISTING_PID" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "[FATAL] Learning curve je već aktivan (PID $EXISTING_PID, lock: $LOCK_FILE)." >&2
        echo "        Proveri: ps -p $EXISTING_PID -o pid,etime,cmd" >&2
        echo "        Ili prati postojeći log: tail -f $OUT_LOG" >&2
        exit 1
    fi
    echo "[INFO] Lock fajl postoji ali PID $EXISTING_PID nije živ — python će ga sam ukloniti."
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USER="${USER:-mls01}"
export LOGNAME="${LOGNAME:-mls01}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"

nohup setsid "$PYTHON_BIN" "$LC_SCRIPT" \
    >> "$OUT_LOG" 2>> "$ERR_LOG" &
disown || true

# Wait for python's own acquire_lock() to write the real PID.
REAL_PID=""
for _ in $(seq 1 40); do
    sleep 0.5
    if [ -f "$LOCK_FILE" ]; then
        CANDIDATE="$(cat "$LOCK_FILE" 2>/dev/null || true)"
        if [ -n "$CANDIDATE" ] && kill -0 "$CANDIDATE" 2>/dev/null; then
            REAL_PID="$CANDIDATE"
            break
        fi
    fi
done

if [ -z "$REAL_PID" ]; then
    echo "[FATAL] Proces se ili ugasio odmah po pokretanju, ili nikad nije upisao $LOCK_FILE." >&2
    echo "        Proveri stderr log:" >&2
    tail -n 40 "$ERR_LOG" >&2 || true
    exit 1
fi

echo "Learning curve experiment pokrenut."
echo "  PID (upisan od strane python skripte): $REAL_PID"
echo "  Output folder: $OUTPUT_DIR"
echo ""
echo "Provera da li proces radi:"
echo "  ps -p $REAL_PID -o pid,etime,%cpu,%mem,cmd"
echo ""
echo "Praćenje logova:"
echo "  tail -f $OUT_LOG"
echo "  tail -f $ERR_LOG"
echo ""
echo "Bezbedno zaustavljanje (SIGTERM; aktivni run ostaje 'running' na disku i"
echo "biće bezbedno restartovan od nule pri sledećem pokretanju):"
echo "  kill $REAL_PID"
