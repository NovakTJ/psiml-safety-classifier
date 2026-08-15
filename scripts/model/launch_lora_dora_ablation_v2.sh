#!/usr/bin/env bash
# Launcher for the Gemma v2 LoRA vs DoRA / attention-only vs all-linear
# ablation. Run directly, no env activation needed:
#   bash launch_lora_dora_ablation_v2.sh
#
# Starts scripts/model/run_lora_dora_ablation_v2.py detached (nohup + setsid)
# so it survives an SSH/browser-tab disconnect, refuses to start a second
# instance while one is already running, and prints exactly the commands
# needed to monitor or stop it.
#
# Unlike the earlier sweep launchers (launch_sweep_lora_v2_phase2.sh etc.),
# this one does NOT rely on `command -v python` / a pre-activated conda env
# -- it hardcodes the ccpp_env interpreter directly, so `conda activate` is
# never required before running this script.
#
# PID-lock contract (same fix as the Phase 2 / multiseed launchers):
#   Only run_lora_dora_ablation_v2.py's acquire_lock() creates, checks, and
#   removes lora_dora_ablation.pid. This launcher NEVER pre-writes a PID
#   into that file before starting python -- "nohup setsid python ..."
#   forks at least once (setsid(1) forks to avoid "Operation not permitted"
#   when it is already a process group leader), so $! here is the PID of an
#   intermediate wrapper process that exits almost immediately, NOT the real
#   python PID. Writing that wrapper PID into the lock file ourselves would
#   race against Python's own acquire_lock() and could leave a lock file
#   pointing at an already-dead PID. Instead: launch, wait briefly, then
#   read back the PID that Python itself wrote.
#
# No time budget: the underlying script loops until all three new runs
# (lora_attention_only, dora_all_linear, dora_attention_only) are completed
# or permanently failed_preflight/exhausted -- there is no
# --time-budget-minutes flag here, unlike the Phase 1/2/multiseed launchers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ABLATION_LAUNCHER_PYTHON_OVERRIDE exists only so this launcher's process
# lifecycle logic (start/detect-double-start/detect-immediate-death) can be
# tested with a stub interpreter, without a GPU or the real training script.
# Production behavior (override unset) is the required, hardcoded ccpp_env
# interpreter -- no `conda activate` needed.
PYTHON_BIN="${ABLATION_LAUNCHER_PYTHON_OVERRIDE:-/home/mls01/ccpp_env/bin/python}"
TRAIN_SCRIPT="$SCRIPT_DIR/run_lora_dora_ablation_v2.py"
OUTPUT_DIR="$SCRIPT_DIR/results/gemma_lora_v2_lora_dora_ablation"
LOCK_FILE="$OUTPUT_DIR/lora_dora_ablation.pid"
OUT_LOG="$OUTPUT_DIR/lora_dora_ablation.out"
ERR_LOG="$OUTPUT_DIR/lora_dora_ablation.err"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "[FATAL] Python interpreter ne postoji ili nije izvršan: $PYTHON_BIN" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[CHECK] Proveravam da $PYTHON_BIN može da importuje torch/transformers/peft/pandas/sklearn ..."
if ! "$PYTHON_BIN" -c "import torch, transformers, peft, pandas, sklearn" 2>/tmp/lora_dora_ablation_import_check.$$.log; then
    echo "[FATAL] Import provera nije prošla za $PYTHON_BIN:" >&2
    cat /tmp/lora_dora_ablation_import_check.$$.log >&2
    rm -f /tmp/lora_dora_ablation_import_check.$$.log
    exit 1
fi
rm -f /tmp/lora_dora_ablation_import_check.$$.log
echo "[OK] Sve potrebne biblioteke su dostupne u $PYTHON_BIN."

# Only refuse to start if a lock file exists AND its PID is confirmed alive
# via kill -0. We never create/touch the lock file ourselves -- only check
# it -- and a stale lock (kill -0 fails) is left for python's own
# acquire_lock() to clean up, so there is exactly one writer of this file.
if [ -f "$LOCK_FILE" ]; then
    EXISTING_PID="$(cat "$LOCK_FILE" 2>/dev/null || true)"
    if [ -n "$EXISTING_PID" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "[FATAL] Ablation je već aktivna (PID $EXISTING_PID, lock: $LOCK_FILE)." >&2
        echo "        Proveri: ps -p $EXISTING_PID -o pid,etime,cmd" >&2
        echo "        Ili prati postojeći log: tail -f $OUT_LOG" >&2
        exit 1
    fi
    echo "[INFO] Lock fajl $LOCK_FILE postoji ali PID $EXISTING_PID nije živ — python će ga sam ukloniti pri pokretanju."
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USER="${USER:-mls01}"
export LOGNAME="${LOGNAME:-mls01}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"

nohup setsid "$PYTHON_BIN" "$TRAIN_SCRIPT" \
    >> "$OUT_LOG" 2>> "$ERR_LOG" &
disown || true

# Give python a moment to run acquire_lock() (it does this before any
# dataset/model loading) and write its own real PID into
# lora_dora_ablation.pid. We do not compare against the lock file's previous
# content/mtime here: any lock that existed before this point was already
# required (above) to belong to a dead PID, so the first LIVE pid we observe
# now can only be the one this invocation's python process just wrote.
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

echo "LoRA/DoRA ablation pokrenuta."
echo "  PID (upisan od strane python skripte): $REAL_PID"
echo "  Output folder: $OUTPUT_DIR"
echo "  Redosled run-ova: lora_attention_only -> dora_all_linear -> dora_attention_only"
echo "  Nema vremenskog budžeta -- radi dok sva tri run-a ne budu completed ili failed_preflight/iscrpljena."
echo ""
echo "Provera da li proces radi:"
echo "  ps -p $REAL_PID -o pid,etime,%cpu,%mem,cmd"
echo ""
echo "Praćenje stdout log-a:"
echo "  tail -f $OUT_LOG"
echo ""
echo "Praćenje stderr log-a:"
echo "  tail -f $ERR_LOG"
echo ""
echo "Bezbedno zaustavljanje (SIGTERM, oslobađa lock preko python signal handlera;"
echo "trenutna konfiguracija ostaje 'running' na disku i biće bezbedno restartovana"
echo "od nule pri sledećem pokretanju):"
echo "  kill $REAL_PID"
echo ""
echo "Ponovno pokretanje nakon prekida (bilo namernog ili usled pada procesa):"
echo "  bash $SCRIPT_DIR/launch_lora_dora_ablation_v2.sh"
echo "  (completed run-ovi se preskaču, 'running' run se arhivira i restartuje od čistog"
echo "  baznog modela, failed_preflight se NE ponavlja automatski, failed se ponavlja do"
echo "  --max-attempts u run_lora_dora_ablation_v2.py, podrazumevano 2)."
