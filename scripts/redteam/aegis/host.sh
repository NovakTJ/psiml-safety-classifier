#!/usr/bin/env bash
#
# host.sh — one-command public host for the AEGIS guarded-chat web UI.
#
# Starts the guard server (Gemma-3-1B+LoRA on CPU) bound to localhost, opens a
# Cloudflare quick tunnel, and prints ONE public https:// link (with the access
# token baked in) for a remote friend to paste into any browser. On any failure
# it prints the relevant log and a clear error instead.
#
#   ./host.sh                       # host until Ctrl-C (tears down server + tunnel)
#   ./host.sh --adapter none        # bare zero-shot Gemma guard (no LoRA)
#   ./host.sh --threshold 0.3 --model openai/gpt-3.5-turbo-0613
#
# Any args after the script name are forwarded verbatim to web.py, so every
# web.py knob works here: --adapter none | --guard-prompt @file | --model ID |
# --threshold F | --check-every N | --temperature F | --max-tokens N | --thinking on.
# (Do NOT pass --host/--port/--token here — this script owns those; use the env
# vars below instead.)  See:  ~/aegis_env/bin/python web.py --help
#
# Overridable via env vars (all optional):
#   AEGIS_PORT   local port to bind          (default 8321)
#   AEGIS_TOKEN  access token in the URL      (default: fresh random each run)
#   AEGIS_PY     venv python                  (default: ~/aegis_env/bin/python)
#   AEGIS_MODEL  OpenRouter target model      (shortcut for --model)
#
set -euo pipefail

# ---- config -----------------------------------------------------------------
REPO="/home/novaktj/psiml-safety-classifier"
AEGIS_DIR="$REPO/scripts/redteam/aegis"
PY="${AEGIS_PY:-$HOME/aegis_env/bin/python}"
PORT="${AEGIS_PORT:-8321}"
CF="$HOME/.cache/aegis/cloudflared"
CF_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
TOKEN="${AEGIS_TOKEN:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(9))')}"
LOG_DIR="$(mktemp -d /tmp/aegis-host.XXXXXX)"
WEB_LOG="$LOG_DIR/web.log"
TUN_LOG="$LOG_DIR/tunnel.log"

WEB_PID=""; TUN_PID=""

die() { echo; echo "ERROR: $*" >&2; exit 1; }

cleanup() {
  set +e
  # polite stop first
  [ -n "$TUN_PID" ] && kill "$TUN_PID" 2>/dev/null
  [ -n "$WEB_PID" ] && kill "$WEB_PID" 2>/dev/null
  # give graceful shutdown a moment, then force-kill any survivor so the
  # public link can never outlive this script
  for _ in 1 2 3 4 5 6; do
    kill -0 "$TUN_PID" 2>/dev/null || kill -0 "$WEB_PID" 2>/dev/null || break
    sleep 0.5
  done
  [ -n "$TUN_PID" ] && kill -9 "$TUN_PID" 2>/dev/null
  [ -n "$WEB_PID" ] && kill -9 "$WEB_PID" 2>/dev/null
  wait 2>/dev/null
  rm -rf "$LOG_DIR" 2>/dev/null
}
trap cleanup EXIT INT TERM

# ---- preflight --------------------------------------------------------------
[ -x "$PY" ]           || die "guard venv python not found at '$PY' (set AEGIS_PY=/path/to/python)."
[ -f "$AEGIS_DIR/web.py" ] || die "web.py not found under $AEGIS_DIR — is the repo path right?"

if [ -z "${OPENROUTER_API_KEY:-}" ] && ! grep -q '^OPENROUTER_API_KEY=' "$REPO/.env" 2>/dev/null; then
  die "OPENROUTER_API_KEY not set (needs to be in $REPO/.env or the environment)."
fi

if ss -tlnp 2>/dev/null | grep -qE ":$PORT([[:space:]]|\$)"; then
  die "port $PORT is already in use — a server may already be running. Use AEGIS_PORT=NNNN to pick another, or stop the old one."
fi

# ---- ensure cloudflared -----------------------------------------------------
if [ ! -x "$CF" ]; then
  echo "cloudflared not found — downloading (~40 MB, one time) ..."
  mkdir -p "$(dirname "$CF")"
  curl -fsSL -o "$CF" "$CF_URL" || die "failed to download cloudflared from $CF_URL (check internet)."
  chmod +x "$CF"
fi

# ---- start guard server -----------------------------------------------------
echo "[1/2] starting guard server on 127.0.0.1:$PORT (loading model — ~1 min on CPU) ..."
( cd "$AEGIS_DIR" && exec "$PY" web.py \
    --host 127.0.0.1 --port "$PORT" --token "$TOKEN" \
    ${AEGIS_MODEL:+--model "$AEGIS_MODEL"} "$@" ) >"$WEB_LOG" 2>&1 &
WEB_PID=$!

ready=""
for _ in $(seq 1 120); do
  if ! kill -0 "$WEB_PID" 2>/dev/null; then
    echo "----- server log -----"; cat "$WEB_LOG"; echo "----------------------"
    die "guard server exited during startup (see log above)."
  fi
  if ss -tlnp 2>/dev/null | grep -qE ":$PORT([[:space:]]|\$)"; then ready=1; break; fi
  sleep 2
done
[ -n "$ready" ] || { echo "----- server log -----"; cat "$WEB_LOG"; die "server did not come up within ~4 min."; }

# ---- start tunnel -----------------------------------------------------------
echo "[2/2] opening public tunnel ..."
"$CF" tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate >"$TUN_LOG" 2>&1 &
TUN_PID=$!

URL=""
for _ in $(seq 1 40); do
  if ! kill -0 "$TUN_PID" 2>/dev/null; then
    echo "----- tunnel log -----"; cat "$TUN_LOG"; echo "----------------------"
    die "cloudflared exited before giving a URL (see log above)."
  fi
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUN_LOG" | head -1 || true)"
  [ -n "$URL" ] && break
  sleep 2
done
[ -n "$URL" ] || { echo "----- tunnel log -----"; cat "$TUN_LOG"; die "no tunnel URL appeared in time."; }

FULL="$URL/?token=$TOKEN"

# ---- banner -----------------------------------------------------------------
echo
echo "=================================================================="
echo "  AEGIS is LIVE — send your friend this exact link:"
echo
echo "    $FULL"
echo
echo "  Keep this laptop awake and this terminal open — closing it"
echo "  (or Ctrl-C) tears down the server and the link dies."
echo "=================================================================="
echo

# Keep running; Ctrl-C -> trap cleanup. Exit if either child dies.
wait -n "$WEB_PID" "$TUN_PID"
die "server or tunnel stopped unexpectedly — see the messages above."
