# aegis-chat — usage

Guarded chat with Qwen3.5-9B (OpenRouter) watched by the Gemma-3-1B LoRA
exchange classifier, with prompt pre-screening and mid-stream blocking.
Spec: `PLAN.md`. Engine: `core.py`.

## Setup (off-cluster: laptop / GCP VM)

On the **cluster** everything just works (`ccpp_env`, `/data/models`, adapter in
`scripts/model/results/`). Off-cluster:

```bash
python3 -m venv ~/aegis_env
~/aegis_env/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/aegis_env/bin/pip install "transformers==4.57.6" peft accelerate
~/aegis_env/bin/pip install fastapi "uvicorn[standard]"   # web.py only

# base model: google/gemma-3-1b-it is license-gated; the unsloth mirror is
# identical and ungated — core.py uses it automatically when /data/models
# is absent (override with AEGIS_BASE_MODEL).
~/aegis_env/bin/hf download unsloth/gemma-3-1b-it

# adapter (25 MB): copy from the cluster into this exact repo-relative path
# (the dir is gitignored, so it must be synced out-of-band):
#   scripts/model/results/gemma_lora_v2_sweep_phase2/lr3e-4_r8_alpha16_dropout0.0_seed42/best_adapter/

# OpenRouter key: put OPENROUTER_API_KEY=... in the repo-root .env
# (cli.py loads it) or export it.
```

## Run

```bash
# human REPL
~/aegis_env/bin/python scripts/redteam/aegis/cli.py --device cpu

# machine mode (one user message per stdin line, JSON events on stdout)
echo "hi" | ~/aegis_env/bin/python scripts/redteam/aegis/cli.py --jsonl --device cpu
```

REPL slash commands: `/reset /thinking on|off /threshold F /check-every N
/stats /config /help /quit`.

Key flags (full list: `--help`): `--adapter PATH|none` (none = bare zero-shot
Gemma), `--guard-prompt '...'|@file`, `--model OPENROUTER_ID` (default
`qwen/qwen3.5-9b`), `--threshold`, `--check-every`, `--thinking on|off`,
`--no-precheck` (disable the token-0 prompt pre-check — ablation knob),
`--log-dir`. See PLAN.md "Expected-to-change pieces" for why these are knobs.

## Web server (`web.py`, frontend 2 — the shareable demo)

Teammates get a URL, nothing else; the server holds the model, adapter and
OpenRouter key. One process serves the chat page AND the WebSocket endpoint
(same origin, no CORS).

### One-command public host (`host.sh`)

For a remote friend/demo off a laptop, `host.sh` does the whole thing: random
token, starts `web.py` on localhost, opens a Cloudflare quick tunnel (downloads
`cloudflared` to `~/.cache/aegis/` on first run), and prints ONE public
`https://…trycloudflare.com/?token=…` link — or a clear error + log if anything
fails. Runs in the foreground; **Ctrl-C tears down both** the server and the
tunnel (the link dies with it). The tunnel URL is fresh every run.

```bash
scripts/redteam/aegis/host.sh                    # default guard, print a link
scripts/redteam/aegis/host.sh --adapter none     # any web.py flag forwards through
AEGIS_PORT=8399 scripts/redteam/aegis/host.sh     # env: AEGIS_PORT/AEGIS_TOKEN/AEGIS_PY/AEGIS_MODEL
```

Args after the script forward verbatim to `web.py`; `--host/--port/--token` are
owned by the script (set port/token via the env vars). Needs `OPENROUTER_API_KEY`
in the repo `.env`.

```bash
# localhost demo
~/aegis_env/bin/python scripts/redteam/aegis/web.py --device cpu

# LAN / GCP VM demo — token REQUIRED when binding beyond localhost
# (the server proxies paid OpenRouter traffic and logs harmful text)
~/aegis_env/bin/python scripts/redteam/aegis/web.py --device cpu \
    --host 0.0.0.0 --port 8321 --token 'shared-secret'
# red-teamers open http://<host>:8321/ and enter the token when prompted
```

All `cli.py` engine flags work identically (`--adapter none`, `--model`,
`--threshold`, `--check-every`, `--thinking`, …). Architecture: ONE shared
`GemmaGuard` loaded at startup, and **one conversation at a time** — the
connection holding the guard ("the seat") gets its own `GuardedSession` (own
history + own session log, `aegis_*_wNNN.jsonl`); everyone else waits in a FIFO
queue, sees their position in the page's queue panel, and cannot send anything
until the seat is granted. The guard is reset at handover so an occupant never
inherits the previous one's conversation.

**Why single occupancy is required, not just polite:** the guard's conversation
state (KV cache, `_committed_ids`, history) lives on the `GemmaGuard`, not on
the `GuardedSession`. Two live sessions on one shared guard would (a) classify
each other's messages with the other's conversation prepended, and (b) wipe each
other's guard history on any block/reset while the target-model history
survived, breaking the guard/target lockstep that reset-on-block enforces. A
per-session guard is the alternative, but costs a full model copy (~2 GB) per
user — not viable on the 7 GB laptop / small GCP VM this runs on.

Protocol (`/ws/chat`): client sends `{"type":"user","text":...}` /
`{"type":"reset"}`; server streams the core.py event schema
(`seat`/`ready`/`token`/`reasoning`/`check`/`verdict`/`error`/`reset_ok`).
`{"type":"seat","state":"queued","position":N,"waiting":M}` arrives on connect
if someone else holds the guard and is re-sent whenever the line moves;
`{"type":"seat","state":"active"}` (followed by `ready`) means the chat is
yours. Anything sent while queued gets an `error` back. The page
renders tokens live, a P(harmful) sparkline per response, and the block
banner — which, per PLAN.md "Expected-to-change" #4, tells the red-teamer
that a blocked BENIGN request is a false positive (guard failure), not a win.

**A block ends the conversation.** When the guard fires, `core.py` clears the
session history and the verdict carries `conversation_reset: true`; the page
draws a "— conversation reset —" divider and the next message starts a fresh
conversation. You cannot continue a blocked conversation from any frontend.
Reason: a blocked turn's harmful content must not survive in EITHER the target's
or the guard's context for a follow-up turn to lean on — `reset()` clears both.

**The guard is history-aware.** Its KV cache is persistent and append-only across
turns, so each classification reflects the whole conversation (interleaved
`USER PROMPT:`/`ASSISTANT RESPONSE:` per turn), letting it catch multi-turn
escalation where each turn is individually benign (the "$x example"). Verified
equal to a full recompute on the real model; the 512-token sliding-window limit
on very early context is a known caveat. See PLAN.md "Design: history-aware guard".

Session logs: one JSONL per session in `psiml_data/redteam_sessions/`
(config header incl. target model + guard prompt, per-turn checks with
latencies, verdicts; the REAL blocked partial is preserved in the log while the
model-visible history gets a block notice). These are red-team evidence and v3
dataset raw material — keep them.

## CPU performance (this laptop, WSL2, 8 cores)

Guard check ≈ 3–5 s (grows slightly with exchange length past the 512-token
sliding window). The A100 does this in single-digit ms. For a snappier local
demo, raise `--check-every` (e.g. 150) at the cost of later blocks.

## Tests

```bash
PY=~/aegis_env/bin/python   # or /home/mls01/ccpp_env/bin/python on the cluster
$PY scripts/redteam/aegis/tests/test_core_logic.py         # no model/network
$PY scripts/redteam/aegis/tests/test_text_construction.py  # no model/network
$PY scripts/redteam/aegis/tests/test_web.py                # web protocol+auth, stub session, no model/network
$PY scripts/redteam/aegis/tests/test_guard_live.py --device cpu [--kv-only|--logit-only] [--dtype fp32]
$PY scripts/redteam/aegis/tests/debug_kv.py                # fp32 exactness proof, crosses the 512 window
```

- `--kv-only` = incremental-vs-full scoring check. bf16 tolerance is 1e-2
  (chunked-vs-full matmul noise, worst 5.7e-3 seen short / 1.8e-2 past the
  512-token window); `--dtype fp32` proves the logic exact (~1e-6).
- `--logit-only` needs the sweep's `validation_predictions_epoch_6.csv`, which
  is cluster-only (gitignored results dir) — run it there.

## Implementation notes (why it looks like this)

- **Token-0 pre-check:** every turn is first classified with an EMPTY
  response, before the target model is ever called (one guard forward over the
  primed prefix + suffix — the KV cache is already warm from `begin_turn`). A
  harmful prompt blocks here: no OpenRouter call, no generated tokens, no API
  spend. Verified live: pipe-bomb prompt → p≈1.0 at token 0; soup prompt →
  p≈2.5e-8, turn proceeds. Mid-stream checks still matter for response-side
  harm that only appears once generation starts (the "$x example"). The
  training rule covers this case ("bad prompt + no response ⇒ HARMFUL"), and
  on v2 data `final_label == prompt_harm_label` on every row, so the guard is
  largely a prompt classifier anyway — the pre-check is where most of its
  blocking power lands. `--no-precheck` disables it for ablations.
- **Score = softmax over first-token logits** (`harm` vs `un`) — identical
  decision to the sweep's greedy generation, thresholdable from one forward.
- **KV reuse:** the real cache holds exactly prefix + response-so-far; each
  check forwards only the BPE-realized delta. The 5 template-suffix tokens are
  forwarded over a **throwaway deep copy** of the cache — never the real one —
  because gemma-3-1b's `DynamicSlidingWindowLayer` refuses to `crop()` once the
  exchange is longer than its 512-token window (the original crop-back scheme
  crashed mid-stream there, 2026-08-13). The rare BPE-realign crop falls back
  to a one-shot re-prime forward if it hits that limit.
- **Blocking:** stream aborted, partial withheld from model history (attacker
  can't read the leak and adapt), fixed block notice substituted; real partial
  in the session log only.
- Blocked-benign turns are false-positive evidence (PLAN.md
  "Expected-to-change" #4) — the log keeps `p_harmful` + token offset for them.
