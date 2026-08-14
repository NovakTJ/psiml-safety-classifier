# AEGIS — guarded-chat interface for red-teaming the Gemma classifier

Status: **FRONTENDS DONE** (spec 2026-08-13; `core.py` + `tests/` + `cli.py`
implemented same day; `web.py` + `tests/test_web.py` + laptop-CPU live smoke
done 2026-08-13 evening — see verification checklist item 6). Serving architecture decided 2026-08-13: **the web server is how
red-teamers get access (a URL — no repo clones, no local deps), hosted off-cluster**
(laptop for ad-hoc, GCP VM for the stable shared instance), NOT in this container —
see "Hosting reality" under Frontend 2.

AEGIS is the interface layer for red-teaming our exchange classifier in a realistic
deployment shape: a human (or an agent) has a normal multi-turn conversation with
**Qwen3.5-9B** (via OpenRouter), while the **Gemma-3-1B LoRA classifier** watches the
response *as it streams* and blocks mid-generation when it fires. This is the demo-able
version of the project's end goal (mid-stream stopping), and every session logs raw
material for the v3 multi-turn dataset.

## Why this exists / context

- The classifier was trained on single (prompt, response) exchanges
  (`data/gemma_v2_no_refusal/`); multi-turn generalization is an open empirical
  question — see **"the $x example"** in `CLAUDE.md`. AEGIS sessions are how we find out.
- A separate augmentation effort (not this spec) is creating benign-prompt +
  harmful-response training pairs; AEGIS logs supplement it with real attack traces.
- GPU constraint at time of writing: teammate is finetuning on the A100, so the
  **target model is served by OpenRouter** and only the 1B guard runs locally
  (~3 GB bf16; ~19.6 GB was free alongside the teammate's run when spec'd).
  Side benefit: this makes the whole stack **portable off-cluster** (1B guard +
  API target), which is what enables the laptop-hosted web demo below.

## Layout

```
scripts/redteam/aegis/
├── PLAN.md    # this file
├── core.py    # the engine — no terminal/HTTP assumptions
├── cli.py     # frontend 1: "aegis-chat"  (human REPL + machine-readable stdio mode)
├── web.py     # frontend 2: FastAPI+WS web server — the shareable demo (off-cluster host)
└── README.md  # usage (written with the implementation)
psiml_data/redteam_sessions/   # session logs (one JSONL per session; gitignored if large)
```

Build order: `core.py` → `cli.py` (proves the engine, is also the agent/test harness)
→ `web.py` (the teammate-facing demo). Deferred (do NOT build yet, but the design
doesn't preclude it — see "Future frontends"): `proxy.py` (OpenAI-compatible
endpoint so pi can use guarded Qwen3.5 as its brain).

## Environment & hard requirements

- **`ccpp_env`** (`/home/mls01/ccpp_env/bin/python`) — it has `peft 0.20.0`,
  transformers 4.57.6. NOT `qwen35_env` (no peft there).
- The **getpwuid env-var block** at the very top of every entrypoint, before
  `import torch`/`transformers` (copy from `scripts/redteam/replay_jailbreaks_local.py`).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (we share the GPU; be polite).
- Guard loads with ~3 GB; check `nvidia-smi` first and keep total footprint small.
  Provide `--device cpu` fallback (1B model, ~1 s/check on CPU — tolerable).
- OpenRouter: `OPENROUTER_API_KEY` from env (`~/.bashrc`), model id DEFAULTS to
  EXACTLY `qwen/qwen3.5-9b` (other qwen3.5 ids are different models — CLAUDE.md
  pitfall). Swappable via `--model` (see "Expected-to-change" #3) — but never
  change the default silently.
- Never write artifacts to /tmp (lost work twice that way).

## Components

### Guard (the classifier)

- Base: `/data/models/gemma-3-1b-it`, bf16, `device_map="auto"`.
- Adapter: `scripts/model/results/gemma_lora_v2_sweep_phase2/lr3e-4_r8_alpha16_dropout0.0_seed42/best_adapter/`
  (Phase-2 sweep winner, val F1 0.9684). Path is a config knob (`--adapter`) so we can
  swap in future checkpoints.
- Classifier input = **exact training format**, copied from `scripts/model/sweep_lora_v2.py`:
  - `PROMPT_1` instruction text (verbatim constant from that file),
  - `build_sample_text(PROMPT_1, prompt, response)` →
    `...USER PROMPT:\n{prompt}\n\nASSISTANT RESPONSE:\n{response}`,
  - `tokenizer.apply_chat_template([{user: ...}], add_generation_prompt=True)`.
- Context policy (v1): **last exchange only** — `prompt` = latest user message,
  `response` = current assistant response-so-far. Matches training distribution.
  Serializing more history is a later experiment (`--context full`, not in v1).
- The assistant response being classified is Qwen3.5's final `content` only — never the
  reasoning/CoT (classifier was trained on content; see thinking-mode notes below).

### Scoring (first-token logits, NOT generate())

At the classification position, take logits for the first target tokens `harm` vs `un`
(training targets tokenize `harm|ful` vs `un|harm|ful`, so the decision is determined at
token 1) and softmax the pair → **P(harmful)** continuous score.

- Identical decision to greedy generation, but one forward pass and thresholdable.
- VALIDATION STEP (do once during implementation): re-score the sweep's
  `validation_predictions` rows with logit-scoring and confirm agreement with the
  generated labels (~100% expected); record the result in the README.

### Mid-stream checking with KV-cache reuse

The point of the exercise: the guard runs every **~50 Gemma tokens** of streamed response
(`--check-every`, configurable) WITHOUT re-encoding the whole exchange each time.

Per assistant turn:

1. Tokenize the fixed prefix (chat template + instruction + user prompt +
   `ASSISTANT RESPONSE:\n`) and run ONE full forward → keep `past_key_values`
   (`DynamicCache`). Cost paid once per turn.
2. As response text streams in from OpenRouter, re-tokenize the response-so-far
   (tokenization is CPU-cheap; the model forward is what we're saving).
3. When the response has grown by ≥ `--check-every` tokens since the last check:
   forward ONLY the new delta tokens + the few template-suffix tokens
   (`<end_of_turn>\n<start_of_turn>model\n`) against the cache, read the score at the
   final position, then **`cache.crop()`** to remove the suffix so the cache holds exactly
   prefix + response tokens. (Verified available: `DynamicCache.crop` exists in
   transformers 4.57.6.)
   - BPE boundary care: the last cached token can merge with newly arrived text. Handle by
     cropping one extra token and re-feeding the overlap — i.e. only trust the cached
     prefix up to the second-to-last token of the previous tokenization.
4. Always run one **final check at stream end**, even if < 50 tokens since the last
   (short responses like refusals never reach the first checkpoint otherwise).
5. Each check ≈ a ~54-token forward through a 1B model → single-digit ms on the A100;
   negligible next to network streaming. 50 is a comfortable default; lowering is free.

### Target (OpenRouter streaming)

- `POST https://openrouter.ai/api/v1/chat/completions` with `"stream": true`,
  `qwen/qwen3.5-9b`, client-side `messages` history (the API is stateless — multi-turn is
  just resending the list).
- `--thinking on|off` flag toggles `"reasoning": {"enabled": ...}`. Default **off** for
  red-teaming (CLAUDE.md pitfall: thinking can burn the whole token budget and return
  `content=""` with `finish_reason="length"`). When on: parse `delta.reasoning` separately,
  display/log it, but classify only `delta.content`; keep generous `max_tokens`.
- On block: **abort the SSE connection** immediately (this is the mid-stream stop).
- **Token-0 pre-check (added 2026-08-13):** before the target model is called at all,
  the prompt is classified with an EMPTY response (one guard forward — the KV cache is
  already primed by `begin_turn`). A harmful prompt blocks at token 0: no OpenRouter
  call, no generated tokens. The training label rule explicitly covers this ("bad prompt
  + no response ⇒ HARMFUL"), and since `final_label == prompt_harm_label` on every v2
  row, the guard is largely a prompt classifier — so this pre-check is where most of its
  blocking power lands; mid-stream checks remain for response-side harm. Disable with
  `--no-precheck` (ablation knob).

### Blocking semantics

When P(harmful) ≥ `--threshold` (default 0.5) at any check (pre-check, mid-stream or final):

- Abort the stream; the partial harmful response is **NOT** appended to the
  model-visible history (the attacker must not read the leak and adapt) — substitute a
  fixed block notice (e.g. `[aegis] response blocked by safety guard at token N`).
- The REAL partial response goes to the session log only.
- Frontend shows a verdict banner with token offset and score.
- **A block ENDS the conversation.** `session.messages` is cleared before the verdict is
  yielded and the verdict carries `conversation_reset: true`; there is no way to continue
  a blocked conversation from any frontend (web, REPL, `--jsonl`). Enforced in `core.py`,
  not in the UI, so a scripted attacker driving `GuardedSession` directly can't opt out.

  *Why (found live on the web UI, 2026-08-13):* the guard is **single-exchange** —
  `GemmaGuard.begin_turn()` builds its classifier input from the CURRENT user message
  only, never from history — while the target model keeps the whole `messages` list.
  Without the reset, a blocked prompt stayed in Qwen's context and it happily answered
  follow-up questions about it ("what's the first letter of my first message?" → "it's
  h…"), while the guard had already forgotten the text and no longer fired. That is the
  worst of both worlds: it looks like a defense and isn't. Clearing history restores the
  invariant **what the target remembers is never more than what the guard has judged**.

  *What this does NOT fix:* an exchange that stays under threshold is still judged in
  isolation on the following turn (the "$x example" in CLAUDE.md) — see "Open: making the
  guard history-aware" below.

### `core.py` API (frontend-agnostic)

Single class, event-generator interface — every frontend (REPL, stdio, web) just renders
events:

```python
session = GuardedSession(adapter=..., device="cuda", thinking=False,
                         check_every=50, threshold=0.5, log_dir="psiml_data/redteam_sessions")
for event in session.send(user_message):
    ...  # event is a dict, one of:
    # {"type": "token",     "text": ...}                     # visible response delta
    # {"type": "reasoning", "text": ...}                     # thinking mode only
    # {"type": "check",     "n_tokens": N, "p_harmful": p, "ms": t}
    # {"type": "verdict",   "blocked": bool, "p_harmful": p, "n_tokens": N,
    #  "finish_reason": ..., "conversation_reset": bool}
    # {"type": "error",     "message": ...}
session.reset()   # new conversation, same loaded models
```

`conversation_reset` is `true` exactly when `blocked` is `true` — by the time the frontend
sees the verdict, `session.messages` is already `[]` (see Blocking semantics).

### Session logging

One JSONL per session in `psiml_data/redteam_sessions/`, named
`aegis_<utc-timestamp>_<pid>.jsonl`. Lines: session header (config, model ids, adapter
path), then per-turn records containing the user message, full response (or blocked
partial), reasoning if any, every check event (token offset, score, latency), final
verdict. These logs are red-team evidence AND v3 dataset raw material — keep them
complete (real harmful partials included) and never in /tmp.

## Frontend 1: `cli.py` ("aegis-chat")

Two modes:

- **Human REPL (default)**: streams tokens as they arrive, prints check scores
  (dim/inline, or behind `--verbose`), verdict banner on block. Slash commands:
  `/reset`, `/thinking on|off`, `/threshold F`, `/check-every N`, `/stats`
  (checks run, blocks, mean check latency), `/help`.
- **Machine mode (`--jsonl`)**: reads one user message per stdin line, writes one JSON
  event per stdout line (same event schema as `core.py`), keeps conversation state
  between lines, exits on EOF. This is how a **coding agent red-teams AEGIS**:
  `subprocess.Popen([...], stdin=PIPE, stdout=PIPE)` and converse. Also our test harness.

## Frontend 2: web server — the shareable red-team demo

`web.py` + small JS frontend. **This is how red-teamers get access: a URL, nothing
else** — no repo clone, no Python env, no keys (decided 2026-08-13; the CLI stays
the dev/test/agent harness, not the teammate distribution path). The server host
holds everything: the adapter, the base model, the OpenRouter key (never exposed to
clients). Goal: a shareable demo for the team's red-teamers now, possibly
crowdsourced red-teaming later (HF-space-style).

### Hosting reality (verified 2026-08-13 — the web server does NOT run on the cluster)

- This container (FMLE platform, compute node dgx03) is behind **Docker NAT**
  (172.17.0.8, bridge network). Ports cannot be published from inside; there is no
  sshd; outbound internet goes through an authenticated HTTP proxy only.
- **The ONLY inbound door is the platform's Jupyter proxy** → container port 6006 at
  the fixed path `/dgx03/<JUPYTER_ID>/` (that's the IP-in-Chrome URL). No
  `jupyter-server-proxy` is installed, so no other container port is reachable from
  anywhere — a web server hosted here is unreachable for everyone, teammates included.
- ⚠️ That Jupyter server runs with **empty token / no password** — the URL is
  unauthenticated code execution as mls01. Treat it as a secret; never put it in
  docs, tickets, or demos.
- SSH credentials are to the **login node (master)** only; master cannot route to the
  container, and ssh *from* the container is broken (uid 1562). But `/home/mls01`
  (incl. `psiml_data/`) is **NFS-exported from master**, and `/data`, `/fairing` are
  shared Lustre — files are the reliable laptop⇄container channel.

Consequence: hosting on the cluster is out. Instead:

- **Option A (CHOSEN): host off-cluster — the user's laptop for ad-hoc demos, a
  small GCP VM for the stable shared instance.** The stack is portable by design:
  guard = Gemma-3-**1B** (1.9 GB, ~1 s/check on CPU — a GCP e2-small is plenty, no
  GPU needed; `--device` fallback already spec'd), target = **OpenRouter**
  (reachable from anywhere). Server host needs: this repo + the 25 MB LoRA adapter
  (copy out via `scp` through the login node) + gemma-3-1b-it base (HF download) +
  an OpenRouter key. Red-teamers just open the URL.
  - **Exposure**: on the laptop, LAN binding is enough for an in-person demo. On the
    GCP VM, an ephemeral IP + firewall rule works for a small trusted group; when a
    stable hostname / HTTPS / proper access gate is wanted, put **Cloudflare Tunnel
    (free tier)** in front — `cloudflared` on the VM, no open inbound ports, and
    Cloudflare Access (email-PIN allowlist) as the auth layer. Not expected to be
    needed day one.
  - **Auth floor regardless of host**: never run it bare-unauthenticated beyond a
    LAN — it proxies paid OpenRouter traffic and logs harmful text. Minimum a shared
    token; Cloudflare Access when the tunnel is in use.
  - **Session logs live on the server host**; they're v3-dataset raw material, so
    after a GCP stint ship `psiml_data/redteam_sessions/` back to the cluster
    (`scp` to the login node lands it in the NFS home).
- **Option B (fallback, only if the cluster must be in the loop — e.g. local Qwen3.5
  as target on the A100): NFS file queue.** The off-cluster FastAPI backend (laptop
  or GCP VM) writes request JSON to `psiml_data/aegis_queue/` (writable via the login
  node), a watcher (`aegis-serve --queue`) in the container runs guard inference and
  appends events to a response file, the backend polls. Zero network changes, works
  today; ~1–3 s overhead per exchange and clunky pseudo-streaming via append-poll.
  Do not build unless Option A proves insufficient.
- **Option C (REJECTED): laptop backend RPCs into JupyterLab's kernel API.** Works in
  principle (REST + WS, no token!), but fragile, couples the demo to a dev tool, and
  leans on the unauthenticated endpoint above. Don't.

### web.py design (unchanged by the hosting decision)

- Backend: **FastAPI + WebSocket** (or SSE) endpoint, e.g. `/ws/chat`. Protocol = the
  exact event schema above, one JSON message per event; client sends
  `{"type": "user", "text": ...}` / `{"type": "reset"}`.
- **One `GuardedSession` per WebSocket connection** (session state = conversation),
  all sharing the single loaded Gemma. Guard checks serialize on one model —
  fine at demo scale (~1 s/check even on CPU); if contention ever matters, a small
  asyncio queue in front of the guard is enough.
- **SUPERSEDED by single occupancy (2026-08-13).** "One session per connection, all
  sharing one guard" is not safe as written: since the guard became history-aware,
  the conversation state (KV cache, `_committed_ids`, `history`) lives on the
  `GemmaGuard`, so concurrent sessions bleed context into each other and each
  other's block/reset desyncs guard vs target history. A lock does not fix it — it
  serializes turns in time, but the conversation slot is still shared. `web.py` now
  admits **exactly one conversation at a time** (`SeatManager`: FIFO queue, position
  broadcast to waiters, `guard.reset_conversation()` at handover via the first
  turn's `pre_turn` under the turn lock, so an abandoned mid-turn worker thread from
  the previous holder can't be reset out from under). Queued clients get
  `{"type":"seat","state":"queued","position":N,"waiting":M}` and a locked composer;
  they open no session and no log file until granted. Per-user guards (the other
  fix) cost ~2 GB each — out of budget on the laptop/VM.
- Frontend: minimal chat page rendering token deltas live, check-score sparkline, red
  block banner. Serve statically from FastAPI for a one-process demo.
- Network checklist (server edition — laptop or GCP VM):
  - laptop: bind `0.0.0.0` for LAN demos, `127.0.0.1` otherwise. GCP VM: bind
    `0.0.0.0` behind the GCP firewall rule, or `127.0.0.1` when fronted by
    `cloudflared`. Pick a free high port either way;
  - **CORS** if the page is served from a different origin than the WS endpoint
    (serving the static page from the same FastAPI process avoids this entirely);
  - the host needs outbound HTTPS to OpenRouter (any normal connection);
  - auth per Option A: shared token minimum beyond LAN, Cloudflare Access when the
    tunnel is in use.
- Reuse check: `core.py` must stay I/O-agnostic; if `web.py` needs anything the CLI
  didn't, extend the event schema, don't special-case.

## Future frontends (deferred, one paragraph)

`proxy.py` ("aegis-serve"): OpenAI-compatible `POST /v1/chat/completions` wrapping
`GuardedSession`, so pi (custom provider) or any OpenAI client uses guarded Qwen3.5 as
its brain — red-teaming a full agentic deployment. Known wrinkle to solve then: agent
frameworks put tool results in user/tool-role messages, so "last user message" is not
always attacker text; the classifier's context policy needs revisiting for that shape.
NOT built in v1 (time).

## Config knobs (CLI flags / GuardedSession args)

| knob | default | notes |
|---|---|---|
| `--adapter` | phase2 winner path above | swap checkpoints; `none` = bare base model (zero-shot guard) |
| `--guard-prompt` | `PROMPT_1` verbatim | classifier instruction text; must match training format when an adapter is loaded |
| `--model` | `qwen/qwen3.5-9b` | OpenRouter target model — see "Expected-to-change" below |
| `--device` | `cuda` if available else `cpu` | `cpu` fallback (~1 s/check) |
| `--thinking` | off | OpenRouter reasoning toggle |
| `--check-every` | 50 | Gemma tokens between checks |
| `--threshold` | 0.5 | P(harmful) block threshold; revisit after logit-validation |
| `--temperature` / `--max-tokens` | 1.0 / 3072 | target sampling (3072 per round-2 lesson) |
| `--log-dir` | `psiml_data/redteam_sessions/` | |
| `--jsonl` | off | machine stdio mode |

## Expected-to-change pieces (design for swap-ability, 2026-08-13)

These are deliberate experiment variables, NOT fixed constants — keep them as
constructor args / CLI flags, never bake them deeper than a default:

1. **The classifier adapter (and whether there IS one).** The Gemma-3-1B base is
   fixed, but the LoRA checkpoint will change as sweeps continue — and we also want
   to red-team the **zero-shot** (no-adapter) guard for comparison. `--adapter none`
   must load bare gemma-3-1b-it.
2. **The classifier instruction prompt** (`PROMPT_1`). Different wordings are an
   experiment axis (zero-shot variants already exist from the v2 eval). Note the
   coupling: with a LoRA adapter loaded, the prompt MUST stay the verbatim training
   prompt or the adapter is off-distribution; prompt-swap experiments make most sense
   with `--adapter none`.
3. **The target model on OpenRouter.** Qwen3.5-9B is proving VERY hard to jailbreak
   (38/300 under our best templates) — for red-team yield we may point AEGIS at a
   weaker/more permissive model id. The guard doesn't care (it classifies text, not
   weights); the tradeoff to record when we do this: positives no longer come from
   the actually-deployed model (CLAUDE.md "generate training data ONLY from the
   guarded model" applies to TRAINING data — red-team demos are allowed to differ,
   but say so in the session log header via the `target_model` field).
4. **Red-teamer briefing: false positives are failures too.** At some point we
   explicitly ask red-teamers to *stop attacking and instead think up benign
   requests that get blocked* — an over-firing guard is as much a classifier
   failure as a jailbreak, and AEGIS logs both. Blocked-benign turns in the session
   logs are FP evidence (threshold calibration + future training data), so the
   block notice/verdict must always record `p_harmful` and the token offset even
   when the blocked text is innocuous. Plan the web UI copy so a blocked benign
   request doesn't look like a red-teamer "win".

## Design: history-aware guard (multi-turn) — IMPLEMENTED 2026-08-13

**Status: built and verified.** `core.py` implements the design below; the
persistent append-only cross-turn cache is proved equal to a full recompute on the
real Gemma model (`tests/test_guard_live.py::test_kv_multiturn_matches_full`, 16/16
live checks incl. a 3rd-turn pre-check over 2 turns of history), plus interface-level
multi-turn commit/reset tests in `tests/test_core_logic.py` (71/71). The 512-token
sliding-window caveat below is a KNOWN, unmeasured limitation on long conversations —
still to be characterized empirically; nothing else in the design is open.

The reset-on-block feature closes the case where a block has ALREADY fired. It does **not**
make the guard see conversation history — and the decision (2026-08-13) is that it should:
*"the entire context should stay in the classifier's memory."* Today
`GemmaGuard.begin_turn(prompt)` builds its input from the single current user message, so a
multi-turn attack that keeps each turn individually under threshold (the "$x example" in
CLAUDE.md) is invisible to it. This is a separate change from the reset; the two **compose**
(reset handles "already fired"; history-awareness handles "accumulating harm not yet fired").

### Decision: interleaved per-turn headers (chat-log serialization)

Repeat the `USER PROMPT:` / `ASSISTANT RESPONSE:` slot headers **once per turn** rather than
flattening history into a single prompt slot. The classifier input at a mid-stream check on
turn *k* is:

```
<bos><start_of_turn>user
{PROMPT_1}

USER PROMPT:
{prompt_1}

ASSISTANT RESPONSE:
{response_1}

USER PROMPT:
{prompt_2}

ASSISTANT RESPONSE:
{response_2}
...
USER PROMPT:
{prompt_k}

ASSISTANT RESPONSE:
{response_k_so_far}          <-- the streaming turn; first-token AFTER this is SCORED
<end_of_turn>
<start_of_turn>model
```

Score meaning is unchanged from single-turn: P(harmful) is the softmax over the `harm`/`un`
logits at the first position after the *current* turn's `ASSISTANT RESPONSE:`, i.e. "is THIS
response harmful **given the whole conversation before it**". Rejected alternatives: (a) all
history mashed into ONE `USER PROMPT:` block scoring one response — puts prior assistant
replies under a "user prompt" label, and forces a full-prefix re-prime every turn; (b)
per-turn independent scoring + running max/EWMA — cannot catch the "$x example" at all, since
each turn is benign in isolation and the cross-turn signal is exactly what it discards.

**Why this one, concretely — the KV cache becomes append-only.** Turn *k*'s serialized text
is `{completed turns 1..k-1, byte-identical to last turn} + {USER PROMPT:\n prompt_k \n\n
ASSISTANT RESPONSE:\n} + {streaming response_k}`. Everything before the current turn never
changes, so a **persistent cross-turn `DynamicCache`** just grows: at each turn start we
forward ONLY the new user-turn header+prompt tokens; then stream response tokens as today. No
re-prime of history, no recompute of prior turns. Accepting a (probably) mediocre multi-turn
F1 — off-distribution, no multi-turn training data — buys this simple, fast design; that
tradeoff is deliberate.

### KV-cache mechanics (and why the crop bug does NOT bite here)

The single-turn design's crop step (append SUFFIX, score, crop it back off) is what hits the
gemma-3-1b sliding-window `crop()` failure past 512 tokens (CLAUDE.md). The multi-turn design
keeps that same *fix* and never introduces a cross-turn crop:

- The persistent cache is **append-only** — we only ever extend it, never `crop()` it. So the
  sliding-window `crop()` ValueError is structurally impossible on the real cache.
- Each mid-stream check still scores via the existing trick: forward `response_delta +
  SUFFIX` over a **deep copy** of the cache and read the last-position logits; the real cache
  gets only the response delta (no suffix). Unchanged from single-turn, already fp32-exact.
- At turn end: append the full `response_k` tokens (no suffix) to the persistent cache so
  turn *k+1* starts from it.

**Invariants this relies on:**
1. The persistent cache holds exactly `prefix + turns[1..k-1] fully + turn-k header + response_k_so_far`
   at all times — never the SUFFIX, never a block notice.
2. Serialization is byte-stable for completed turns: once turn *j* is appended, its text/token
   ids are frozen (so the cache and any re-tokenization agree). BPE boundary at each new turn's
   header is handled the same way `check_incremental` already realigns the response boundary.
3. A turn is appended to history **only if it did NOT block** — see composition below.

**Perf cost to accept:** the per-check deep copy grows with conversation length (copying a
1000-token cache every `check_every` tokens). Correctness-neutral; optimize later (e.g. copy
only the tail layers' new entries) if it bites on CPU.

### Composition with reset-on-block (clean, no ambiguity)

Because a block now **ends the conversation** (history cleared, `conversation_reset: true`),
every prior assistant turn still present in a live conversation is a REAL, un-blocked response.
So the serialized history never contains a `[aegis] response blocked...` notice and we never
face "do we feed the real partial or the notice back to the guard" — the reset already
guaranteed it away. The block-notice substitution stays a *target-model-history* concern only.

### Pre-check in multi-turn (gets strictly stronger)

The token-0 pre-check becomes "classify full history + current prompt + EMPTY response". In
the "$x example", turn-2's prompt ("how to make $y") is harmful *given* turn-1 — which is now
in the history — so the pre-check can fire on the harmful FOLLOW-UP prompt before the target
model is called, not only mid-response. Keep it on by default.

### The real limitation: the 512-token sliding window (verified against config)

gemma-3-1b (`unsloth/gemma-3-1b-it`, verified 2026-08-13): **26 layers, `sliding_window=512`,
`sliding_window_pattern=6`** → only **4 global/full-attention layers (indices 5, 11, 17, 23)**;
the other 22 are sliding-window. Consequence: once serialized history exceeds 512 tokens, the
classification position attends to the full early context **only through those 4 global
layers**; the 22 local layers see just the last 512 tokens. So a cross-turn attack whose key
context (`$y = bomb`) sits >512 tokens back is carried by 4/26 layers only. Whether that is
enough signal is an **empirical question, not a given** — measure it; do not assume long-range
multi-turn works. Mitigations if it doesn't: cap/compress history to fit the window, or drop
the oldest turns (sliding *conversation* window) and accept the recency bias.

### Off-distribution & calibration

The LoRA adapter was trained on single exchanges (v2, `final_label == prompt_harm_label`), so
multi-turn serialized inputs are off-distribution — expect the operating threshold (tuned on
single-exchange val) to drift. Plan, cheapest-first (matches project culture):
1. **Cheap test, no new training:** run the current adapter over the "$x example" hand-built
   cases + `data/switched_prompt_dataset.jsonl` (benign-prompt/harmful-response pairs) served
   as 2-turn conversations. Measures whether response-side / cross-turn detection fires at all.
2. If it fails or is weak: build a **v3 multi-turn dataset** (serialized multi-turn exchanges;
   red-team session logs are raw material) and LoRA-retrain, then **re-calibrate the
   threshold** on a multi-turn val split.

### Interface (frontend-agnostic; event schema UNCHANGED) — as built

- `GemmaGuard` holds a persistent cross-turn cache (`cache`/`cached_seq`), the committed base
  (`_committed_ids`), the current-turn prefix (`_prefix_ids`), and `history`.
  - `begin_turn(prompt)` computes `_prefix_ids = _committed_ids + turn_header_ids` and does
    **nothing to the cache** — the header is appended lazily by the turn's first
    `check_incremental` (via `_advance_response` → `_forward_append`). This is what makes an
    abandoned turn need no rollback.
  - `commit_turn(response)` advances the cache through the final response (usually a no-op
    after the final check) and freezes `_committed_ids`/`history`.
  - `reset_conversation()` clears it all.
  - `check_incremental` is unchanged in contract (scores the streaming delta) — refactored onto
    `_advance_response`/`_forward_append`/`_reprime`.
- `GuardedSession.send`: commits non-blocked turns (`guard.commit_turn(content_so_far)`);
  `reset()` clears `messages` AND the guard cache in lockstep (called on every block).
- Frontends: **no change** — same event dicts.

**Verified invariants (see `GemmaGuard` docstring):** `cached_seq` is always a prefix of
`_prefix_ids + current_response`; `_committed_ids` is a prefix of `cached_seq`; the cache never
holds the SUFFIX (deep-copy scoring) nor a block notice (a block resets first). Append-only ⇒
no cross-turn `crop()` ⇒ the sliding-window crop bug is structurally impossible.

**Remaining empirical work (not code):** the 512-token limitation above — measure whether a
cross-turn attack whose key context sits >512 tokens back is still caught (only the 4 global
layers carry it). And the calibration plan: run the current single-turn adapter over the
"$x example" + `switched_prompt_dataset.jsonl` served as 2-turn convos before deciding whether a
v3 multi-turn dataset + retrain + threshold recalibration is needed.

## Verification checklist (status 2026-08-13, laptop CPU)

1. ✅ **KV-cache numeric check** — DONE, and it caught two different things:
   (a) the originally-suspected "stale cache" bug **does not exist** — fp32
   ablation (`tests/debug_kv.py`) shows incremental == full recompute to
   ~1.5e-6 even past the 512-token sliding window; the bf16 wobble (≤6e-3
   short, ≤2e-2 long) is matmul reduction-order noise + bf16 logit
   quantization, harmless at the 0.5 threshold; (b) a REAL bug it did catch:
   `DynamicSlidingWindowLayer.crop()` raises once the exchange is longer than
   gemma-3-1b's 512-token window → scheme switched to "suffix over a throwaway
   deep-copied cache", see README "Implementation notes".
2. ⏳ **Logit-scoring check** — pending: needs the sweep's
   `validation_predictions_epoch_6.csv`, which is cluster-only (gitignored
   results dir). Run `tests/test_guard_live.py --logit-only` on the cluster.
3. ✅ **Live smoke (jsonl mode, laptop CPU)**: benign soup recipe streamed to
   completion (742 Gemma tokens, 15 checks, no block, p≈0) crossing the 512
   window; direct harmful request blocked mid-stream at token 50 (p≈1.0,
   classification = harmful OR-rule fires on harmful prompt + refusal).
   **Update (token-0 pre-check)**: the same harmful request now blocks at token 0
   (p≈0.9999977, no OpenRouter call made); benign prompt scores p≈2.5e-8 at
   token 0 and proceeds normally.
   Known-successful-jailbreak block test deferred — per 2026-08-13 decision,
   Qwen3.5 is very hard to jailbreak so we are NOT re-deriving live jailbreaks
   as a smoke step; the OR-rule block above already exercises the block path.
4. ✅ Session log inspection: header carries full config (incl. target model +
   guard prompt), blocked turn keeps the REAL partial in the log while
   `messages_after` shows the block notice.
5. ✅ `README.md` written; `CLAUDE.md` Status updated.
6. ✅ **Web server live smoke (laptop CPU, 2026-08-13)**: `web.py --device cpu`
   served the page + `/api/config`; over a real WebSocket a benign turn
   streamed token/check/verdict events to completion (p≈0), a harmful prompt
   was blocked by the token-0 pre-check (p=1.0, block notice in history, real
   turn in the `aegis_*_w001.jsonl` web session log), and `reset` worked.
   `tests/test_web.py` (stub session, no model/network) covers page/config
   endpoints, event ordering, blocked-turn protocol, reset, malformed messages,
   token auth (accepted/rejected), and — added with single occupancy
   (2026-08-13) — queueing: second/third client queued with position, queued
   client's `user`/`reset` rejected, no session or log file opened while queued,
   holder served normally while others wait, seat handed to the next in line on
   disconnect, and `guard.reset_conversation()` called exactly once per
   occupant. **Status: the queue tests have NOT been run yet** (the one run
   attempted was killed — the box was in use). Note for whoever runs them: the
   multi-connection block enters `TestClient` as a context manager on purpose —
   an un-entered TestClient makes a new portal (thread + event loop) per
   request, and the `SeatManager`'s asyncio primitives are single-loop (as in
   production under uvicorn). Still not done: browser-side UI check (page
   verified served, JS not exercised headlessly).

## Pitfalls carried over (read CLAUDE.md for full text)

- getpwuid env-var block before torch/transformers imports — scripts too, not just notebooks.
- OpenRouter model id exactly `qwen/qwen3.5-9b`.
- Thinking mode: `content` can be empty with `finish_reason="length"` — handle, don't crash.
- GPU is shared with a teammate's finetuning run — keep footprint ~3 GB, don't kill
  `[Not Found]` PIDs without asking.
- `scripts/model/results/` is gitignored → the adapter path is machine-local; note this
  in README for anyone rerunning elsewhere. For off-cluster serving (Option A above):
  the adapter is only **25 MB** — copy it from the cluster to the server host via the
  login node
  (`scp mls01@<master>:/home/mls01/scripts/model/results/gemma_lora_v2_sweep_phase2/lr3e-4_r8_alpha16_dropout0.0_seed42/best_adapter/*`);
  gemma-3-1b-it base (1.9 GB) comes from HF. Teammates never need either — they get
  the web URL.
