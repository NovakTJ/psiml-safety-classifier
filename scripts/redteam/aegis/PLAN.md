# AEGIS — guarded-chat interface for red-teaming the Gemma classifier

Status: **PLANNED, not implemented** (spec written 2026-08-13; implementation next).

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

## Layout

```
scripts/redteam/aegis/
├── PLAN.md    # this file
├── core.py    # the engine — no terminal/HTTP assumptions
├── cli.py     # frontend 1: "aegis-chat"  (human REPL + machine-readable stdio mode)
└── README.md  # usage (written with the implementation)
psiml_data/redteam_sessions/   # session logs (one JSONL per session; gitignored if large)
```

Deferred (do NOT build yet, but design doesn't preclude them — see "Future frontends"):
`proxy.py` (OpenAI-compatible endpoint so pi can use guarded Qwen3.5 as its brain) and
`web.py` (browser frontend).

## Environment & hard requirements

- **`ccpp_env`** (`/home/mls01/ccpp_env/bin/python`) — it has `peft 0.20.0`,
  transformers 4.57.6. NOT `qwen35_env` (no peft there).
- The **getpwuid env-var block** at the very top of every entrypoint, before
  `import torch`/`transformers` (copy from `scripts/redteam/replay_jailbreaks_local.py`).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (we share the GPU; be polite).
- Guard loads with ~3 GB; check `nvidia-smi` first and keep total footprint small.
  Provide `--device cpu` fallback (1B model, ~1 s/check on CPU — tolerable).
- OpenRouter: `OPENROUTER_API_KEY` from env (`~/.bashrc`), model id EXACTLY
  `qwen/qwen3.5-9b` (other qwen3.5 ids are different models — CLAUDE.md pitfall).
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

### Blocking semantics

When P(harmful) ≥ `--threshold` (default 0.5) at any check (mid-stream or final):

- Abort the stream; the partial harmful response is **NOT** appended to the
  model-visible history (the attacker must not read the leak and adapt) — substitute a
  fixed block notice (e.g. `[aegis] response blocked by safety guard at token N`).
- The REAL partial response goes to the session log only.
- Frontend shows a verdict banner with token offset and score.

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
    #  "finish_reason": ...}
    # {"type": "error",     "message": ...}
session.reset()   # new conversation, same loaded models
```

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

## Frontend 2 (future, spec'd now): web server

`web.py` + small JS frontend — the goal is a shareable demo (possibly crowdsourced
red-teaming, HF-space-style).

- Backend: **FastAPI + WebSocket** (or SSE) endpoint, e.g. `/ws/chat`. Protocol = the
  exact event schema above, one JSON message per event; client sends
  `{"type": "user", "text": ...}` / `{"type": "reset"}`.
- **One `GuardedSession` per WebSocket connection** (session state = conversation),
  all sharing the single loaded Gemma on GPU. Guard checks serialize on one model —
  fine at demo scale (ms per check); if contention ever matters, a small asyncio queue
  in front of the guard is enough.
- Frontend: minimal chat page rendering token deltas live, check-score sparkline, red
  block banner. Serve statically from FastAPI for a one-process demo.
- Network checklist (cluster-specific, all solvable):
  - bind `0.0.0.0` + pick a free high port; external access likely needs an
    **SSH tunnel** (`ssh -L 8080:localhost:8080 <cluster>`) since cluster ingress is
    restricted — verify before promising a URL;
  - **CORS** if the page is served from a different origin than the WS endpoint;
  - cluster already has outbound HTTPS to OpenRouter (the jailbreak scripts use it);
  - do NOT expose the endpoint unauthenticated to the public internet while it proxies
    paid OpenRouter traffic and logs harmful text — at minimum a shared token.
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
| `--adapter` | phase2 winner path above | swap checkpoints |
| `--device` | `cuda` | `cpu` fallback |
| `--thinking` | off | OpenRouter reasoning toggle |
| `--check-every` | 50 | Gemma tokens between checks |
| `--threshold` | 0.5 | P(harmful) block threshold; revisit after logit-validation |
| `--temperature` / `--max-tokens` | 1.0 / 3072 | target sampling (3072 per round-2 lesson) |
| `--log-dir` | `psiml_data/redteam_sessions/` | |
| `--jsonl` | off | machine stdio mode |

## Verification checklist (before calling it done)

1. **KV-cache numeric check**: incremental score (delta+crop scheme) == full-recompute
   score on a fixed exchange, to fp tolerance. This validates the trickiest code.
2. **Logit-scoring check**: agreement with sweep `validation_predictions` labels.
3. **Live smoke**: one benign prompt (no block, full stream) + one known-successful
   jailbreak prompt from `psiml_data/jailbreak_v1/successful_jailbreaks.jsonl`
   (expect mid-stream block) through `aegis-chat`, both modes (REPL via pexpect or
   `--jsonl` directly).
4. Session log inspection: complete, well-formed, real partial preserved on block.
5. Update `CLAUDE.md` Status (AEGIS entry + pointer here) and write `README.md`.

## Pitfalls carried over (read CLAUDE.md for full text)

- getpwuid env-var block before torch/transformers imports — scripts too, not just notebooks.
- OpenRouter model id exactly `qwen/qwen3.5-9b`.
- Thinking mode: `content` can be empty with `finish_reason="length"` — handle, don't crash.
- GPU is shared with a teammate's finetuning run — keep footprint ~3 GB, don't kill
  `[Not Found]` PIDs without asking.
- `scripts/model/results/` is gitignored → the adapter path is machine-local; note this
  in README for anyone rerunning elsewhere.
