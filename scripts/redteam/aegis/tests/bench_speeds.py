#!/usr/bin/env python3
"""Measure the two speeds that matter for the streaming-vs-classifier UX:

  A) Qwen3.5-9B streaming throughput from OpenRouter (tokens/sec as received)
  B) GemmaGuard.check_incremental() latency on this machine, at the response
     lengths the mid-stream loop actually checks at (50, 100, ... tokens)

Run:  ~/aegis_env/bin/python scripts/redteam/aegis/tests/bench_speeds.py
Needs OPENROUTER_API_KEY (repo .env is loaded). CPU or CUDA both fine.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cli import load_dotenv  # noqa: E402
from core import REPO_ROOT, GemmaGuard, GuardedSession  # noqa: E402

load_dotenv(os.path.join(REPO_ROOT, ".env"))

PROMPT = ("Explain how a bicycle derailleur works, mechanically, in detail. "
          "Cover the cage, jockey wheels, spring tension and cable actuation.")


def bench_qwen_stream():
    """Raw OpenRouter stream timing: TTFT, per-token gaps, total tok/s."""
    sess = GuardedSession.__new__(GuardedSession)  # bypass guard load
    sess.model_id = "qwen/qwen3.5-9b"
    sess.thinking = False
    sess.temperature = 1.0
    sess.max_tokens = 400
    sess._api_key = os.environ["OPENROUTER_API_KEY"]
    sess.messages = [{"role": "user", "content": PROMPT}]

    t0 = time.time()
    stream = sess._open_stream()
    ttft = None
    n_tok_events = 0
    text = ""
    for ev in sess._read_stream(stream):
        if ev["type"] == "token":
            if ttft is None:
                ttft = time.time() - t0
            n_tok_events += 1
            text += ev["text"]
    total = time.time() - t0
    stream.close()
    gen_time = total - (ttft or 0)
    print(f"[qwen] TTFT={ttft:.2f}s  total={total:.2f}s  "
          f"sse_events={n_tok_events}  chars={len(text)}")
    print(f"[qwen] ~{len(text)/4:.0f} tokens in {gen_time:.2f}s of generation "
          f"-> ~{len(text)/4/gen_time:.0f} tok/s (chars/4 approx)")
    return text


def bench_guard(response_text):
    """check_incremental latency as the response grows in 50-token steps."""
    print("[guard] loading GemmaGuard …", flush=True)
    guard = GemmaGuard(device=None)  # auto (cpu here)
    guard.begin_turn(PROMPT)
    tok = guard.tokenizer
    ids = tok.encode(response_text, add_special_tokens=False)
    print(f"[guard] device={guard.device}  response={len(ids)} tokens")
    for upto in range(0, len(ids) + 1, 50):
        chunk = tok.decode(ids[:upto])
        t0 = time.time()
        p, n, _ = guard.check_incremental(chunk)
        ms = (time.time() - t0) * 1000
        print(f"[guard] check at token {n:4d}: {ms:7.0f} ms  p={p:.4f}",
              flush=True)
    guard.close()


if __name__ == "__main__":
    text = bench_qwen_stream()
    bench_guard(text)
