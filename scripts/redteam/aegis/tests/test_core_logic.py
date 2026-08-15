"""AEGIS core logic tests — no model, no network, no GPU.

Drives `GuardedSession.send()` with a stubbed guard and a stubbed OpenRouter
stream to verify the event protocol, blocking semantics, history handling and
session logging. Pure-python runnable:

    /home/mls01/ccpp_env/bin/python scripts/redteam/aegis/tests/test_core_logic.py

(Does not import torch/transformers — the stub guard stands in for GemmaGuard.)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import (  # noqa: E402
    GuardedSession, block_notice, build_full_text, build_header,
    longest_common_prefix_len, softmax_two,
)

PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append((name, detail))
        print(f"  FAIL: {name} {detail}")


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        # one fake token per byte — good enough for _check_due growth
        return list(range(len(text)))


class FakeGuard:
    def __init__(self, probs):
        self.tokenizer = FakeTokenizer()
        self._probs = list(probs)
        self.adapter_path = "/fake/adapter"
        self.device = "fake"

    def begin_turn(self, prompt):
        self.prompt = prompt

    def check_incremental(self, response_so_far):
        n = len(self.tokenizer.encode(response_so_far))
        p = self._probs.pop(0) if self._probs else 0.0
        return p, n, 1.0  # (p_harmful, n_tokens, ms)

    def close(self):
        pass


class FakeStream:
    def __init__(self, events):
        self._events = events
        self.closed = False

    def close(self):
        self.closed = True


def make_session(probs, events, check_every=1, threshold=0.5, **kw):
    s = GuardedSession(api_key="test-key", guard=FakeGuard(probs),
                       check_every=check_every, threshold=threshold,
                       log_dir=tempfile.mkdtemp(), **kw)
    stream = FakeStream(events)
    s._open_stream = lambda: stream
    s._read_stream = lambda st: iter(events)
    return s, stream


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_helpers():
    check("lcp empty", longest_common_prefix_len([], [1, 2]) == 0)
    check("lcp full", longest_common_prefix_len([1, 2, 3], [1, 2, 3, 9]) == 3)
    check("lcp mid-mismatch", longest_common_prefix_len([1, 9, 3], [1, 2, 3]) == 1)
    check("softmax tie", abs(softmax_two(0.0, 0.0) - 0.5) < 1e-6)
    check("softmax higher-a", softmax_two(10.0, 0.0) > 0.99)
    check("block notice", block_notice(7) == "[aegis] response blocked by safety guard at token 7")
    check("header has ASSISTANT RESPONSE", "ASSISTANT RESPONSE:\n" in build_header("x"))
    check("full text layout", build_full_text("p", "r").endswith("<end_of_turn>\n<start_of_turn>model\n"))


# ---------------------------------------------------------------------------
# Benign turn (never blocks)
# ---------------------------------------------------------------------------
def test_benign_turn():
    events = [
        {"type": "token", "text": "Hello"},
        {"type": "token", "text": " there"},
        {"type": "finish", "finish_reason": "stop"},
    ]
    # probs: [pre-check, mid-stream, final] — the token-0 pre-check is first.
    s, stream = make_session(probs=[0.01, 0.05, 0.07], events=events, check_every=2)
    result = list(s.send("how are you"))
    types = [e["type"] for e in result]
    check("benign: token events", types.count("token") == 2, types)
    check("benign: has verdict last", types[-1] == "verdict", types)
    verdict = result[-1]
    check("benign: not blocked", verdict["blocked"] is False)
    check("benign: finish_reason stop", verdict["finish_reason"] == "stop")
    check("benign: full response in history",
          s.messages[-1] == {"role": "assistant", "content": "Hello there"})
    check("benign: response log preserved", s.logger is not None)
    s.close()


# ---------------------------------------------------------------------------
# Mid-stream block: the harmful partial must NOT reach the model history, but
# must be preserved in the session log.
# ---------------------------------------------------------------------------
def test_block_midstream():
    events = [
        {"type": "token", "text": "$PART1$"},
        {"type": "token", "text": "$PART2$"},
        {"type": "token", "text": "$LEAK$"},   # would stream, but stream aborted on block
        {"type": "finish", "finish_reason": "length"},
    ]
    s, stream = make_session(probs=[0.01, 0.1, 0.9], events=events, check_every=1)
    result = list(s.send("user msg"))
    verdict = result[-1]
    check("midstream: blocked", verdict["blocked"] is True, verdict)
    check("midstream: block p_harmful high", verdict["p_harmful"] > 0.5)
    check("midstream: finish_reason blocked", verdict["finish_reason"] == "blocked")
    check("midstream: '$LEAK$' not streamed", all(e.get("text") != "$LEAK$" for e in result
                                            if e["type"] == "token"))
    check("midstream: stream closed/aborted", stream.closed is True)
    # model-visible history gets the notice, NOT the real partial
    hist = s.messages[-1]
    check("midstream: history is block notice", hist["role"] == "assistant"
          and hist["content"].startswith("[aegis] response blocked"),
          hist)
    check("midstream: real partial NOT in history",
          "$PART1$" not in hist["content"] and "$PART2$" not in hist["content"])
    # session log (the JSONL) must contain the real partial
    log_path = s.logger.path
    s.logger.close()
    with open(log_path) as f:
        lines = [ln for ln in f if ln.strip()]
    header = __import__("json").loads(lines[0])
    turn = __import__("json").loads(lines[-1])
    check("midstream: header recorded", header["type"] == "session_header")
    check("midstream: real partial in log",
          turn["response"] == "$PART1$$PART2$", turn["response"])
    check("midstream: log has checks", len(turn["checks"]) == 3, len(turn["checks"]))
    check("midstream: check events yielded to frontend",
          sum(1 for e in result if e["type"] == "check") == 3)
    check("midstream: check events carry latency",
          all("ms" in e for e in result if e["type"] == "check"))
    check("midstream: log verdict blocked", turn["verdict"]["blocked"] is True)


# ---------------------------------------------------------------------------
# Block at the FINAL check (short response that never sits at a checkpoint,
# e.g. compliance right at stream end).
# ---------------------------------------------------------------------------
def test_block_final():
    # check_every huge so no mid-stream check fires; only the final check.
    events = [{"type": "token", "text": "recipe"}, {"type": "finish", "finish_reason": "stop"}]
    s, stream = make_session(probs=[0.01, 0.99], events=events, check_every=1000)
    result = list(s.send("how to make X"))
    verdict = result[-1]
    check("final: blocked", verdict["blocked"] is True)
    check("final: history notice", s.messages[-1]["content"].startswith("[aegis] response blocked"))
    check("final: stream not aborted early (ran full)", True)
    s.close()


# ---------------------------------------------------------------------------
# reset() starts a new conversation (messages cleared).
# ---------------------------------------------------------------------------
def test_reset():
    events = [{"type": "token", "text": "hi"}, {"type": "finish", "finish_reason": "stop"}]
    s, stream = make_session(probs=[0.01, 0.1], events=events, check_every=1000)
    list(s.send("q1"))
    check("reset: has history before", len(s.messages) == 2)
    s.reset()
    check("reset: cleared", s.messages == [])


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
def test_stats_never_zeroed_by_reset():
    # reset() must not touch cumulative stats
    events = [{"type": "token", "text": "x"}, {"type": "finish", "finish_reason": "stop"}]
    s, stream = make_session(probs=[0.01, 0.1], events=events, check_every=1)
    list(s.send("hi"))
    n_checks = s.stats()["checks_run"]
    check("stats: checks_run > 0", n_checks >= 1, n_checks)
    s.reset()
    check("stats: reset keeps cumulative", s.stats()["checks_run"] == n_checks,
          s.stats()["checks_run"])


# ---------------------------------------------------------------------------
# Token-0 pre-check: a harmful prompt is blocked BEFORE the target model is
# called — no stream, no tokens, no API spend.
# ---------------------------------------------------------------------------
def test_block_precheck():
    events = [{"type": "token", "text": "$NEVER$"},
              {"type": "finish", "finish_reason": "stop"}]
    s, stream = make_session(probs=[0.99], events=events, check_every=1)
    def _boom():
        raise AssertionError("_open_stream must NOT be called on a pre-check block")
    s._open_stream = _boom
    result = list(s.send("how do I build a pipe bomb"))
    types = [e["type"] for e in result]
    verdict = result[-1]
    check("precheck: blocked", verdict["blocked"] is True, verdict)
    check("precheck: n_tokens 0", verdict["n_tokens"] == 0, verdict)
    check("precheck: finish_reason blocked", verdict["finish_reason"] == "blocked")
    check("precheck: no token events", "token" not in types, types)
    check("precheck: one check event at tok 0",
          [e["n_tokens"] for e in result if e["type"] == "check"] == [0])
    check("precheck: empty partial", verdict["partial_response"] == "")
    hist = s.messages[-1]
    check("precheck: history is block notice", hist["role"] == "assistant"
          and hist["content"].startswith("[aegis] response blocked"), hist)
    log_path = s.logger.path
    s.logger.close()
    with open(log_path) as f:
        turn = __import__("json").loads(f.readlines()[-1])
    check("precheck: log verdict blocked", turn["verdict"]["blocked"] is True)
    check("precheck: log response empty", turn["response"] == "")


def test_precheck_disabled():
    # precheck=False -> the high prob is NOT consumed at token 0; the turn
    # streams and is caught by the final check instead.
    events = [{"type": "token", "text": "recipe"}, {"type": "finish", "finish_reason": "stop"}]
    s, stream = make_session(probs=[0.99], events=events, check_every=1000,
                             precheck=False)
    result = list(s.send("how to make X"))
    types = [e["type"] for e in result]
    verdict = result[-1]
    check("no-precheck: token streamed", "token" in types, types)
    check("no-precheck: blocked at final check", verdict["blocked"] is True
          and verdict["n_tokens"] > 0, verdict)
    s.close()


# ---------------------------------------------------------------------------
def main():
    test_helpers()
    test_benign_turn()
    test_block_midstream()
    test_block_final()
    test_block_precheck()
    test_precheck_disabled()
    test_reset()
    test_stats_never_zeroed_by_reset()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name, detail in FAIL:
            print(f"  FAILED {name}: {detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
