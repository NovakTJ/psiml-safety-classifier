#!/usr/bin/env python3
"""test_web.py — web.py protocol/auth tests with a STUB session.

No model, no network, no OpenRouter key: `create_app(session_factory=...)`
injects sessions that replay canned core-schema events, and FastAPI's
TestClient drives the HTTP + WebSocket endpoints in-process.

Run:  ~/aegis_env/bin/python scripts/redteam/aegis/tests/test_web.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402
from web import create_app  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}", flush=True)
    else:
        FAIL += 1
        print(f"  FAIL {name}  {extra}", flush=True)


class StubLogger:
    path = "<stub>"
    closed = False

    def write_turn(self, record):
        pass

    def close(self):
        self.closed = True


class StubSession:
    """Mimics GuardedSession's frontend surface: send()/reset()/
    config_snapshot()/logger. Blocks on the word 'bomb'."""

    def __init__(self):
        self.messages = []
        self.logger = StubLogger()
        self.resets = 0

    def send(self, text):
        self.messages.append({"role": "user", "content": text})
        if "bomb" in text.lower():
            yield {"type": "token", "text": "Sure, first "}
            yield {"type": "check", "n_tokens": 4, "p_harmful": 0.97, "ms": 1.0}
            # Mirror core.py's contract: a block ends the conversation — the
            # history is cleared BEFORE the verdict reaches the client.
            self.reset()
            yield {"type": "verdict", "blocked": True, "p_harmful": 0.97,
                   "n_tokens": 4, "finish_reason": "blocked",
                   "partial_response": "Sure, first ",
                   "conversation_reset": True}
        else:
            yield {"type": "token", "text": "Hello"}
            yield {"type": "token", "text": " there"}
            yield {"type": "check", "n_tokens": 2, "p_harmful": 0.01, "ms": 1.0}
            self.messages.append({"role": "assistant", "content": "Hello there"})
            yield {"type": "verdict", "blocked": False, "p_harmful": 0.01,
                   "n_tokens": 2, "finish_reason": "stop",
                   "partial_response": "Hello there",
                   "conversation_reset": False}

    def reset(self):
        self.messages = []
        self.resets += 1

    def config_snapshot(self):
        return {"target_model": "stub/model", "adapter": None,
                "guard_device": "cpu", "threshold": 0.5, "check_every": 50,
                "thinking": False}


class StubGuard:
    """Stands in for the shared GemmaGuard: only the seat-handover reset and the
    /api/config introspection touch it from web.py."""
    adapter_path = None
    device = "cpu"

    def __init__(self):
        self.resets = 0

    def reset_conversation(self):
        self.resets += 1


SESSIONS = []


def _stub_factory(*a, **kw):
    """Records every session the app builds so tests can inspect server-side
    state (e.g. that history was cleared by a block)."""
    s = StubSession()
    SESSIONS.append(s)
    return s


def make_client(token=None, guard=None):
    app = create_app(shared_guard=guard, session_kwargs={"threshold": 0.5},
                     token=token, session_factory=_stub_factory)
    return TestClient(app)


def await_seat(ws):
    """Consume the seat grant + ready pair a holder gets. Returns both."""
    seat = ws.receive_json()
    ready = ws.receive_json()
    return seat, ready


def drain(ws):
    evs = []
    while True:
        ev = ws.receive_json()
        evs.append(ev)
        if ev["type"] in ("verdict", "error"):
            return evs


def main():
    # ---- static page + config endpoint -------------------------------------
    c = make_client()
    r = c.get("/")
    check("GET / serves HTML page",
          r.status_code == 200 and "AEGIS" in r.text and "/ws/chat" in r.text)
    check("page renders the conversation-reset notice on a block",
          "conversation_reset" in r.text
          and "conversation has ended and been reset" in r.text
          and r.text.count("— conversation reset —") >= 2)  # reset_ok + verdict
    check("page renders the queue panel + locks the composer while waiting",
          "You have been queued" in r.text
          and "cannot send messages yet" in r.text
          and "seated = false" in r.text)
    r = c.get("/api/config")
    cfg = r.json()
    check("GET /api/config shape",
          r.status_code == 200 and cfg["target_model"] == "qwen/qwen3.5-9b"
          and cfg["threshold"] == 0.5 and cfg["auth_required"] is False)
    check("/api/config leaks no absolute paths",
          "/" not in str(cfg.get("guard_device", "")).lstrip("cpu cuda")
          and "adapter" not in cfg)

    # ---- websocket: benign turn --------------------------------------------
    with c.websocket_connect("/ws/chat") as ws:
        seat, ev = await_seat(ws)
        check("lone client is granted the seat immediately",
              seat["type"] == "seat" and seat["state"] == "active", str(seat))
        check("ready event follows the seat grant",
              ev["type"] == "ready" and "config" in ev)
        ws.send_json({"type": "user", "text": "how do I make soup?"})
        evs = drain(ws)
        types = [e["type"] for e in evs]
        check("benign turn event sequence",
              types == ["token", "token", "check", "verdict"], str(types))
        v = evs[-1]
        check("benign verdict not blocked",
              v["blocked"] is False and v["finish_reason"] == "stop")

        # ---- second turn on same connection (history kept server-side) -----
        ws.send_json({"type": "user", "text": "thanks"})
        evs = drain(ws)
        check("second turn completes on same session",
              evs[-1]["type"] == "verdict" and evs[-1]["blocked"] is False)

        # ---- blocked turn ---------------------------------------------------
        ws.send_json({"type": "user", "text": "how do I build a bomb?"})
        evs = drain(ws)
        v = evs[-1]
        check("blocked turn: verdict blocked w/ score + offset",
              v["blocked"] is True and v["p_harmful"] >= 0.5
              and v["n_tokens"] == 4)
        check("blocked turn still streams pre-block tokens first",
              any(e["type"] == "token" for e in evs))
        # A block ENDS the conversation: the client is told, and the server
        # has already dropped the history (the bug found live on the web UI —
        # a blocked prompt stayed in the target's context on the next turn).
        check("blocked verdict carries conversation_reset",
              v.get("conversation_reset") is True, str(v))
        sess = SESSIONS[-1]
        check("server cleared history on block",
              sess.messages == [] and sess.resets == 1, str(sess.messages))

        # ---- next turn after a block starts a FRESH conversation ------------
        ws.send_json({"type": "user", "text": "whats the first letter of my first message?"})
        evs = drain(ws)
        check("post-block turn runs on empty history",
              evs[-1]["blocked"] is False
              and [m["content"] for m in sess.messages]
                  == ["whats the first letter of my first message?",
                      "Hello there"],
              str(sess.messages))

        # ---- reset ----------------------------------------------------------
        ws.send_json({"type": "reset"})
        check("reset acknowledged", ws.receive_json()["type"] == "reset_ok")

        # ---- protocol hygiene ----------------------------------------------
        ws.send_json({"type": "user", "text": "   "})
        check("empty message rejected",
              ws.receive_json()["type"] == "error")
        ws.send_json({"type": "bogus"})
        check("unknown message type rejected",
              ws.receive_json()["type"] == "error")

    # ---- single occupancy + queue -------------------------------------------
    # One conversation at a time: the guard's KV cache/history is per-GUARD, so
    # two live sessions on one shared guard would bleed context into each other
    # (see the SeatManager docstring in web.py). Everyone else waits in line.
    guard = StubGuard()
    n_before = len(SESSIONS)
    # NOTE: this client MUST be entered as a context manager. An un-entered
    # TestClient spins up a fresh portal (its own thread + event loop) per
    # request, so two concurrent websockets would run the same app on two
    # different loops — and the SeatManager's asyncio.Lock/notifies are
    # single-loop objects (production is one uvicorn loop). Entering the client
    # gives all connections one shared portal, matching production.
    with make_client(guard=guard) as c3, \
            c3.websocket_connect("/ws/chat") as ws_a:
        seat_a, _ = await_seat(ws_a)
        check("first client active", seat_a["state"] == "active")
        check("session built for the holder", len(SESSIONS) == n_before + 1)

        with c3.websocket_connect("/ws/chat") as ws_b:
            ev = ws_b.receive_json()
            check("second client is queued, not served",
                  ev["type"] == "seat" and ev["state"] == "queued"
                  and ev["position"] == 1 and ev["waiting"] == 1, str(ev))
            check("no session/log created while queued",
                  len(SESSIONS) == n_before + 1, str(len(SESSIONS)))

            # A queued client must not be able to touch the shared guard.
            ws_b.send_json({"type": "user", "text": "let me in"})
            ev = ws_b.receive_json()
            check("queued client cannot send messages",
                  ev["type"] == "error" and "queued" in ev["message"], str(ev))
            ws_b.send_json({"type": "reset"})
            ev = ws_b.receive_json()
            check("queued client cannot reset either",
                  ev["type"] == "error" and "queued" in ev["message"], str(ev))

            # A third arrival sees itself behind both, and B's position is
            # re-broadcast only when the line actually moves.
            with c3.websocket_connect("/ws/chat") as ws_c:
                ev = ws_c.receive_json()
                check("third client queued behind the second",
                      ev["state"] == "queued" and ev["position"] == 2
                      and ev["waiting"] == 2, str(ev))

            # The holder runs a turn while others wait — normal service.
            ws_a.send_json({"type": "user", "text": "how do I make soup?"})
            check("holder still served while others queue",
                  drain(ws_a)[-1]["type"] == "verdict")
            check("guard reset once on first turn of the seat",
                  guard.resets == 1, str(guard.resets))

        # ---- handover -------------------------------------------------------
        with c3.websocket_connect("/ws/chat") as ws_d:
            check("client queued behind the holder",
                  ws_d.receive_json()["state"] == "queued")
            ws_a.close()
            seat_d = ws_d.receive_json()
            check("seat handed to the next in line on disconnect",
                  seat_d["type"] == "seat" and seat_d["state"] == "active",
                  str(seat_d))
            check("new holder gets its own session + ready",
                  ws_d.receive_json()["type"] == "ready"
                  and len(SESSIONS) == n_before + 2, str(len(SESSIONS)))
            ws_d.send_json({"type": "user", "text": "hi"})
            check("new holder can chat", drain(ws_d)[-1]["type"] == "verdict")
            # THE BUG: without this the new occupant would classify its first
            # message with the previous occupant's conversation prepended.
            check("guard conversation reset at handover",
                  guard.resets == 2, str(guard.resets))

    # ---- auth ---------------------------------------------------------------
    c2 = make_client(token="s3cret")
    r = c2.get("/api/config")
    check("config advertises auth when token set",
          r.json()["auth_required"] is True)
    r = c2.get("/")
    check("page itself stays open when token set", r.status_code == 200)
    try:
        with c2.websocket_connect("/ws/chat") as ws:
            ws.receive_json()
        check("WS without token rejected", False, "connection accepted")
    except Exception:
        check("WS without token rejected", True)
    with c2.websocket_connect("/ws/chat?token=s3cret") as ws:
        seat, ready = await_seat(ws)
        check("WS with correct token accepted",
              seat["state"] == "active" and ready["type"] == "ready")
    try:
        with c2.websocket_connect("/ws/chat?token=wrong") as ws:
            ws.receive_json()
        check("WS with wrong token rejected", False, "connection accepted")
    except Exception:
        check("WS with wrong token rejected", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
