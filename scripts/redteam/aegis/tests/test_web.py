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
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {extra}")


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
            yield {"type": "verdict", "blocked": True, "p_harmful": 0.97,
                   "n_tokens": 4, "finish_reason": "blocked",
                   "partial_response": "Sure, first "}
        else:
            yield {"type": "token", "text": "Hello"}
            yield {"type": "token", "text": " there"}
            yield {"type": "check", "n_tokens": 2, "p_harmful": 0.01, "ms": 1.0}
            yield {"type": "verdict", "blocked": False, "p_harmful": 0.01,
                   "n_tokens": 2, "finish_reason": "stop",
                   "partial_response": "Hello there"}

    def reset(self):
        self.messages = []
        self.resets += 1

    def config_snapshot(self):
        return {"target_model": "stub/model", "adapter": None,
                "guard_device": "cpu", "threshold": 0.5, "check_every": 50,
                "thinking": False}


def make_client(token=None):
    app = create_app(shared_guard=None, session_kwargs={"threshold": 0.5},
                     token=token, session_factory=StubSession)
    return TestClient(app)


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
        ev = ws.receive_json()
        check("ready event first on connect",
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
        check("WS with correct token accepted",
              ws.receive_json()["type"] == "ready")
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
