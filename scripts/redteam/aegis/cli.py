#!/usr/bin/env python3
"""aegis-chat — CLI frontend for the AEGIS guarded-chat engine (PLAN.md frontend 1).

Two modes:
  * Human REPL (default): streams tokens live, dim inline check scores, verdict
    banner on block. Slash commands: /reset /thinking on|off /threshold F
    /check-every N /stats /config /help /quit.
  * Machine mode (--jsonl): one user message per stdin line, one JSON event per
    stdout line (core.py event schema), stateful between lines, exits on EOF.
    This is the agent/test-harness interface.

Examples:
  ~/aegis_env/bin/python scripts/redteam/aegis/cli.py --device cpu
  ~/aegis_env/bin/python scripts/redteam/aegis/cli.py --jsonl --device cpu <<EOF
  Hello there
  EOF
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (GuardedSession, DEFAULT_ADAPTER, DEFAULT_LOG_DIR,  # noqa: E402
                  MODEL_ID, REPO_ROOT)

# --- terminal colors (auto-disabled when not a tty) -------------------------
_TTY = sys.stdout.isatty()


def _c(code, text):
    return f"\x1b[{code}m{text}\x1b[0m" if _TTY else text


def dim(t):
    return _c("2", t)


def red(t):
    return _c("1;31", t)


def green(t):
    return _c("1;32", t)


def yellow(t):
    return _c("1;33", t)


def load_dotenv(path):
    """Minimal .env loader (KEY=VALUE lines; no override of existing env)."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
def render_event_repl(ev, state):
    """Print one event in human mode. `state` tracks whether we're mid-line.

    The guard's P(harmful) score is NEVER shown (realistic deployment: the
    user never sees guard internals; scores remain in the session log).
    """
    t = ev.get("type")
    if t == "token":
        print(ev["text"], end="", flush=True)
        state["mid_line"] = True
    elif t == "reasoning":
        print(dim(ev["text"]), end="", flush=True)
        state["mid_line"] = True
    elif t == "check":
        print(dim(f"\n  [check tok={ev['n_tokens']} "
                  f"({ev['ms']:.0f} ms)]"), flush=True)
        state["mid_line"] = False
    elif t == "verdict":
        if state["mid_line"]:
            print()
        if ev["blocked"]:
            print(red(f"\n  *** BLOCKED by safety guard at token {ev['n_tokens']}"
                      f" ***"))
            print(dim("  (partial response withheld from model history; "
                      "real partial is in the session log)"))
            if ev.get("conversation_reset"):
                print(yellow("  — conversation ended and reset — the next "
                             "message starts a NEW conversation; the target "
                             "model no longer has this exchange in context"))
        else:
            print(green(f"\n  [ok] finish_reason={ev['finish_reason']} "
                        f"({ev['n_tokens']} tokens classified)"))
        state["mid_line"] = False
    elif t == "error":
        print(red(f"\n  [error] {ev['message']}"), flush=True)
        state["mid_line"] = False


HELP = """\
commands:
  /reset             start a new conversation (model history cleared)
  /thinking on|off   toggle OpenRouter reasoning mode (default off)
  /threshold F       block threshold on P(harmful), e.g. /threshold 0.7
  /check-every N     Gemma tokens between guard checks
  /stats             checks run, blocks, mean check latency
  /config            show current session config
  /help              this text
  /quit              exit
Anything else is sent as a user message to the guarded model."""


def repl(session):
    cfg = session.config_snapshot()
    print(_c("1", "aegis-chat") + dim(f" — target={cfg['target_model']} "
          f"guard=gemma-3-1b-it@{cfg['guard_device']} "
          f"adapter={cfg['adapter'] or 'NONE (zero-shot)'} threshold={cfg['threshold']}"))
    print(dim("  /help for commands; Ctrl-D or /quit to exit"))
    while True:
        try:
            line = input(_c("1;36", "\nyou> ") if _TTY else "you> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print(dim("\n(interrupted — /quit to exit)"))
            continue
        if not line:
            continue
        if line.startswith("/"):
            parts = line.split(None, 1)
            cmd, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
            try:
                if cmd in ("/quit", "/exit"):
                    break
                elif cmd == "/reset":
                    session.reset()
                    print(dim("  conversation reset"))
                elif cmd == "/thinking":
                    if arg not in ("on", "off"):
                        print(yellow("  usage: /thinking on|off"))
                    else:
                        session.thinking = arg == "on"
                        print(dim(f"  thinking={arg}"))
                elif cmd == "/threshold":
                    session.threshold = float(arg)
                    print(dim(f"  threshold={session.threshold}"))
                elif cmd == "/check-every":
                    session.check_every = int(arg)
                    print(dim(f"  check_every={session.check_every}"))
                elif cmd == "/stats":
                    print(dim("  " + json.dumps(session.stats())))
                elif cmd == "/config":
                    print(dim("  " + json.dumps(session.config_snapshot(), indent=2)))
                elif cmd == "/help":
                    print(HELP)
                else:
                    print(yellow(f"  unknown command {cmd} (/help)"))
            except ValueError:
                print(yellow(f"  bad argument: {arg!r}"))
            continue
        state = {"mid_line": False}
        try:
            for ev in session.send(line):
                render_event_repl(ev, state)
        except KeyboardInterrupt:
            print(yellow("\n  (turn interrupted — stream may be incomplete)"))
        print()


def jsonl_mode(session):
    """One user message per stdin line; JSON events on stdout.

    The guard's P(harmful) score is stripped from every event (the caller
    never sees guard internals; scores remain in the session log).
    """
    for line in sys.stdin:
        msg = line.rstrip("\n")
        if not msg.strip():
            continue
        for ev in session.send(msg):
            ev = {k: v for k, v in ev.items() if k != "p_harmful"}
            sys.stdout.write(json.dumps(ev, ensure_ascii=False) + "\n")
            sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(prog="aegis-chat", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter path (default: locked phase-2 winner; "
                         "'none' = bare zero-shot Gemma)")
    ap.add_argument("--guard-prompt", default=None,
                    help="classifier instruction text (default: training PROMPT_1; "
                         "'@path' reads from a file)")
    ap.add_argument("--model", default=None,
                    help=f"OpenRouter target model (default: {MODEL_ID})")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    ap.add_argument("--check-every", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-precheck", action="store_true",
                    help="disable the token-0 prompt pre-check (ablation knob; "
                         "default: pre-check ON — harmful prompts are blocked "
                         "before the target model is ever called)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    ap.add_argument("--jsonl", action="store_true", help="machine stdio mode")
    args = ap.parse_args()

    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("error: OPENROUTER_API_KEY not set (repo .env or environment)",
              file=sys.stderr)
        sys.exit(2)

    guard_prompt = args.guard_prompt
    if guard_prompt and guard_prompt.startswith("@"):
        with open(guard_prompt[1:], encoding="utf-8") as f:
            guard_prompt = f.read().strip()

    session = GuardedSession(
        adapter=args.adapter, device=args.device, thinking=args.thinking == "on",
        check_every=args.check_every, threshold=args.threshold,
        temperature=args.temperature, max_tokens=args.max_tokens,
        log_dir=args.log_dir, model=args.model, guard_prompt=guard_prompt,
        precheck=not args.no_precheck)
    try:
        if args.jsonl:
            jsonl_mode(session)
        else:
            repl(session)
    finally:
        print(dim(f"\n[session log: {session.logger.path}]"), file=sys.stderr)
        session.close()


if __name__ == "__main__":
    main()
