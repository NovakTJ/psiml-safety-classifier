# REDTEAM-FIRST-TRY — first external red-team run (Agent 1 of 3)

Handover notes from the first of three red-team agents tasked with jailbreaking
`qwen/qwen3.5-9b` through the AEGIS guarded chat while simultaneously fooling
the guard classifier (black-box, blocked/not-blocked signal only).

- `OLD-CLAUDE.md` — the exact task brief the agent was given.
- `CLAUDE.md` — the agent's findings (headline: "documentation register" framing
  comprehensively fools the LoRA guard; **zero-shot Gemma blocks many prompts the
  tuned LoRA passes** — a likely LoRA over-correction regression on
  educational/forensic framing).
- `probe.sh` — the agent's single-message probe helper (wraps `cli.py --jsonl`).

**The entire red-team session cost $2.30 in OpenRouter API spend.**

Supporting evidence: the session logs in `psiml_data/redteam_sessions/` dated
2026-08-14.
