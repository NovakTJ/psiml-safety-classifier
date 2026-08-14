#!/bin/bash
# usage: ./probe.sh "message" [model] -> prints tokens + verdict only
MSG="$1"
MODEL="${2:-}"
ARGS=()
[ -n "$MODEL" ] && ARGS+=(--model "$MODEL")
cd ~/psiml-safety-classifier
printf '%s\n' "$MSG" | timeout 300 ~/aegis_env/bin/python scripts/redteam/aegis/cli.py --jsonl --device cpu "${ARGS[@]}" 2>/dev/null \
  | ~/aegis_env/bin/python -c '
import sys, json
resp = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("{"): print(line); continue
    ev = json.loads(line)
    if ev["type"] == "token": resp.append(ev["text"])
    elif ev["type"] == "verdict":
        print("RESPONSE:", "".join(resp)[:1500])
        print("VERDICT:", json.dumps({k: v for k, v in ev.items() if k != "partial_response"}))
'
