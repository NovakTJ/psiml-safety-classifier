"""
OpenRouter variant of generate_jailbreak_dataset.py. Per CLAUDE.md, OpenRouter is a
completely good option for generation (and much faster — parallel requests); local
`/data/models/Qwen3.5-9B` hf generation also works but is slow (~10 tok/s, no
flash-attn in qwen35_env).

Model MUST be `qwen/qwen3.5-9b` (the guarded model). `qwen/qwen3.5-plus-02-15`
etc. are DIFFERENT models — do not substitute.

CRITICAL: `"reasoning": {"enabled": false}` — without it Qwen3.5 burns the whole
token budget on CoT and returns content="" with finish_reason="length"
(see CLAUDE.md pitfall; the Anthropic-style `thinking: {type: disabled}` is WRONG).

    /home/mls01/ccpp_env/bin/python generate_jailbreak_dataset_openrouter.py \
        --n-prompts 10 --out /home/mls01/psiml_data/jailbreak_v1/raw_qwen35.jsonl

Output schema matches the local script, plus finish_reason per row.
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jailbreak_templates import TEMPLATES, build_messages  # noqa: E402

from generate_jailbreak_dataset import load_harmful_en_prompts, stratify, COMPLETE_DATASET  # noqa: E402

MODEL_ID = "qwen/qwen3.5-9b"  # the guarded model; do NOT substitute other qwen3.5 ids
API_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_api(api_key, messages, max_tokens, temperature, top_p, retries=4):
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "reasoning": {"enabled": False},  # REQUIRED — see docstring
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read())
            choice = body["choices"][0]
            return {
                "content": choice["message"].get("content") or "",
                "finish_reason": choice.get("finish_reason"),
                "error": None,
            }
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt * 5)
                continue
            return {"content": "", "finish_reason": None,
                    "error": f"HTTP {e.code}: {detail}"}
        except Exception as e:  # noqa: BLE001
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 5)
                continue
            return {"content": "", "finish_reason": None, "error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/home/mls01/psiml_data/jailbreak_v1/raw_qwen35.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set (see ~/.bashrc)")

    prompts = load_harmful_en_prompts(COMPLETE_DATASET)
    selected = stratify(prompts, args.n_prompts, seed=args.seed)
    print(f"Selected {len(selected)} prompts across "
          f"{len({s for _, s, _ in selected})} subcategories.", flush=True)

    jobs = []
    for row_id, subcat, harmful_prompt in selected:
        for tpl in TEMPLATES:
            jobs.append((row_id, subcat, harmful_prompt, tpl))

    total = len(jobs)
    results = [None] * total

    def run_one(idx_job):
        idx, (row_id, subcat, harmful_prompt, tpl) = idx_job
        messages = build_messages(tpl["id"], harmful_prompt)
        r = call_api(api_key, messages, args.max_new_tokens,
                     args.temperature, args.top_p)
        rec = {
            "row_id": row_id,
            "subcategory": subcat,
            "prompt": harmful_prompt,
            "template_id": tpl["id"],
            "template_codename": tpl["codename"],
            "response": r["content"].strip(),
            "finish_reason": r["finish_reason"],
            "generation_params": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "reasoning": {"enabled": False},
            },
            "model": f"{MODEL_ID}-openrouter",
        }
        if r["error"]:
            rec["error"] = r["error"]
        return idx, rec

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for idx, rec in ex.map(run_one, enumerate(jobs)):
            results[idx] = rec
            n_done = sum(x is not None for x in results)
            print(f"[{n_done}/{total}] {rec['row_id']} / {rec['template_id']} "
                  f"-> {len(rec['response'])} chars "
                  f"finish={rec['finish_reason']} "
                  f"err={rec.get('error')}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_err = sum(1 for r in results if r.get("error"))
    n_empty = sum(1 for r in results if not r["response"])
    print(f"Done in {time.time()-t0:.0f}s. Wrote {total} records to {args.out} "
          f"({n_err} errors, {n_empty} empty).", flush=True)


if __name__ == "__main__":
    main()
