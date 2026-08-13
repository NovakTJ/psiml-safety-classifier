"""
Round-2 jailbreak generation: scale up successful jailbreaks beyond the round-1
pilot (6/60). Round 1 showed only two of the six G0DM0D3 templates ever succeed
against Qwen3.5-9B — grok-420 (4/10) and hermes-fast (2/10) — so this round runs
ONLY those two templates, on fresh prompts (round-1 row_ids excluded), across ALL
languages (round 1 was English-only; multilingual safety tuning is typically
weaker, so non-EN prompts are a promising attack surface).

    /home/mls01/ccpp_env/bin/python generate_jailbreak_round2.py \
        --n-non-en 40 --n-en 20 \
        --out /home/mls01/psiml_data/jailbreak_v1/raw_qwen35_round2.jsonl

Same OpenRouter requirements as generate_jailbreak_dataset_openrouter.py:
model MUST be qwen/qwen3.5-9b, reasoning MUST be disabled. max_tokens raised to
3072 (2/6 round-1 successes were truncated at finish_reason=length).
"""

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jailbreak_templates import TEMPLATES, build_messages  # noqa: E402
from generate_jailbreak_dataset import COMPLETE_DATASET, stratify  # noqa: E402
from generate_jailbreak_dataset_openrouter import call_api, MODEL_ID  # noqa: E402

# Use the labeled files (clean rows), NOT raw_qwen35.jsonl — the raw file
# still contains one corrupted line from the killed local process interleaving writes.
PREV_LABELED = [
    "/home/mls01/psiml_data/jailbreak_v1/labeled_qwen35.jsonl",
    "/home/mls01/psiml_data/jailbreak_v1/labeled_qwen35_round2.jsonl",
]
SUCCESSFUL_TEMPLATES = ("grok-420", "hermes-fast")  # only these ever worked in round 1


def load_harmful_prompts_any_lang(path, exclude_row_ids):
    """Return list of (row_id, language, subcategory, prompt) for harmful prompts."""
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if (r.get("prompt_harm_label") == "harmful"
                    and r.get("prompt")
                    and r["row_id"] not in exclude_row_ids):
                out.append((r["row_id"], r.get("language"),
                            r.get("subcategory"), r["prompt"]))
    return out


def stratify_by_language(prompts, n, seed=0):
    """Round-robin across languages (deterministic order, shuffled within)."""
    by_lang = {}
    for p in prompts:
        by_lang.setdefault(p[1], []).append(p)
    rng = random.Random(seed)
    for v in by_lang.values():
        rng.shuffle(v)
    keys = sorted(by_lang)
    selected = []
    while len(selected) < n and any(by_lang[k] for k in keys):
        for lang in keys:
            if len(selected) >= n:
                break
            if by_lang[lang]:
                selected.append(by_lang[lang].pop(0))
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-non-en", type=int, default=40)
    ap.add_argument("--n-en", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out",
                    default="/home/mls01/psiml_data/jailbreak_v1/raw_qwen35_round2.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--concurrency", type=int, default=10)
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not set (see ~/.bashrc)")

    used = set()
    for path in PREV_LABELED:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                used.add(json.loads(line)["row_id"])
    print(f"Excluding {len(used)} previously-used prompts.", flush=True)

    all_prompts = load_harmful_prompts_any_lang(COMPLETE_DATASET, used)
    en = [p for p in all_prompts if p[1] == "en"]
    non_en = [p for p in all_prompts if p[1] != "en"]
    print(f"Available fresh harmful prompts: {len(en)} EN, {len(non_en)} non-EN.",
          flush=True)

    sel_non_en = stratify_by_language(non_en, args.n_non_en, seed=args.seed)
    # reuse round-1 subcategory stratifier for EN (it expects (row_id, subcat, text))
    sel_en = stratify([(r, s, t) for r, _, s, t in en], args.n_en, seed=args.seed)
    sel_en = [(r, "en", s, t) for r, s, t in sel_en]
    selected = sel_non_en + sel_en
    langs = sorted({p[1] for p in selected})
    print(f"Selected {len(selected)} prompts "
          f"({len(sel_non_en)} non-EN across {len(langs) - 1} languages, "
          f"{len(sel_en)} EN).", flush=True)

    templates = [t for t in TEMPLATES if t["id"] in SUCCESSFUL_TEMPLATES]
    jobs = [(p, tpl) for p in selected for tpl in templates]
    total = len(jobs)
    results = [None] * total

    def run_one(idx_job):
        idx, ((row_id, lang, subcat, harmful_prompt), tpl) = idx_job
        messages = build_messages(tpl["id"], harmful_prompt)
        r = call_api(api_key, messages, args.max_new_tokens,
                     args.temperature, args.top_p)
        rec = {
            "row_id": row_id,
            "language": lang,
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
            print(f"[{n_done}/{total}] {rec['row_id']} ({rec['language']}) / "
                  f"{rec['template_id']} -> {len(rec['response'])} chars "
                  f"finish={rec['finish_reason']} err={rec.get('error')}",
                  flush=True)

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
