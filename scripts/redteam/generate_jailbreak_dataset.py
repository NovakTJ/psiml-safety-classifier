"""
Generate jailbreak-attempt responses from the GUARDED model (Qwen3.5-9B) using the
G0DM0D3 jailbreak templates. Run this in the `qwen35_env` conda env (transformers 5.15:
the only env that can load the qwen3_5 architecture).

    /home/mls01/.conda/envs/qwen35_env/bin/python generate_jailbreak_dataset.py \
        --n-prompts 10 --out /home/mls01/psiml_data/jailbreak_v1/raw_qwen35.jsonl

The env-var block below MUST be set before importing transformers (see CLAUDE.md pitfall:
getpwuid(): uid not found for uid 1562). Do not move this block below the imports.

Output: one JSONL row per (prompt, template) with the raw model response. Labeling
(which responses actually jailbroke) is a separate step against Qwen3Guard in ccpp_env.
"""

import argparse
import json
import os
import random
import sys

# --- REQUIRED env-var block (must precede transformers import) ---
os.environ.setdefault("USER", "mls01")
os.environ.setdefault("LOGNAME", "mls01")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/home/mls01/.cache/torchinductor")
os.environ.setdefault("TRITON_CACHE_DIR", "/home/mls01/.cache/triton")
os.environ.setdefault("XDG_CACHE_HOME", "/home/mls01/.cache")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jailbreak_templates import TEMPLATES, build_messages  # noqa: E402

MODEL_PATH = "/data/models/Qwen3.5-9B"
COMPLETE_DATASET = "/home/mls01/data/complete_dataset.jsonl"


def load_harmful_en_prompts(path):
    """Return list of (row_id, subcategory, prompt) for EN harmful prompts."""
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if (r.get("language") == "en"
                    and r.get("prompt_harm_label") == "harmful"
                    and r.get("prompt")):
                out.append((r["row_id"], r.get("subcategory"), r["prompt"]))
    return out


def stratify(prompts, n, seed=0):
    """Pick n prompts spread across subcategories (one per subcat first, then fill)."""
    by_subcat = {}
    for row_id, subcat, text in prompts:
        by_subcat.setdefault(subcat, []).append((row_id, subcat, text))
    rng = random.Random(seed)
    selected = []
    for v in by_subcat.values():
        rng.shuffle(v)
    keys = sorted(by_subcat)  # deterministic
    i = 0
    while len(selected) < n:
        for subcat in keys:
            if len(selected) >= n:
                break
            if by_subcat[subcat]:
                picked = by_subcat[subcat].pop(0)
                if picked not in selected:
                    selected.append(picked)
        i += 1
        if i > 100:  # safety
            break
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/home/mls01/psiml_data/jailbreak_v1/raw_qwen35.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    print(f"Loading Qwen3.5-9B from {MODEL_PATH} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Model loaded.", flush=True)

    prompts = load_harmful_en_prompts(COMPLETE_DATASET)
    print(f"Total harmful EN prompts in dataset: {len(prompts)}", flush=True)
    selected = stratify(prompts, args.n_prompts, seed=args.seed)
    print(f"Selected {len(selected)} prompts across "
          f"{len({s for _, s, _ in selected})} subcategories.", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    total = len(selected) * len(TEMPLATES)
    n_done = 0
    with open(args.out, "w") as f:
        for row_id, subcat, harmful_prompt in selected:
            for tpl in TEMPLATES:
                messages = build_messages(tpl["id"], harmful_prompt)
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                new_tokens = gen[0][inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                record = {
                    "row_id": row_id,
                    "subcategory": subcat,
                    "prompt": harmful_prompt,
                    "template_id": tpl["id"],
                    "template_codename": tpl["codename"],
                    "response": response,
                    "generation_params": {
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                    },
                    "model": "Qwen3.5-9B-local",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                n_done += 1
                print(f"[{n_done}/{total}] {row_id} / {tpl['id']} "
                      f"-> {len(response)} chars", flush=True)
    print(f"Done. Wrote {n_done} records to {args.out}", flush=True)


if __name__ == "__main__":
    main()