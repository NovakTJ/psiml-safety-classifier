"""
Label jailbreak-attempt responses with Qwen3Guard-Gen-8B (the exchange classifier).
Run in `ccpp_env` (transformers 4.57.6 can load Qwen3Guard; the qwen3_5 architecture
is NOT needed here).

    /home/mls01/ccpp_env/bin/python label_jailbreak_dataset.py \
        --in /home/mls01/psiml_data/jailbreak_v1/raw_qwen35.jsonl \
        --out /home/mls01/psiml_data/jailbreak_v1/labeled_qwen35.jsonl

Qwen3Guard-Gen-8B is used in its NATIVE response-moderation mode: messages =
[user: prompt, assistant: response]. Its chat template then evaluates THE LAST
ASSISTANT's response and emits three lines: 'Safety: Safe|Unsafe|Controversial',
'Categories: ...', 'Refusal: Yes|No'. (Do NOT wrap the exchange in a single user
message with custom instructions — that triggers the prompt-moderation branch,
which would just classify our always-harmful prompts as Unsafe and tell us nothing
about jailbreak success.)

Output: same rows as input plus guard_raw / guard_safety / guard_categories /
guard_refusal / is_refusal (regex pre-filter; weak signal, not a harm label).
"""

import argparse
import json
import os
import sys

# --- REQUIRED env-var block (must precede transformers import) ---
# Same getpwuid(uid 1562) pitfall as the generation script — uid has no
# /etc/passwd entry on this shared account (see CLAUDE.md).
os.environ.setdefault("USER", "mls01")
os.environ.setdefault("LOGNAME", "mls01")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/home/mls01/.cache/torchinductor")
os.environ.setdefault("TRITON_CACHE_DIR", "/home/mls01/.cache/triton")
os.environ.setdefault("XDG_CACHE_HOME", "/home/mls01/.cache")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jailbreak_templates import is_refusal  # noqa: E402

MODEL_PATH = "/data/models/Qwen3Guard-Gen-8B"


def parse_guard_output(text):
    """Parse the 3-line Qwen3Guard response into (safety, categories, refusal)."""
    safety = categories = refusal = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Safety:"):
            safety = line.split(":", 1)[1].strip()
        elif line.startswith("Categories:"):
            categories = line.split(":", 1)[1].strip()
        elif line.startswith("Refusal:"):
            refusal = line.split(":", 1)[1].strip()
    return safety, categories, refusal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default="/home/mls01/psiml_data/jailbreak_v1/raw_qwen35.jsonl")
    ap.add_argument("--out",
                    default="/home/mls01/psiml_data/jailbreak_v1/labeled_qwen35.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    print(f"Loading Qwen3Guard-Gen-8B from {MODEL_PATH} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Model loaded.", flush=True)

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    print(f"Read {len(rows)} records from {args.inp}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for i, r in enumerate(rows):
            prompt = r.get("prompt", "")
            response = r.get("response", "")
            if not response.strip():
                r["guard_raw"] = None
                r["guard_safety"] = "EMPTY_RESPONSE"
                r["guard_categories"] = None
                r["guard_refusal"] = None
                r["is_refusal"] = False
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{i+1}/{len(rows)}] EMPTY RESPONSE — skipped judge", flush=True)
                continue
            messages = [{"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            out_tokens = generated[0][inputs["input_ids"].shape[1]:]
            guard_output = tokenizer.decode(out_tokens, skip_special_tokens=True).strip()

            safety, categories, refusal = parse_guard_output(guard_output)
            r["guard_raw"] = guard_output
            r["guard_safety"] = safety
            r["guard_categories"] = categories
            r["guard_refusal"] = refusal
            r["is_refusal"] = is_refusal(response)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i+1}/{len(rows)}] safety={safety} refusal={refusal} "
                  f"regex_refusal={r['is_refusal']} len={len(response)}", flush=True)
    print(f"Done. Wrote {len(rows)} records to {args.out}", flush=True)


if __name__ == "__main__":
    main()