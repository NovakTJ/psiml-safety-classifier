"""
Replay the 38 successful OpenRouter jailbreaks (psiml_data/jailbreak_v1/
successful_jailbreaks.jsonl) against LOCAL /data/models/Qwen3.5-9B and capture
residual-stream activations for linear-probe training.

Why: the probe guards the LOCAL Qwen3.5-9B, so training activations must come
from its own forward passes, not OpenRouter's. Sampling (temp 1.0) means not
every jailbreak will succeed again — that is fine: under the EXCHANGE-classifier
label rule a harmful prompt makes the exchange HARMFUL regardless of the
response, so refusals are still positive training rows. Qwen3Guard labels are
kept as metadata (run label_jailbreak_dataset.py afterwards) so we can later
slice performance by scenario (prompt-driven vs response-driven harm) without
re-generating.

Run with qwen35_env (transformers 5.15.0 — the ONLY env that loads qwen3_5):

    /home/mls01/.conda/envs/qwen35_env/bin/python replay_jailbreaks_local.py

Then label (ccpp_env):

    /home/mls01/ccpp_env/bin/python label_jailbreak_dataset.py \
        --in  /home/mls01/psiml_data/probes/local_replay/raw_local.jsonl \
        --out /home/mls01/psiml_data/probes/local_replay/labeled_local.jsonl

Artifacts (psiml_data/probes/local_replay/ — NOT /tmp, per CLAUDE.md):
- raw_local.jsonl            one row per replay: original metadata, local response,
                             generation params (incl. seed), token counts, guard
                             labels from the ORIGINAL OpenRouter run (openrouter_*),
                             and the activation-file metadata below.
- activations/{row_id}.npz   "hidden": [n_layers, n_tokens, 4096] fp16 residual
                             stream at CAPTURE_LAYERS, "token_ids": int32.
                             Span = last PROMPT_TAIL prompt tokens + full response,
                             so the 16-token probe window can cross the boundary
                             and prefix truncation needs no model re-run.

Disk: 8 layers x 4096 x fp16 = 64 KB/token; ~2.5k-token median response ->
~160 MB/row, ~6 GB total for 38 rows. (All 32 layers would be ~24 GB; capture
is only ~1-2 s/row, so re-capturing more layers later is cheap.)

Resume: rows already present in raw_local.jsonl are skipped (activations are
rewritten only if the .npz is missing). Single-threaded writes + flush — the
round-1 corrupted-line incident came from concurrent writers, not this pattern.
"""

import argparse
import json
import os
import sys
import time

# --- REQUIRED env-var block (must precede torch/transformers import) ---
# uid 1562 has no /etc/passwd entry; getpwuid() crashes cache-dir resolution
# (see CLAUDE.md "Required env-var block").
os.environ.setdefault("USER", "mls01")
os.environ.setdefault("LOGNAME", "mls01")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/home/mls01/.cache/torchinductor")
os.environ.setdefault("TRITON_CACHE_DIR", "/home/mls01/.cache/triton")
os.environ.setdefault("XDG_CACHE_HOME", "/home/mls01/.cache")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jailbreak_templates import build_messages  # noqa: E402

MODEL_PATH = "/data/models/Qwen3.5-9B"
IN_PATH = "/home/mls01/psiml_data/jailbreak_v1/successful_jailbreaks.jsonl"
OUT_DIR = "/home/mls01/psiml_data/probes/local_replay"
OUT_PATH = os.path.join(OUT_DIR, "raw_local.jsonl")
ACT_DIR = os.path.join(OUT_DIR, "activations")

# Layers whose residual stream we save. Qwen3.5-9B has 32 layers; these are the
# 8 full-attention layers (every 4th, incl. the final layer) — a good probe
# sweep grid. hidden_states[l + 1] is the OUTPUT of layer l.
CAPTURE_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
PROMPT_TAIL = 64          # prompt tokens kept before the response span
BATCH_SIZE = 6
BASE_SEED = 20260813      # per-BATCH seed (batched sampling shares one RNG)

# Copied from the OpenRouter generation_params of the source rows.
GEN_KWARGS = dict(max_new_tokens=2048, do_sample=True, temperature=1.0, top_p=0.95)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_done_row_ids():
    done = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(json.loads(line)["row_id"])
                    except json.JSONDecodeError:
                        log("WARNING: skipping corrupt line in existing raw_local.jsonl")
    return done


def capture_activations(model, row_id, full_ids):
    """One teacher-forcing pass over prompt+response; save selected layers fp16.

    full_ids: LongTensor [1, seq_len] on model device.
    Returns (n_tokens_saved, ) — span is [span_start, seq_len) where
    span_start = max(0, prompt_len - PROMPT_TAIL) is computed by the caller and
    encoded in full_ids already being the FULL sequence; we slice here.
    """
    seq_len = full_ids.shape[1]
    span_start = max(0, seq_len - RESP_LEN_GLOBAL[0] - PROMPT_TAIL)
    with torch.inference_mode():
        try:
            out = model(full_ids, output_hidden_states=True, logits_to_keep=1)
        except TypeError:  # older arg name / unsupported kwarg
            out = model(full_ids, output_hidden_states=True)
    hs = out.hidden_states  # tuple of 33: [0]=embeddings, [l+1]=output of layer l
    layers = np.stack(
        [hs[l + 1][0, span_start:].to(torch.float16).cpu().numpy()
         for l in CAPTURE_LAYERS])
    token_ids = full_ids[0, span_start:].to(torch.int32).cpu().numpy()
    path = os.path.join(ACT_DIR, f"{row_id}.npz")
    np.savez(path, hidden=layers, token_ids=token_ids)
    return path, span_start


# capture_activations needs the response length of the row being processed;
# passed via this one-cell list to keep the function signature trivial.
RESP_LEN_GLOBAL = [0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=IN_PATH)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    out_path = os.path.join(args.out_dir, "raw_local.jsonl")
    act_dir = os.path.join(args.out_dir, "activations")
    os.makedirs(act_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    done = load_done_row_ids() if args.out_dir == OUT_DIR else set()
    todo = [r for r in rows if r["row_id"] not in done]
    log(f"{len(rows)} source rows, {len(done)} already done, {len(todo)} to replay")
    if not todo:
        return

    log(f"Loading {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # batched generation
    log("Model loaded.")

    with open(out_path, "a") as out_f:
        for b0 in range(0, len(todo), args.batch_size):
            batch = todo[b0:b0 + args.batch_size]
            batch_idx = b0 // args.batch_size
            seed = BASE_SEED + batch_idx
            torch.manual_seed(seed)

            prompts, prompt_ids_list = [], []
            for r in batch:
                messages = build_messages(r["template_id"], r["prompt"])
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
                prompts.append(text)
                prompt_ids_list.append(
                    tokenizer(text, return_tensors="pt",
                              add_special_tokens=False)["input_ids"][0])

            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
            prompt_lens = enc["attention_mask"].sum(1).tolist()
            log(f"batch {batch_idx}: {len(batch)} rows, "
                f"prompt tokens {min(prompt_lens)}..{max(prompt_lens)}, seed {seed}")

            t0 = time.time()
            with torch.inference_mode():
                # Stop on BOTH <|im_end|> (chat turn end, 248046) and
                # <|endoftext|> (248044, also the pad token). Without this the
                # model sails past <|im_end|> (see raw_local.jsonl fixups).
                gen = model.generate(
                    **enc, **GEN_KWARGS,
                    eos_token_id=[tokenizer.pad_token_id, tokenizer.eos_token_id])
            log(f"batch {batch_idx}: generated in {time.time()-t0:.0f}s")

            for i, r in enumerate(batch):
                n_prompt = prompt_lens[i]
                gen_row = gen[i][enc["input_ids"].shape[1]:]  # generated part
                # In a left-padded batch, a row that finished early has its tail
                # filled with pad => detect stop BEFORE stripping anything.
                stopped_early = (len(gen_row) > 0
                                 and gen_row[-1].item() == tokenizer.pad_token_id)
                # strip pad tokens (fill after early stop)
                gen_row = gen_row[gen_row != tokenizer.pad_token_id]
                # drop everything after the first eos; keep the eos itself
                eos_hits = [j for j, t in enumerate(gen_row.tolist())
                            if t in (tokenizer.pad_token_id, tokenizer.eos_token_id)]
                if eos_hits:
                    gen_row = gen_row[:eos_hits[0] + 1]
                finish = "stop" if (stopped_early or eos_hits) else "length"
                kept = gen_row
                response = tokenizer.decode(kept, skip_special_tokens=True)

                prompt_ids = prompt_ids_list[i]
                assert prompt_ids.shape[0] == n_prompt, \
                    f"prompt-length mismatch for {r['row_id']}"
                full_ids = torch.cat(
                    [prompt_ids, kept.cpu()]).unsqueeze(0).to(model.device)
                RESP_LEN_GLOBAL[0] = kept.shape[0]
                npz_path, span_start = capture_activations(
                    model, r["row_id"], full_ids)

                rec = {
                    "row_id": r["row_id"],
                    "subcategory": r.get("subcategory"),
                    "language": r.get("language"),
                    "prompt": r["prompt"],
                    "template_id": r["template_id"],
                    "template_codename": r.get("template_codename"),
                    "model": "local:/data/models/Qwen3.5-9B",
                    "generation_params": {**GEN_KWARGS, "enable_thinking": False,
                                          "seed": seed, "batch_idx": batch_idx},
                    "response": response,
                    "finish_reason": finish,
                    "n_prompt_tokens": n_prompt,
                    "n_response_tokens": int(kept.shape[0]),
                    # exchange-classifier label: harmful prompt => HARMFUL
                    # regardless of response; guard fields below are metadata
                    # for later scenario slicing, NOT the training label.
                    "exchange_label": "harmful",
                    "openrouter_guard_safety": r.get("guard_safety"),
                    "openrouter_guard_refusal": r.get("guard_refusal"),
                    "activations_file": os.path.relpath(npz_path, args.out_dir),
                    "activations_layers": CAPTURE_LAYERS,
                    "activations_span_start_in_full_seq": span_start,
                    "activations_response_offset": n_prompt - span_start,
                    "activations_prompt_tail": PROMPT_TAIL,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                log(f"  {r['row_id']} [{r['template_id']}] {finish} "
                    f"resp={kept.shape[0]}tok act={os.path.getsize(npz_path)//2**20}MB")

    log("DONE")


if __name__ == "__main__":
    main()
