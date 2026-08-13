"""
Capture Qwen3.5-9B activations for PLAIN (no-jailbreak-template) prompts from
data/complete_dataset.jsonl — the complement to replay_jailbreaks_local.py:

- harmful prompts, plain user turn => the local model refuses (OpenRouter probe:
  0/30 adversarial harmful prompts succeeded without templates, so refusal is
  near-certain). Under the exchange-classifier rule these are still
  exchange_label="harmful" (harmful prompt), and they are the deployment-time
  "guarded model resists" distribution for the probe.
- benign prompts, plain user turn => exchange_label="safe".

No Qwen3Guard labeling (per user instruction 2026-08-13) — guard fields are
written as null; run label_jailbreak_dataset.py later if scenario slicing is
wanted (prompt_harm_label already gives the exchange label).

Row selection: excludes every row_id used in jailbreak rounds 1-3 (those are
captured WITH templates in psiml_data/probes/local_replay/). 125 harmful
(adversarial=False only — near-certain refusals) + 125 benign (both adversarial
flags), each stratified across languages (round-robin by language, EN included),
seed 42. NOTE: row_id prefix (orig/trans/weird) is the augmentation TYPE, not
adversarialness — use the `adversarial` boolean (trans = translation, ~half of
every prefix is adversarial=False; benign is ~half non-EN across 31 languages).

Run with qwen35_env (transformers 5.15.0 — only env that loads qwen3_5):

    /home/mls01/.conda/envs/qwen35_env/bin/python capture_plain_activations.py

Artifacts in psiml_data/probes/plain_prompts/ (activations/ is gitignored):
raw_plain.jsonl + activations/{row_id}.npz, same npz schema as local_replay
(fp16 hidden [8 layers, n_tokens, 4096] @ layers 3,7,11,15,19,23,27,31;
span = last 64 prompt tokens + full response).
"""

import json
import os
import sys
import time

# --- REQUIRED env-var block (must precede torch/transformers import) ---
os.environ.setdefault("USER", "mls01")
os.environ.setdefault("LOGNAME", "mls01")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/home/mls01/.cache/torchinductor")
os.environ.setdefault("TRITON_CACHE_DIR", "/home/mls01/.cache/triton")
os.environ.setdefault("XDG_CACHE_HOME", "/home/mls01/.cache")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_PATH = "/data/models/Qwen3.5-9B"
DATASET = "/home/mls01/data/complete_dataset.jsonl"
JAILBREAK_DIR = "/home/mls01/psiml_data/jailbreak_v1"
OUT_DIR = "/home/mls01/psiml_data/probes/plain_prompts"
OUT_PATH = os.path.join(OUT_DIR, "raw_plain.jsonl")
ACT_DIR = os.path.join(OUT_DIR, "activations")

CAPTURE_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]   # full-attention layers (of 32)
PROMPT_TAIL = 64
BATCH_SIZE = 8
BASE_SEED = 20260814
SELECT_SEED = 42
N_HARM, N_BENIGN = 125, 125

# Same sampling params as the jailbreak replay, for a consistent capture protocol.
GEN_KWARGS = dict(max_new_tokens=2048, do_sample=True, temperature=1.0, top_p=0.95)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def select_rows():
    used = set()
    for fn in ["raw_qwen35.jsonl", "raw_qwen35_round2.jsonl",
               "raw_qwen35_round3.jsonl"]:
        for line in open(os.path.join(JAILBREAK_DIR, fn)):
            if line.strip():
                try:
                    used.add(json.loads(line)["row_id"])
                except json.JSONDecodeError:
                    pass  # round-1 raw file has one known corrupt line
    rows = [json.loads(l) for l in open(DATASET) if l.strip()]
    rows = [r for r in rows if r["prompt"].strip() and r["row_id"] not in used]
    harm = [r for r in rows if r["prompt_harm_label"] == "harmful"
            and r["adversarial"] is False]
    benign = [r for r in rows if r["prompt_harm_label"] == "unharmful"]
    rng = np.random.default_rng(SELECT_SEED)

    def stratified_by_language(pool, n):
        # EN gets its proportional pool share; the rest is round-robin across
        # non-EN languages (pure round-robin over all languages would shrink
        # EN to ~1/32 of the sample — not deployment-realistic).
        by_lang = {}
        for r in pool:
            by_lang.setdefault(r.get("language") or "?", []).append(r)
        for v in by_lang.values():
            rng.shuffle(v)
        n_en = round(n * len(by_lang.get("en", [])) / len(pool))
        picked = by_lang.get("en", [])[:n_en]
        non_en = sorted(l for l in by_lang if l != "en")
        i = 0
        while len(picked) < n and any(by_lang[l] for l in non_en):
            l = non_en[i % len(non_en)]
            if by_lang[l]:
                picked.append(by_lang[l].pop())
            i += 1
        return picked[:n]

    sel = stratified_by_language(harm, N_HARM) \
        + stratified_by_language(benign, N_BENIGN)
    rng.shuffle(sel)
    return sel


def load_done_row_ids():
    done = set()
    if os.path.exists(OUT_PATH):
        for line in open(OUT_PATH):
            if line.strip():
                try:
                    done.add(json.loads(line)["row_id"])
                except json.JSONDecodeError:
                    log("WARNING: skipping corrupt line in existing raw_plain.jsonl")
    return done


def capture_activations(model, row_id, full_ids, prompt_len):
    """One teacher-forcing pass; save CAPTURE_LAYERS residual stream as fp16."""
    span_start = max(0, prompt_len - PROMPT_TAIL)
    with torch.inference_mode():
        try:
            out = model(full_ids, output_hidden_states=True, logits_to_keep=1)
        except TypeError:
            out = model(full_ids, output_hidden_states=True)
    hs = out.hidden_states  # [0]=embeddings, [l+1]=output of layer l
    layers = np.stack(
        [hs[l + 1][0, span_start:].to(torch.float16).cpu().numpy()
         for l in CAPTURE_LAYERS])
    token_ids = full_ids[0, span_start:].to(torch.int32).cpu().numpy()
    path = os.path.join(ACT_DIR, f"{row_id}.npz")
    np.savez(path, hidden=layers, token_ids=token_ids)
    return path, span_start


def main():
    os.makedirs(ACT_DIR, exist_ok=True)
    selected = select_rows()
    done = load_done_row_ids()
    todo = [r for r in selected if r["row_id"] not in done]
    log(f"{len(selected)} selected rows, {len(done)} done, {len(todo)} to capture")
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
    tokenizer.padding_side = "left"
    eos_ids = [tokenizer.pad_token_id, tokenizer.eos_token_id]
    log(f"Model loaded. eos ids for generate: {eos_ids}")

    with open(OUT_PATH, "a") as out_f:
        for b0 in range(0, len(todo), BATCH_SIZE):
            batch = todo[b0:b0 + BATCH_SIZE]
            batch_idx = b0 // BATCH_SIZE
            torch.manual_seed(BASE_SEED + batch_idx)

            prompts, prompt_ids_list = [], []
            for r in batch:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": r["prompt"]}],
                    tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
                prompts.append(text)
                prompt_ids_list.append(
                    tokenizer(text, return_tensors="pt",
                              add_special_tokens=False)["input_ids"][0])

            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
            prompt_lens = enc["attention_mask"].sum(1).tolist()
            t0 = time.time()
            with torch.inference_mode():
                gen = model.generate(**enc, **GEN_KWARGS, eos_token_id=eos_ids)
            log(f"batch {batch_idx}: {len(batch)} rows generated in "
                f"{time.time()-t0:.0f}s")

            for i, r in enumerate(batch):
                n_prompt = prompt_lens[i]
                gen_row = gen[i][enc["input_ids"].shape[1]:]
                stopped_early = (len(gen_row) > 0
                                 and gen_row[-1].item() == tokenizer.pad_token_id)
                gen_row = gen_row[gen_row != tokenizer.pad_token_id]
                eos_hits = [j for j, t in enumerate(gen_row.tolist())
                            if t in eos_ids]
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
                npz_path, span_start = capture_activations(
                    model, r["row_id"], full_ids, n_prompt)

                harmful = r["prompt_harm_label"] == "harmful"
                rec = {
                    "row_id": r["row_id"],
                    "subcategory": r.get("subcategory"),
                    "language": r.get("language"),
                    "adversarial": r.get("adversarial"),
                    "prompt": r["prompt"],
                    "template_id": None,   # plain prompt, no jailbreak template
                    "model": "local:/data/models/Qwen3.5-9B",
                    "generation_params": {**GEN_KWARGS, "enable_thinking": False,
                                          "seed": BASE_SEED + batch_idx,
                                          "batch_idx": batch_idx},
                    "response": response,
                    "finish_reason": finish,
                    "n_prompt_tokens": n_prompt,
                    "n_response_tokens": int(kept.shape[0]),
                    # exchange-classifier label: from the PROMPT only.
                    "exchange_label": "harmful" if harmful else "safe",
                    # no Qwen3Guard run (per user); label later if needed
                    "guard_safety": None,
                    "guard_refusal": None,
                    "guard_categories": None,
                    "activations_file": os.path.relpath(npz_path, OUT_DIR),
                    "activations_layers": CAPTURE_LAYERS,
                    "activations_span_start_in_full_seq": span_start,
                    "activations_response_offset": n_prompt - span_start,
                    "activations_prompt_tail": PROMPT_TAIL,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                log(f"  {r['row_id']} [{rec['exchange_label']}] {finish} "
                    f"resp={kept.shape[0]}tok")

    log("DONE")


if __name__ == "__main__":
    main()
