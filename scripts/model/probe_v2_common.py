"""
Shared machinery for the v2 linear-probe experiment (teacher-forced Qwen3.5-9B
activations on data/gemma_v2_no_refusal/).

Experiment definition (locked for attempt 1):
- Input: v2 (prompt, response) exchange, teacher-forced (prefill only, no decoding).
  Format A (row with response): chat-template prompt prefix with
  add_generation_prompt=True, enable_thinking=False + response tokens + <|im_end|>
  (separate tokenization + concatenate, per linear_probe_preflight REPORT section 7).
  Format B (response == ""): prompt prefix only, same add_generation_prompt=True
  boundary (preflight option 1, recommended for consistency).
- Feature: residual-stream activation at the LAST token, decoder layer 27 =
  second-to-last full-attention layer (full-attention layers are
  3,7,11,15,19,23,27,31; layer 27 is a raw pre-norm residual value, unlike
  post-norm layer 31). hidden_states[27 + 1] since hidden_states[0] = embeddings.
- Label: v2 final_label (harmful=1, unharmful=0).

Kill-safety: every row's activation is written to
activations/{split}/{row_id}.npy via temp-file + os.replace (atomic) immediately
after its batch finishes, plus one appended line per row in
progress_{split}.jsonl (fsync'd). Re-running a script skips rows whose .npy
already exists, so a kill loses at most the in-flight batch.
"""

import json
import os
import time

# --- REQUIRED env-var block (must precede torch/transformers import) ---
os.environ.setdefault("USER", "mls01")
os.environ.setdefault("LOGNAME", "mls01")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/home/mls01/.cache/torchinductor")
os.environ.setdefault("TRITON_CACHE_DIR", "/home/mls01/.cache/triton")
os.environ.setdefault("XDG_CACHE_HOME", "/home/mls01/.cache")
# Fragmentation headroom (a foreign ~4 GiB orphan process shares the GPU).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_PATH = "/data/models/Qwen3.5-9B"
DATA_DIR = "/home/mls01/data/gemma_v2_no_refusal"
OUT_DIR = "/home/mls01/scripts/model/results/linear_probe_v2_layer27_lasttoken"
RUN_LOG = os.path.join(OUT_DIR, "run.log")

LAYER = 27                      # second-to-last full-attention layer
HIDDEN_SIZE = 4096
SPLITS = ("train", "validation", "test")

# Batching: padded-token budget per batch (max_len_in_batch * batch_size).
# 24576 padded tokens => hidden_states transient ~33*24576*4096*2B = 6.6 GB,
# plus ~17 GB weights => ~24 GB peak. (32768 was the original budget, but a
# validation batch OOM'd on fragmentation with a foreign ~4 GiB orphan
# process on the GPU; 24576 + expandable_segments leaves real headroom.)
PADDED_TOKEN_BUDGET = 24576
MAX_BATCH = 16


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(line + "\n")


def load_rows(split):
    assert split in SPLITS
    path = os.path.join(DATA_DIR, f"{split}.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows


def build_input_ids(tokenizer, prompt, response):
    """Teacher-forced token ids for one exchange (Format A / Format B opt. 1)."""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if response and response.strip():
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        return prompt_ids + response_ids + [im_end_id]
    return prompt_ids


def plan_batches(items, token_budget=PADDED_TOKEN_BUDGET, max_batch=MAX_BATCH):
    """items: list of (row_index, n_tokens). Sort by length, greedy-pack so
    max_len_in_batch * batch_size <= token_budget. Returns list of index lists."""
    ordered = sorted(items, key=lambda t: t[1])
    batches, cur, cur_max = [], [], 0
    for idx, n in ordered:
        new_max = max(cur_max, n)
        if cur and (new_max * (len(cur) + 1) > token_budget
                    or len(cur) >= max_batch):
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = n
        cur.append(idx)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def load_model_and_tokenizer(max_memory=None):
    """Load Qwen3.5-9B + tokenizer. max_memory: optional accelerate cap,
    e.g. {"cuda:0": "12GiB", "cpu": "200GiB"} — layers that don't fit are
    offloaded to CPU (hidden-state capture still works). None = old behavior
    (everything that fits goes to GPU)."""
    log(f"Loading tokenizer from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True)
    log(f"Loading model from {MODEL_PATH} (bf16, device_map=auto, "
        f"max_memory={max_memory}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto", max_memory=max_memory)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Sanity-check the layer assumption this experiment is built on.
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types is not None:
        assert layer_types[LAYER] == "full_attention", \
            f"layer {LAYER} is {layer_types[LAYER]}, expected full_attention"
        full_attn = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        assert full_attn[-2] == LAYER, \
            f"second-to-last full-attention layer is {full_attn[-2]}, not {LAYER}"
        log(f"Layer check OK: full-attention layers = {full_attn}, "
            f"capturing layer {LAYER} (second-to-last).")
    assert text_cfg.hidden_size == HIDDEN_SIZE
    return model, tokenizer


def act_dir_for(split):
    return os.path.join(OUT_DIR, "activations", split)


def done_row_ids(split):
    d = act_dir_for(split)
    if not os.path.isdir(d):
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".npy")}


def capture_split(model, tokenizer, split,
                  token_budget=PADDED_TOKEN_BUDGET, max_batch=MAX_BATCH):
    """Teacher-forced capture of last-token layer-27 activations for one split.
    Cache-aware: rows with an existing .npy are skipped. Returns nothing.
    token_budget/max_batch: defaults = module constants (old behavior);
    pass smaller values to shrink the hidden-states transient when sharing
    the GPU."""
    act_dir = act_dir_for(split)
    os.makedirs(act_dir, exist_ok=True)
    progress_path = os.path.join(OUT_DIR, f"progress_{split}.jsonl")

    rows = load_rows(split)
    done = done_row_ids(split)
    todo_idx = [i for i, r in enumerate(rows) if r["row_id"] not in done]
    log(f"[{split}] {len(rows)} rows, {len(done)} cached, {len(todo_idx)} to capture")
    if not todo_idx:
        return

    # Tokenize everything up front (CPU), keep only what we still need.
    t0 = time.time()
    ids_by_idx = {}
    for i in todo_idx:
        r = rows[i]
        ids_by_idx[i] = build_input_ids(tokenizer, r["prompt"],
                                        r.get("response") or "")
    log(f"[{split}] tokenized {len(todo_idx)} rows in {time.time()-t0:.1f}s")

    batches = plan_batches([(i, len(ids_by_idx[i])) for i in todo_idx],
                           token_budget=token_budget, max_batch=max_batch)
    total_tokens = sum(len(ids_by_idx[i]) for i in todo_idx)
    log(f"[{split}] {len(batches)} batches, {total_tokens} tokens to prefill")

    pad_id = tokenizer.pad_token_id
    captured_tokens = 0
    t_start = time.time()
    with open(progress_path, "a") as prog_f:
        for b_idx, batch in enumerate(batches):
            seqs = [ids_by_idx[i] for i in batch]
            max_len = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), max_len), pad_id,
                                   dtype=torch.long)
            attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
            for j, s in enumerate(seqs):  # left padding
                input_ids[j, max_len - len(s):] = torch.tensor(s)
                attn[j, max_len - len(s):] = 1
            input_ids = input_ids.to(model.device)
            attn = attn.to(model.device)

            t_b = time.time()
            with torch.inference_mode():
                try:
                    out = model(input_ids=input_ids, attention_mask=attn,
                                output_hidden_states=True, logits_to_keep=1)
                except TypeError:
                    out = model(input_ids=input_ids, attention_mask=attn,
                                output_hidden_states=True)
            # hidden_states[0] = embeddings; [l+1] = output of decoder layer l.
            # Left padding => last sequence position is the last real token
            # for every row in the batch.
            vecs = out.hidden_states[LAYER + 1][:, -1, :]
            vecs = vecs.to(torch.float16).cpu().numpy()
            del out

            wall = time.time() - t_b
            batch_tokens = int(attn.sum().item())
            captured_tokens += batch_tokens
            for j, i in enumerate(batch):
                r = rows[i]
                tmp = os.path.join(act_dir, f".{r['row_id']}.tmp")
                final = os.path.join(act_dir, f"{r['row_id']}.npy")
                with open(tmp, "wb") as vf:  # file object: np.save won't append .npy
                    np.save(vf, vecs[j])
                os.replace(tmp, final)  # atomic
                prog_f.write(json.dumps({
                    "row_id": r["row_id"], "split": split,
                    "label": r["final_label"],
                    "n_tokens": len(ids_by_idx[i]),
                    "batch_idx": b_idx, "batch_wall_s": round(wall, 3),
                    "ts": time.time(),
                }, ensure_ascii=False) + "\n")
            prog_f.flush()
            os.fsync(prog_f.fileno())

            done_n = len(done) + sum(len(b) for b in batches[:b_idx + 1])
            rate = captured_tokens / (time.time() - t_start)
            eta_s = (total_tokens - captured_tokens) / max(rate, 1e-9)
            log(f"[{split}] batch {b_idx+1}/{len(batches)} "
                f"({len(batch)} rows, {batch_tokens} tok, {wall:.1f}s) | "
                f"rows {done_n}/{len(rows)} | {rate:.0f} tok/s | "
                f"ETA {eta_s/60:.1f} min")
            del input_ids, attn, vecs

    log(f"[{split}] capture complete: {len(rows)} rows cached "
        f"({time.time()-t_start:.0f}s for the new rows)")


def load_activation_matrix(split):
    """Assemble cached per-row .npy files into (X float32, y int, row_ids),
    in dataset row order. Asserts full coverage and correct shape."""
    rows = load_rows(split)
    act_dir = act_dir_for(split)
    X = np.empty((len(rows), HIDDEN_SIZE), dtype=np.float32)
    y = np.empty(len(rows), dtype=np.int64)
    row_ids = []
    for i, r in enumerate(rows):
        path = os.path.join(act_dir, f"{r['row_id']}.npy")
        assert os.path.exists(path), f"missing activation for {r['row_id']}"
        v = np.load(path)
        assert v.shape == (HIDDEN_SIZE,), \
            f"{r['row_id']}: bad shape {v.shape}"
        X[i] = v.astype(np.float32)
        y[i] = 1 if r["final_label"] == "harmful" else 0
        row_ids.append(r["row_id"])
    log(f"[{split}] assembled matrix: X{X.shape}, "
        f"harmful {int(y.sum())}/{len(y)}")
    return X, y, row_ids, rows
