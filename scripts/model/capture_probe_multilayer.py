"""
Multi-layer, multi-pooling activation capture for the v2 linear-probe
experiment — a follow-up to attempt 1/2
(results/linear_probe_v2_layer27_lasttoken[_sweep]/), which locked layer 27 +
last-token without ever comparing against another layer or pooling. That
sweep's grid (30 LogReg configs across C/penalty/class_weight) landed on a
tight F1 plateau (0.908-0.923) with recall stuck at ~0.900 in almost every
good cell — the signature of a feature-limited ceiling, not a classifier one.

This script captures ALL 8 full-attention layers (3,7,11,15,19,23,27,31) x
4 poolings per row in ONE forward pass, at zero extra GPU cost: the model
already materializes all 33 hidden_states per forward call (output_hidden_states
=True), and attempt-1/2 threw away 32 of them. Poolings:

  - last          : last-token residual stream (attempt-1/2's feature, kept
                    for a same-layer sanity comparison against the old cache)
  - mean_prompt   : mean over prompt tokens only
  - mean_response : mean over response tokens (incl. the trailing <|im_end|>);
                    for the ~58% of v2 rows with an EMPTY response, this is
                    UNDEFINED and falls back to the last-token vector (there
                    is no response span to average) -- has_response in
                    progress_{split}.jsonl marks which rows hit the fallback
  - mean_last16   : mean over the last 16 tokens of the full sequence,
                    matching the final system design's 16-token sliding probe
                    window (see CLAUDE.md "Final System Design") -- unlike
                    mean_response this is ALWAYS defined and, unlike
                    last-token, evaluable on a truncated prefix

Preflight (results/linear_probe_preflight/) sized "option 4: prompt-mean +
response-mean + last-token, 8 layers" at ~486 MB for all 2471 rows but never
built it. This script builds that option plus mean_last16 (~650 MB for all
2471 rows, 8 layers x 4 poolings x 4096 dims x fp16 per row).

Feature/label/input-format are otherwise LOCKED from attempt 1 (same
teacher-forced Format A/B chat-template construction, same v2 dataset). No
LogReg training here -- this is capture only; a follow-up sweep script picks
the best (layer, pooling, hyperparameters) combination on these caches.

Storage: activations/{split}/{row_id}.npz (cache-aware -- rows with an
existing .npz are skipped, like attempt-1's .npy cache), keys
"layer{L}_{pooling}" e.g. "layer27_last", each a (4096,) fp16 array.
progress_{split}.jsonl gets one appended, fsync'd line per row.

Run (qwen35_env, only env with transformers 5.x for qwen3_5):

    /home/mls01/.conda/envs/qwen35_env/bin/python \\
        scripts/model/capture_probe_multilayer.py --split validation

Sanity check on a handful of rows before a full run:

    /home/mls01/.conda/envs/qwen35_env/bin/python \\
        scripts/model/capture_probe_multilayer.py --split validation \\
        --limit 8 --out-dir results/linear_probe_v3_multilayer_pooling/sanity_check
"""

import argparse
import json
import os
import time

# --- REQUIRED env-var block (must precede torch/transformers import) ---
os.environ.setdefault("USER", "mls01")
os.environ.setdefault("LOGNAME", "mls01")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/home/mls01/.cache/torchinductor")
os.environ.setdefault("TRITON_CACHE_DIR", "/home/mls01/.cache/triton")
os.environ.setdefault("XDG_CACHE_HOME", "/home/mls01/.cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import probe_v2_common as C  # noqa: E402 (MODEL_PATH, DATA_DIR, load_rows, plan_batches)

DEFAULT_OUT_DIR = "/home/mls01/scripts/model/results/linear_probe_v3_multilayer_pooling"
CAPTURE_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
POOLINGS = ["last", "mean_prompt", "mean_response", "mean_last16"]
LAST_N = 16
HIDDEN_SIZE = 4096


def make_logger(out_dir):
    run_log = os.path.join(out_dir, "run.log")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        os.makedirs(out_dir, exist_ok=True)
        with open(run_log, "a") as f:
            f.write(line + "\n")
    return log


def build_input_ids_with_spans(tokenizer, prompt, response):
    """Same teacher-forced construction as probe_v2_common.build_input_ids
    (Format A/B, locked from attempt 1), but also returns (n_prompt, n_response)
    token-count spans needed for prompt/response-aware pooling."""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if response and response.strip():
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        full_ids = prompt_ids + response_ids + [im_end_id]
        return full_ids, len(prompt_ids), len(response_ids) + 1
    return prompt_ids, len(prompt_ids), 0


def load_model_and_tokenizer(log):
    log(f"Loading tokenizer from {C.MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        C.MODEL_PATH, local_files_only=True, trust_remote_code=True)
    log(f"Loading model from {C.MODEL_PATH} (bf16, device_map=auto) ...")
    model = AutoModelForCausalLM.from_pretrained(
        C.MODEL_PATH, local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="auto")
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    layer_types = getattr(text_cfg, "layer_types", None)
    if layer_types is not None:
        full_attn = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        assert full_attn == CAPTURE_LAYERS, (
            f"full-attention layers {full_attn} != CAPTURE_LAYERS {CAPTURE_LAYERS}; "
            "model architecture changed since preflight, update CAPTURE_LAYERS")
        log(f"Layer check OK: full-attention layers = {full_attn} "
            f"(capturing all {len(CAPTURE_LAYERS)}).")
    assert text_cfg.hidden_size == HIDDEN_SIZE
    return model, tokenizer


def act_dir_for(out_dir, split):
    return os.path.join(out_dir, "activations", split)


def done_row_ids(out_dir, split):
    d = act_dir_for(out_dir, split)
    if not os.path.isdir(d):
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".npz")}


def pool_row(hs_by_layer, j, start, max_len, n_prompt, n_response):
    """hs_by_layer: {layer: np.ndarray [B, T, H] fp16}. Returns dict of
    {"layer{L}_{pooling}": (H,) fp16 array} for row j in the batch."""
    out = {}
    last16_start = max(start, max_len - LAST_N)
    for L, hs in hs_by_layer.items():
        last_vec = hs[j, max_len - 1, :]
        prompt_vec = hs[j, start:start + n_prompt, :].mean(axis=0)
        if n_response > 0:
            response_vec = hs[j, start + n_prompt:max_len, :].mean(axis=0)
        else:
            response_vec = last_vec.copy()
        last16_vec = hs[j, last16_start:max_len, :].mean(axis=0)
        out[f"layer{L}_last"] = last_vec.astype(np.float16)
        out[f"layer{L}_mean_prompt"] = prompt_vec.astype(np.float16)
        out[f"layer{L}_mean_response"] = response_vec.astype(np.float16)
        out[f"layer{L}_mean_last16"] = last16_vec.astype(np.float16)
    return out


def capture_split(model, tokenizer, split, out_dir, log, limit=None):
    act_dir = act_dir_for(out_dir, split)
    os.makedirs(act_dir, exist_ok=True)
    progress_path = os.path.join(out_dir, f"progress_{split}.jsonl")

    rows = C.load_rows(split)
    if limit is not None:
        rows = rows[:limit]
    done = done_row_ids(out_dir, split)
    todo_idx = [i for i, r in enumerate(rows) if r["row_id"] not in done]
    log(f"[{split}] {len(rows)} rows, {len(done)} cached, {len(todo_idx)} to capture")
    if not todo_idx:
        return

    t0 = time.time()
    span_by_idx = {}
    for i in todo_idx:
        r = rows[i]
        full_ids, n_prompt, n_response = build_input_ids_with_spans(
            tokenizer, r["prompt"], r.get("response") or "")
        span_by_idx[i] = (full_ids, n_prompt, n_response)
    log(f"[{split}] tokenized {len(todo_idx)} rows in {time.time()-t0:.1f}s")

    batches = C.plan_batches([(i, len(span_by_idx[i][0])) for i in todo_idx])
    total_tokens = sum(len(span_by_idx[i][0]) for i in todo_idx)
    log(f"[{split}] {len(batches)} batches, {total_tokens} tokens to prefill")

    pad_id = tokenizer.pad_token_id
    captured_tokens = 0
    t_start = time.time()
    with open(progress_path, "a") as prog_f:
        for b_idx, batch in enumerate(batches):
            seqs = [span_by_idx[i][0] for i in batch]
            max_len = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
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
            hs_by_layer = {
                L: out.hidden_states[L + 1].to(torch.float16).cpu().numpy()
                for L in CAPTURE_LAYERS
            }
            del out

            wall = time.time() - t_b
            batch_tokens = int(attn.sum().item())
            captured_tokens += batch_tokens
            for j, i in enumerate(batch):
                r = rows[i]
                full_ids, n_prompt, n_response = span_by_idx[i]
                L_j = len(full_ids)
                start = max_len - L_j
                vecs = pool_row(hs_by_layer, j, start, max_len, n_prompt, n_response)

                tmp = os.path.join(act_dir, f".{r['row_id']}.tmp")
                final = os.path.join(act_dir, f"{r['row_id']}.npz")
                with open(tmp, "wb") as vf:
                    np.savez(vf, **vecs)
                os.replace(tmp, final)  # atomic

                prog_f.write(json.dumps({
                    "row_id": r["row_id"], "split": split,
                    "label": r["final_label"],
                    "n_prompt_tokens": n_prompt,
                    "n_response_tokens": n_response,
                    "n_total_tokens": L_j,
                    "has_response": n_response > 0,
                    "batch_idx": b_idx, "batch_wall_s": round(wall, 3),
                    "ts": time.time(),
                }, ensure_ascii=False) + "\n")
            prog_f.flush()
            os.fsync(prog_f.fileno())
            del hs_by_layer

            done_n = len(done) + sum(len(b) for b in batches[:b_idx + 1])
            rate = captured_tokens / (time.time() - t_start)
            eta_s = (total_tokens - captured_tokens) / max(rate, 1e-9)
            log(f"[{split}] batch {b_idx+1}/{len(batches)} "
                f"({len(batch)} rows, {batch_tokens} tok, {wall:.1f}s) | "
                f"rows {done_n}/{len(rows)} | {rate:.0f} tok/s | "
                f"ETA {eta_s/60:.1f} min")
            del input_ids, attn

    log(f"[{split}] capture complete: {len(rows)} rows cached "
        f"({time.time()-t_start:.0f}s for the new rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=C.SPLITS)
    ap.add_argument("--limit", type=int, default=None,
                     help="cap to the first N rows of the split (sanity checks)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    log = make_logger(out_dir)
    log("=== capture_probe_multilayer starting "
        f"(split={args.split}, limit={args.limit}, out_dir={out_dir}) ===")

    model, tokenizer = load_model_and_tokenizer(log)
    capture_split(model, tokenizer, args.split, out_dir, log, limit=args.limit)

    log("=== capture_probe_multilayer DONE ===")


if __name__ == "__main__":
    main()
