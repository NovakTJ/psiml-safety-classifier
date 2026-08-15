"""
Same first-token log-odds extraction as extract_lora_logit_scores.py, but
pointed at the LoRA attention-only ablation adapter (results/
gemma_lora_v2_lora_dora_ablation/lora_attention_only/best_adapter/) instead
of the all-linear Phase 2 winner. Motivation: the ensemble's classifier
stage was using the all-linear adapter (standalone test F1 0.9163), but the
attention-only adapter independently generalizes better on test (F1 0.9375,
best of the 4 LoRA/DoRA ablation variants) -- swapping it in as the
ensemble's Stage-2 judge should raise the ensemble's ceiling too.

A separate script rather than parameterizing extract_lora_logit_scores.py:
that script's assertions are specific to the Phase 2 run_config (lr=3e-4,
dropout=0.0, all-linear target_modules) and is the artifact backing the
already-locked, already-tested `ensemble_v2_final`/`ensemble_v2_final_maxf1`
results -- left unchanged so those remain exactly reproducible.

Run with ccpp_env (GPU, Gemma-3-1B, ~486 rows val+test, one forward pass
each):

    /home/mls01/ccpp_env/bin/python \\
        scripts/model/ensemble_v3/extract_lora_logit_scores_attention_only.py
"""

import os

# ---------------------------------------------------------------------------
# Required env-var block — MUST run before importing torch/transformers.
# ---------------------------------------------------------------------------
os.environ.setdefault("USER", "mls01")
os.environ.setdefault("LOGNAME", "mls01")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/home/mls01/.cache/torchinductor")
os.environ.setdefault("TRITON_CACHE_DIR", "/home/mls01/.cache/triton")
os.environ.setdefault("XDG_CACHE_HOME", "/home/mls01/.cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

# sweep_lora_v2_phase2.py (teammate's, shared) and results/ both live one
# level up in scripts/model/, not in this ensemble_v3/ subfolder.
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
import sweep_lora_v2_phase2 as p2  # noqa: E402  (reuse exact pipeline, not reimplemented)

RUN_DIR = SCRIPT_DIR / "results" / "gemma_lora_v2_lora_dora_ablation" / "lora_attention_only"
ADAPTER_DIR = RUN_DIR / "best_adapter"
RUN_CONFIG_PATH = RUN_DIR / "run_config.json"

DATA_DIR = Path("/home/mls01/data/gemma_v2_no_refusal")
OUT_DIR = SCRIPT_DIR / "results" / "gemma_lora_v2_logit_scores_attention_only"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_adapter_dir(adapter_dir):
    files = sorted(p.name for p in adapter_dir.iterdir() if p.is_file())
    return {name: sha256_file(adapter_dir / name) for name in files}


def main():
    print("=== extract_lora_logit_scores_attention_only starting ===")

    if not (RUN_CONFIG_PATH.exists() and ADAPTER_DIR.exists()):
        raise FileNotFoundError(f"Expected attention-only ablation run at {RUN_DIR}")
    run_config = json.loads(RUN_CONFIG_PATH.read_text())
    expected = {"method": "LoRA", "use_dora": False, "target_scope": "attention_only",
               "learning_rate": 0.0002, "rank": 8, "alpha": 16, "dropout": 0.05, "seed": 42}
    for k, v in expected.items():
        assert run_config[k] == v, f"run_config.json[{k}]={run_config[k]!r}, expected {v!r}"
    assert run_config["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert run_config["prompt"] == p2.PROMPT_1
    print(f"[1] Adapter verified: {ADAPTER_DIR} "
          f"(LoRA, attention_only, lr=2e-4, dropout=0.05)")

    hashes_before = hash_adapter_dir(ADAPTER_DIR)

    tokenizer, end_of_turn_id, build_example, parse_label = p2.build_tokenizer_and_helpers()
    harm_first_id = tokenizer.encode("harmful", add_special_tokens=False)[0]
    un_first_id = tokenizer.encode("unharmful", add_special_tokens=False)[0]
    print(f"[2] First-token ids: harm={harm_first_id} "
          f"({tokenizer.decode([harm_first_id])!r}), "
          f"un={un_first_id} ({tokenizer.decode([un_first_id])!r})")
    assert harm_first_id != un_first_id

    print("[3] Loading base model + adapter ...")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained(
        p2.MODEL_PATH, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager",
    ).to("cuda")
    peft_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    peft_model.eval()
    print(f"    [OK] {p2.MODEL_PATH} + {ADAPTER_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("validation", "test"):
        df = pd.read_json(DATA_DIR / f"{split}.jsonl", lines=True)
        print(f"[{split}] {len(df)} rows")

        rows_out = []
        t0 = time.time()
        with torch.inference_mode():
            for i, row in enumerate(df.itertuples(index=False), start=1):
                ex = build_example(row.prompt, row.response, row.final_label)
                prefix = ex["input_ids"][:ex["prefix_len"]]
                input_ids = torch.tensor([prefix], device="cuda")
                attention_mask = torch.ones_like(input_ids)
                out = peft_model(input_ids=input_ids, attention_mask=attention_mask)
                next_logits = out.logits[0, -1, :]
                score = float(next_logits[harm_first_id] - next_logits[un_first_id])
                rows_out.append({"row_id": row.row_id, "score": score,
                                 "truncated": ex["truncated"]})
                if i % 60 == 0 or i == len(df):
                    print(f"      {split}: {i}/{len(df)} ({time.time()-t0:.1f}s)")

        out_path = OUT_DIR / f"{split}_scores.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row_id", "score", "truncated"])
            for r in rows_out:
                w.writerow([r["row_id"], r["score"], r["truncated"]])
        scores = [r["score"] for r in rows_out]
        n_trunc = sum(1 for r in rows_out if r["truncated"])
        print(f"[{split}] saved {len(rows_out)} rows -> {out_path} "
              f"(score range [{min(scores):.3f}, {max(scores):.3f}], "
              f"truncated {n_trunc}/{len(rows_out)})")

    del base_model, peft_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    hashes_after = hash_adapter_dir(ADAPTER_DIR)
    hash_unchanged = hashes_before == hashes_after
    print(f"[hash check] adapter unchanged: {hash_unchanged}")
    if not hash_unchanged:
        raise RuntimeError("Adapter files changed during scoring — investigate before trusting scores.")

    meta = {
        "adapter_dir": str(ADAPTER_DIR),
        "score_definition": "logit('harm') - logit('un'), first generated token, "
                            "single forward pass (no generation loop)",
        "harm_first_token_id": harm_first_id, "un_first_token_id": un_first_id,
        "adapter_sha256_unchanged": hash_unchanged,
    }
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("=== extract_lora_logit_scores_attention_only DONE ===")


if __name__ == "__main__":
    main()
