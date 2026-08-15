"""
Generic version of extract_lora_logit_scores.py -- same first-token log-odds
extraction (logit("harm") - logit("un"), one forward pass per row, no
generation loop), parameterized by --run-dir so it works for any of the 4
LoRA/DoRA ablation adapters instead of being hardcoded to one.

Used to extract classifier scores from dora_attention_only and
dora_all_linear (the two ablation variants not yet scored) for a stacked
probe+4-classifier ensemble on VALIDATION only -- see
stack_ensemble_v2.py. Does not touch or modify extract_lora_logit_scores.py
or extract_lora_logit_scores_attention_only.py (those back the already-
locked ensemble_v2_final/ensemble_v2_final_maxf1 results).

Run with ccpp_env:

    /home/mls01/ccpp_env/bin/python \\
        scripts/model/extract_lora_logit_scores_generic.py \\
        --run-dir results/gemma_lora_v2_lora_dora_ablation/dora_attention_only
"""

import argparse
import os

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

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import sweep_lora_v2_phase2 as p2  # noqa: E402

DATA_DIR = Path("/home/mls01/data/gemma_v2_no_refusal")


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="e.g. results/gemma_lora_v2_lora_dora_ablation/dora_attention_only")
    args = ap.parse_args()

    run_dir = (SCRIPT_DIR / args.run_dir) if not args.run_dir.is_absolute() else args.run_dir
    adapter_dir = run_dir / "best_adapter"
    run_config_path = run_dir / "run_config.json"
    out_dir = SCRIPT_DIR / "results" / f"gemma_lora_v2_logit_scores_{run_dir.name}"

    print(f"=== extract_lora_logit_scores_generic starting ({run_dir.name}) ===")
    if not (run_config_path.exists() and adapter_dir.exists()):
        raise FileNotFoundError(f"Expected run at {run_dir}")
    run_config = json.loads(run_config_path.read_text())
    assert run_config["prompt"] == p2.PROMPT_1
    print(f"[1] Adapter verified: {adapter_dir} (method={run_config['method']}, "
          f"target_scope={run_config['target_scope']}, lr={run_config['learning_rate']}, "
          f"dropout={run_config['dropout']})")

    hashes_before = hash_adapter_dir(adapter_dir)

    tokenizer, end_of_turn_id, build_example, parse_label = p2.build_tokenizer_and_helpers()
    harm_first_id = tokenizer.encode("harmful", add_special_tokens=False)[0]
    un_first_id = tokenizer.encode("unharmful", add_special_tokens=False)[0]
    assert harm_first_id != un_first_id

    print("[2] Loading base model + adapter ...")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained(
        p2.MODEL_PATH, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager",
    ).to("cuda")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    peft_model.eval()
    print(f"    [OK] {p2.MODEL_PATH} + {adapter_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

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
                rows_out.append({"row_id": row.row_id, "score": score})
                if i % 60 == 0 or i == len(df):
                    print(f"      {split}: {i}/{len(df)} ({time.time()-t0:.1f}s)")

        out_path = out_dir / f"{split}_scores.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row_id", "score"])
            for r in rows_out:
                w.writerow([r["row_id"], r["score"]])
        print(f"[{split}] saved -> {out_path}")

    del base_model, peft_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    hashes_after = hash_adapter_dir(adapter_dir)
    hash_unchanged = hashes_before == hashes_after
    print(f"[hash check] adapter unchanged: {hash_unchanged}")
    if not hash_unchanged:
        raise RuntimeError("Adapter files changed during scoring.")

    print(f"=== DONE ({run_dir.name}) ===")


if __name__ == "__main__":
    main()
