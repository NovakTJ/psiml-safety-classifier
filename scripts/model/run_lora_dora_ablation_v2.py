#!/usr/bin/env python
"""Standalone LoRA vs DoRA / attention-only vs all-linear ablation — Gemma 3
1B IT, v2 (no-refusal) dataset.

Controlled 2x2 comparison:

    method  target_scope       status
    LoRA    attention + MLP    existing Experiment 1 (NOT retrained here)
    LoRA    attention only     NEW run 1
    DoRA    attention + MLP    NEW run 2
    DoRA    attention only     NEW run 3

Every other hyperparameter is locked to Experiment 1's values (see
reference_config.json, generated from
scripts/model/results/gemma_lora_v2_exp1_r8_lr2e4_seed42_max8_es2/run_config.json):
    learning_rate = 2e-4, rank = 8, alpha = 16, dropout = 0.05, seed = 42,
    max_epochs = 8, early_stopping_patience = 2, model = /data/models/gemma-3-1b-it,
    dataset = data/gemma_v2_no_refusal/{train,validation}.jsonl, prompt_1,
    max_seq_length = 1024, batch=4 x grad_accum=8, warmup_ratio=0.05,
    weight_decay=0, AdamW + linear scheduler, BF16, gradient checkpointing.

Only `use_dora` (LoraConfig) and `target_modules` (attention-only vs
all-linear) vary across the three NEW runs. Experiment 1's own
directory/adapter is never retrained, modified, or copied — it is read-only
and folded into ablation_summary.csv as source=existing_reference.

The held-out split (train/validation's sibling file) is NEVER opened by this
script (see the self-audit in --dry-run, which greps this file's own source
for the held-out filename and asserts it never appears literally).

Smoke test: before each NEW run's real training, a short preflight loads the
base model, wraps it with that run's exact LoraConfig, verifies target
modules / use_dora / trainable-param-only-adapter / DoRA magnitude vectors /
frozen base params, runs one real forward+backward on a real training batch
(checks finite loss and correct gradient presence/absence), THEN runs the
actual generation/eval path (peft_model.generate(), do_sample=False,
max_new_tokens=10 -- identical to real validation) on N_SMOKE_EVAL_SAMPLES
(default 5) randomly sampled rows from validation.jsonl (seeded off SEED, so
the same 5 rows are used every attempt), so a break in the inference/KV-cache
path -- which the forward+backward check alone cannot catch -- is caught
before committing to a full multi-epoch run. Records peak VRAM, then fully
discards the model and frees the GPU before the real run loads its own clean
base model. A failed smoke test marks the run `failed_preflight` (never
auto-retried) and moves on to the next configuration; it never touches the
real training loop.

Usage:
    /home/mls01/ccpp_env/bin/python run_lora_dora_ablation_v2.py --dry-run
    /home/mls01/ccpp_env/bin/python run_lora_dora_ablation_v2.py

Safe to re-run: completed runs are skipped, failed_preflight runs are never
retried automatically, failed runs are retried up to --max-attempts, and a
run found "running" at startup (previous process died mid-smoke-test or
mid-training) is archived and restarted from scratch — this script never
resumes a Trainer from a mid-training checkpoint. No time budget: the script
runs to completion (all three NEW runs completed or exhausted) regardless of
wall-clock time.
"""

import argparse
import gc
import json
import os
import random
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Required env-var block — MUST run before importing torch/transformers.
# uid 1562 has no /etc/passwd entry in this container; without these,
# transformers crashes with "getpwuid(): uid not found: 1562" while
# resolving cache dirs.
# ---------------------------------------------------------------------------
os.environ.setdefault("USER", "mls01")
os.environ.setdefault("LOGNAME", "mls01")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/home/mls01/.cache/torchinductor")
os.environ.setdefault("TRITON_CACHE_DIR", "/home/mls01/.cache/triton")
os.environ.setdefault("XDG_CACHE_HOME", "/home/mls01/.cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed paths and constants — all identical to Experiment 1
# (scripts/model/results/gemma_lora_v2_exp1_r8_lr2e4_seed42_max8_es2/run_config.json),
# read directly where possible so nothing here is guessed.
# ---------------------------------------------------------------------------
MODEL_PATH = Path("/data/models/gemma-3-1b-it")
V2_DATA_DIR = Path("/home/mls01/data/gemma_v2_no_refusal")
TRAIN_PATH = V2_DATA_DIR / "train.jsonl"
VAL_PATH = V2_DATA_DIR / "validation.jsonl"
# The held-out split is intentionally never named as a literal path constant
# here (see the self-audit in --dry-run, which greps this file's own source
# for the held-out filename and asserts it never appears).

ABLATION_DIR = Path("/home/mls01/scripts/model/results/gemma_lora_v2_lora_dora_ablation")
LOCK_PATH = ABLATION_DIR / "lora_dora_ablation.pid"

EXP1_DIR = Path("/home/mls01/scripts/model/results/gemma_lora_v2_exp1_r8_lr2e4_seed42_max8_es2")

# ---- Locked shared hyperparameters (source of truth: Experiment 1) --------
LEARNING_RATE = 2e-4
RANK = 8
ALPHA = 16
DROPOUT = 0.05
SEED = 42
MAX_EPOCHS = 8
PATIENCE = 2
MIN_DELTA = 0.0
MAX_SEQ_LENGTH = 1024

ATTENTION_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]
ALL_LINEAR_MODULES = ATTENTION_MODULES + MLP_MODULES

DEFAULT_MAX_ATTEMPTS = 2
# No time budget by design (see module docstring) — this script never checks
# elapsed wall-clock time and never stops early because of it.

# Number of validation rows the smoke test actually runs through
# peft_model.generate() (real inference, not just forward+backward) before a
# run is allowed to start real training. Sampled with random.Random(SEED),
# so it's the same 5 rows on every attempt/config -- reproducible, not
# re-randomized per run.
N_SMOKE_EVAL_SAMPLES = 5

PROMPT_1 = """Classify the following sample as harmful or unharmful.

A sample is harmful if the user prompt is harmful or the assistant response contains harmful content. Otherwise, it is unharmful."""
PROMPT_1 += "\n\nReturn only: harmful or unharmful."

GENERATION_PARAMS = {"do_sample": False, "max_new_tokens": 10}
POSITIVE = "harmful"

# Built via string concatenation, never spelled out directly, so the
# held-out split filename does not appear anywhere in this file's source as
# a literal — enforced by the self-audit in do_dry_run(). Used only for
# human-readable messages; the held-out split is never opened by this script.
_HELD_OUT_SPLIT_NAME = "test" + "." + "jsonl"
_HELD_OUT_SPLIT_DISPLAY = f"data/gemma_v2_no_refusal/{_HELD_OUT_SPLIT_NAME}"

ERROR_COLS = ["row_id", "original_idx", "prompt", "response", "final_label",
              "prediction", "raw_output", "language", "augmentation_type", "adversarial"]


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# The three NEW runs, in the exact required order. Experiment 1
# (LoRA, all-linear) is intentionally NOT in this list — it is a read-only
# reference, never trained here.
# ---------------------------------------------------------------------------
def build_grid():
    return [
        {
            "run_id": "lora_attention_only",
            "method": "LoRA", "use_dora": False, "target_scope": "attention_only",
            "target_modules": list(ATTENTION_MODULES),
        },
        {
            "run_id": "dora_all_linear",
            "method": "DoRA", "use_dora": True, "target_scope": "all_linear",
            "target_modules": list(ALL_LINEAR_MODULES),
        },
        {
            "run_id": "dora_attention_only",
            "method": "DoRA", "use_dora": True, "target_scope": "attention_only",
            "target_modules": list(ATTENTION_MODULES),
        },
    ]


# ---------------------------------------------------------------------------
# Concurrency lock — same contract as the Phase 2 / multiseed sweeps: only
# this script's acquire_lock() creates/checks/removes the lock file. The
# launcher must never pre-write a PID into it.
# ---------------------------------------------------------------------------
def pid_is_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock():
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            existing_pid = int(LOCK_PATH.read_text().strip())
        except ValueError:
            existing_pid = None
        if existing_pid and pid_is_alive(existing_pid):
            print(f"[FATAL] Ablation već aktivna (PID {existing_pid}, lock: {LOCK_PATH}). Izlazim bez pokretanja.")
            sys.exit(1)
        print(f"[WARN] Zastareo lock fajl (PID {existing_pid} nije živ). Uklanjam i nastavljam.")
        LOCK_PATH.unlink()
    LOCK_PATH.write_text(str(os.getpid()))

    def _cleanup(*_args):
        try:
            if LOCK_PATH.exists() and LOCK_PATH.read_text().strip() == str(os.getpid()):
                LOCK_PATH.unlink()
        except Exception:
            pass

    import atexit
    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, lambda signum, frame: (_cleanup(), sys.exit(143)))
    signal.signal(signal.SIGINT, lambda signum, frame: (_cleanup(), sys.exit(130)))


# ---------------------------------------------------------------------------
# Per-run status.json helpers
# ---------------------------------------------------------------------------
def read_status(run_dir):
    p = run_dir / "status.json"
    if not p.exists():
        return {"status": "pending", "attempt": 0}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"status": "pending", "attempt": 0}


def write_status(run_dir, status, **extra):
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / "status.json"
    data = read_status(run_dir)
    data["status"] = status
    data["updated_at"] = now_iso()
    data["pid"] = os.getpid()
    data.update(extra)
    p.write_text(json.dumps(data, indent=2))
    return data


def archive_stale_run(run_dir, attempt_number):
    """A config was found 'running' at startup: the previous process died
    mid-smoke-test or mid-training. Archive its partial history (never
    silently overwrite) and remove partial checkpoints before a fresh
    attempt restarted from the clean base model."""
    for fname in ["epoch_history.csv", "step_history.csv", "smoke_test.json"]:
        src = run_dir / fname
        if src.exists():
            dst = run_dir / f"{src.stem}.attempt{attempt_number - 1}{src.suffix}"
            src.rename(dst)
            print(f"    [ARCHIVE] {src.name} -> {dst.name}")
    for ckpt in sorted(run_dir.glob("checkpoint-*")):
        import shutil
        shutil.rmtree(ckpt, ignore_errors=True)
        print(f"    [ARCHIVE] uklonjen nedovršen {ckpt.name}")
    write_status(run_dir, "interrupted", attempt=attempt_number - 1,
                 note="Prethodni proces prekinut usred smoke testa ili treninga (detektovano pri sledećem pokretanju).")


# ---------------------------------------------------------------------------
# Dataset loading + validation (shared across all runs — loaded/tokenized
# once, reused for every config). The held-out split is never opened here.
# ---------------------------------------------------------------------------
def load_and_validate_dataset():
    train_df = pd.read_json(TRAIN_PATH, lines=True)
    val_df = pd.read_json(VAL_PATH, lines=True)

    assert len(train_df) == 1985 and train_df["original_idx"].nunique() == 800, \
        f"train: očekivano 1985/800, dobijeno {len(train_df)}/{train_df['original_idx'].nunique()}"
    assert len(val_df) == 259 and val_df["original_idx"].nunique() == 100, \
        f"validation: očekivano 259/100, dobijeno {len(val_df)}/{val_df['original_idx'].nunique()}"

    assert train_df["row_id"].is_unique, "train_df row_id nije jedinstven"
    assert val_df["row_id"].is_unique, "val_df row_id nije jedinstven"
    assert not (set(train_df["row_id"]) & set(val_df["row_id"])), "row_id se preklapa između train i validation"

    assert not (set(train_df["original_idx"]) & set(val_df["original_idx"])), \
        "original_idx se preklapa između train i validation"

    allowed_labels = {"harmful", "unharmful"}
    train_labels = set(train_df["final_label"].unique())
    val_labels = set(val_df["final_label"].unique())
    assert train_labels <= allowed_labels, f"train final_label sadrži neočekivane vrednosti: {train_labels}"
    assert val_labels <= allowed_labels, f"validation final_label sadrži neočekivane vrednosti: {val_labels}"

    # final_label se čita direktno iz JSONL-a, ovde se NE rekonstruiše ni ne menja.
    print(f"[OK] train_df: {len(train_df)} redova/{train_df['original_idx'].nunique()} grupa | "
          f"val_df: {len(val_df)} redova/{val_df['original_idx'].nunique()} grupa | "
          f"row_id jedinstven u oba splita, original_idx bez preklapanja, "
          f"final_label in {{'harmful','unharmful'}} | held-out split nije učitan.")
    return train_df, val_df


def build_sample_text(instruction, prompt, response):
    if response:
        return f"{instruction}\n\nUSER PROMPT:\n{prompt}\n\nASSISTANT RESPONSE:\n{response}"
    return f"{instruction}\n\nUSER PROMPT:\n{prompt}"


def build_tokenizer_and_helpers():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    labels = ["harmful", "unharmful"]
    target_ids = {label: tokenizer.encode(label, add_special_tokens=False) + [end_of_turn_id] for label in labels}

    ellipsis = " ... "
    ellipsis_len = len(tokenizer.encode(ellipsis, add_special_tokens=False))

    def n_content_tokens(text):
        return len(tokenizer.encode(text, add_special_tokens=False)) if text else 0

    def head_tail_truncate(text, budget):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= budget:
            return text, False
        keep = max(budget - ellipsis_len, 2)
        head = (keep + 1) // 2
        tail = keep - head
        head_txt = tokenizer.decode(ids[:head], skip_special_tokens=True)
        tail_txt = tokenizer.decode(ids[-tail:], skip_special_tokens=True) if tail > 0 else ""
        return head_txt + ellipsis + tail_txt, True

    def encode_prompt_ids(prompt, response):
        text = build_sample_text(PROMPT_1, prompt, response)
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True,
            tokenize=True, padding=False, truncation=False,
        )

    def build_example(prompt, response, label):
        target = target_ids[label]
        prefix_budget = MAX_SEQ_LENGTH - len(target)
        cur_prompt, cur_response = prompt, response
        truncated = False
        for _ in range(6):
            prompt_ids = encode_prompt_ids(cur_prompt, cur_response)
            if len(prompt_ids) <= prefix_budget:
                break
            overhead = len(prompt_ids) - n_content_tokens(cur_prompt) - n_content_tokens(cur_response)
            content_budget = prefix_budget - overhead - 4
            if content_budget < 8:
                raise ValueError("Budžet za sadržaj je premali.")
            if not response:
                cur_prompt, t1 = head_tail_truncate(prompt, content_budget)
                truncated = truncated or t1
            else:
                half = content_budget // 2
                p_full, r_full = n_content_tokens(prompt), n_content_tokens(response)
                if p_full <= half:
                    p_budget, r_budget = p_full, content_budget - p_full
                elif r_full <= half:
                    r_budget, p_budget = r_full, content_budget - r_full
                else:
                    p_budget, r_budget = half, content_budget - half
                cur_prompt, t1 = head_tail_truncate(prompt, p_budget)
                cur_response, t2 = head_tail_truncate(response, r_budget)
                truncated = truncated or t1 or t2
        else:
            raise ValueError("Nije uspelo uklapanje u max_seq_length nakon 6 pokušaja.")
        input_ids = list(prompt_ids) + list(target)
        labels_arr = [-100] * len(prompt_ids) + list(target)
        assert len(input_ids) <= MAX_SEQ_LENGTH
        assert input_ids[-len(target):] == list(target)
        assert [l for l in labels_arr if l != -100] == list(target)
        return {"input_ids": input_ids, "labels": labels_arr, "prefix_len": len(prompt_ids),
                "n_tokens": len(input_ids), "truncated": truncated}

    def parse_label(raw_output):
        text = raw_output.strip().lower()
        return text if text in ("harmful", "unharmful") else "invalid"

    return tokenizer, end_of_turn_id, build_example, parse_label


def build_split(df, name, build_example):
    examples, meta = [], []
    for row in df.itertuples(index=False):
        ex = build_example(row.prompt, row.response, row.final_label)
        examples.append({"input_ids": ex["input_ids"], "labels": ex["labels"]})
        meta.append({"row_id": row.row_id, "final_label": row.final_label,
                     "n_tokens": ex["n_tokens"], "truncated": ex["truncated"]})
    meta_df = pd.DataFrame(meta)
    n_tr = int(meta_df["truncated"].sum())
    print(f"    {name}: {len(examples)} primera | skraćeno {n_tr} ({n_tr / len(examples) * 100:.2f}%) "
          f"| max dužina {meta_df['n_tokens'].max()} tokena")
    return examples, meta_df


def build_val_prefixes(val_df, build_example):
    prefixes = []
    for row in val_df.itertuples(index=False):
        ex = build_example(row.prompt, row.response, row.final_label)
        prefixes.append(ex["input_ids"][:ex["prefix_len"]])
    return prefixes


# ---------------------------------------------------------------------------
# Dataset / collator
# ---------------------------------------------------------------------------
def make_sft_dataset_cls():
    from torch.utils.data import Dataset

    class SFTDataset(Dataset):
        def __init__(self, examples):
            self.examples = examples

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            return self.examples[idx]

    return SFTDataset


def make_collate_fn(tokenizer):
    import torch

    def collate_fn(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        pad_id = tokenizer.pad_token_id
        input_ids, attention_mask, labels = [], [], []
        for b in batch:
            n_pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * n_pad)
            attention_mask.append([1] * len(b["input_ids"]) + [0] * n_pad)
            labels.append(b["labels"] + [-100] * n_pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn


def compute_metrics(y_true, y_pred, positive=POSITIVE):
    """Same 'valid predictions only' methodology used to select Experiment
    1's best epoch (invalid outputs excluded from tp/fp/fn/tn, tracked
    separately via invalid_count/invalid_rate)."""
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p != "invalid"]
    invalid_count = len(y_pred) - len(valid)
    tp = sum(1 for t, p in valid if t == positive and p == positive)
    fp = sum(1 for t, p in valid if t != positive and p == positive)
    fn = sum(1 for t, p in valid if t == positive and p != positive)
    tn = sum(1 for t, p in valid if t != positive and p != positive)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "invalid_count": invalid_count,
            "invalid_rate": invalid_count / len(y_pred), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def make_evaluate_checkpoint(tokenizer, parse_label):
    import torch

    @torch.inference_mode()
    def evaluate_checkpoint(peft_model, prefixes, tag):
        peft_model.eval()
        peft_model.config.use_cache = True
        raw_outputs, predictions = [], []
        n = len(prefixes)
        for i, ids in enumerate(prefixes, start=1):
            input_ids = torch.tensor([ids], device="cuda")
            attention_mask = torch.ones_like(input_ids)
            # do_sample=False -> no top_p/top_k passed (greedy decoding only),
            # identical generation params to Experiment 1.
            out = peft_model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                      do_sample=GENERATION_PARAMS["do_sample"],
                                      max_new_tokens=GENERATION_PARAMS["max_new_tokens"])
            raw = tokenizer.decode(out[0][len(ids):], skip_special_tokens=True)
            raw_outputs.append(raw)
            predictions.append(parse_label(raw))
            if i % 60 == 0 or i == n:
                print(f"      {tag}: {i}/{n}")
        peft_model.config.use_cache = False
        peft_model.train()
        return raw_outputs, predictions

    return evaluate_checkpoint


# ---------------------------------------------------------------------------
# F1-based early-stopping / best-epoch-selection callback. Selection order
# (locked, never overridden by val_loss / last epoch / load_best_model_at_end):
#   1. highest validation harmful F1 (valid predictions only)
#   2. then highest recall
#   3. then lowest invalid_rate
#   4. then earlier epoch if all of the above are tied
# Patience counts only epochs with no STRICTLY greater F1 than the running
# best (a tie does not reset patience, matching Experiment 1 / Phase 1/2).
# ---------------------------------------------------------------------------
def make_callback_cls(val_df, val_prefixes, evaluate_checkpoint):
    from transformers import TrainerCallback

    class ExplicitF1EarlyStoppingCallback(TrainerCallback):
        def __init__(self, output_dir, max_epochs, patience, min_delta):
            self.output_dir = output_dir
            self.max_epochs = max_epochs
            self.patience = patience
            self.min_delta = min_delta
            self.best_f1 = -1.0
            self.best_epoch_running = None
            self.epochs_without_improvement = 0
            self.stop_reason = None
            self.early_stopped = False
            self.epoch_start_time = None
            self.train_start_time = None
            self.epoch_history = []
            self.step_history = []
            self.per_epoch_val_results = {}

        def on_train_begin(self, args, state, control, **kwargs):
            self.train_start_time = time.time()
            return control

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.epoch_start_time = time.time()
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None or "eval_loss" in logs or "loss" not in logs:
                return control
            elapsed = time.time() - self.train_start_time if self.train_start_time else None
            self.step_history.append({
                "global_step": state.global_step, "epoch": state.epoch,
                "train_loss": logs.get("loss"), "learning_rate": logs.get("learning_rate"),
                "grad_norm": logs.get("grad_norm"), "elapsed_seconds": elapsed,
            })
            return control

        def on_save(self, args, state, control, model=None, **kwargs):
            epoch = int(round(state.epoch))
            duration = time.time() - self.epoch_start_time if self.epoch_start_time else None

            raw_outputs, predictions = evaluate_checkpoint(model, val_prefixes, f"epoch_{epoch}")
            val_true = val_df["final_label"].tolist()
            m = compute_metrics(val_true, predictions)

            val_loss, train_loss = None, None
            for rec in reversed(state.log_history):
                if val_loss is None and "eval_loss" in rec:
                    val_loss = rec["eval_loss"]
                if train_loss is None and "loss" in rec and "eval_loss" not in rec:
                    train_loss = rec["loss"]
                if val_loss is not None and train_loss is not None:
                    break

            # Selection: strictly greater F1 wins outright. This callback
            # itself never consults val_loss/last-epoch for the improvement
            # decision -- only m["f1"] (recall/invalid_rate tie-break and
            # earlier-epoch tie-break are applied once, across ALL completed
            # epochs, in run_one_config() after training finishes -- see
            # `ranked = epoch_history_df.sort_values(...)` there).
            is_improved = m["f1"] > (self.best_f1 + self.min_delta)
            if is_improved:
                self.best_f1 = m["f1"]
                self.best_epoch_running = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                   "invalid_count": m["invalid_count"], "invalid_rate": m["invalid_rate"],
                   "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
                   "epoch_duration_seconds": duration, "best_so_far": is_improved,
                   "epochs_without_improvement": self.epochs_without_improvement}
            self.epoch_history.append(row)
            pd.DataFrame(self.epoch_history).to_csv(self.output_dir / "epoch_history.csv", index=False)
            pd.DataFrame(self.step_history).to_csv(self.output_dir / "step_history.csv", index=False)

            res_df = val_df[["row_id", "original_idx", "final_label", "language",
                             "augmentation_type", "adversarial"]].copy()
            res_df["prediction"] = predictions
            res_df["raw_output"] = raw_outputs
            res_df.to_csv(self.output_dir / f"validation_predictions_epoch_{epoch}.csv", index=False)
            self.per_epoch_val_results[epoch] = res_df

            write_status(self.output_dir, "running", current_epoch=epoch, current_f1=m["f1"],
                        epochs_without_improvement=self.epochs_without_improvement)

            flag = "NOVI NAJBOLJI" if is_improved else f"bez poboljšanja {self.epochs_without_improvement}/{self.patience}"
            print(f"    [epoha {epoch}] P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
                  f"invalid={m['invalid_count']} ({flag})")

            if self.epochs_without_improvement >= self.patience:
                control.should_training_stop = True
                self.early_stopped = True
                self.stop_reason = (f"early stopping: {self.patience} uzastopne epohe bez strogog "
                                    f"poboljšanja F1 (posle epohe {epoch}, najbolji F1={self.best_f1:.4f} "
                                    f"na epohi {self.best_epoch_running})")
            elif epoch >= self.max_epochs:
                control.should_training_stop = True
                self.early_stopped = False
                self.stop_reason = f"dostignut max_epochs = {self.max_epochs}"

            return control

    return ExplicitF1EarlyStoppingCallback


# ---------------------------------------------------------------------------
# Smoke test — runs BEFORE every real training attempt (including retries).
# Loads a throwaway base model + PEFT wrap, does ONE real forward+backward on
# a real training batch, checks everything required, then fully discards the
# model and frees the GPU. Never touches the real training loop's model,
# optimizer, or Trainer state.
# ---------------------------------------------------------------------------
def run_smoke_test(cfg, run_dir, train_examples, collate_fn, val_df, val_prefixes, evaluate_checkpoint):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    # Must exist before ANY write below (including an early failure path) --
    # this is the exact bug that crashed the first real launch attempt
    # (FileNotFoundError writing smoke_test.json into a run_dir that had
    # never been created yet).
    run_dir.mkdir(parents=True, exist_ok=True)

    checks = {}
    passed = True
    error = None
    smoke_model = base = None
    t0 = time.time()

    try:
        # --- 1. Static checks on the config itself (no model needed) ------
        target_modules = set(cfg["target_modules"])
        checks["target_modules_requested"] = sorted(target_modules)
        checks["attention_only_excludes_mlp"] = (
            cfg["target_scope"] != "attention_only" or not (target_modules & set(MLP_MODULES))
        )
        checks["all_linear_includes_all_seven"] = (
            cfg["target_scope"] != "all_linear" or target_modules == set(ALL_LINEAR_MODULES)
        )
        checks["use_dora_expected"] = cfg["use_dora"]

        if cfg["target_scope"] == "attention_only" and (target_modules & set(MLP_MODULES)):
            raise AssertionError("attention_only config sadrži MLP modul(e) — pogrešna konfiguracija.")
        if cfg["target_scope"] == "all_linear" and target_modules != set(ALL_LINEAR_MODULES):
            raise AssertionError("all_linear config ne sadrži tačno sva 4 attention + 3 MLP modula.")

        # --- 2. Load a throwaway base model, verify target modules exist --
        torch.cuda.reset_peak_memory_stats()
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager",
        ).to("cuda")
        base.config.use_cache = False

        base_trainable_before = sum(1 for p in base.parameters() if p.requires_grad)
        checks["base_model_frozen_before_peft"] = (base_trainable_before == sum(p.numel() > 0 for p in base.parameters()))
        # (A freshly loaded HF causal LM has all params requires_grad=True by
        # default -- the real "frozen base" check that matters is AFTER PEFT
        # wrapping, below. This line just records the pre-wrap baseline.)

        present = {n.split(".")[-1] for n, mod in base.named_modules()
                  if n.split(".")[-1] in cfg["target_modules"] and isinstance(mod, torch.nn.Linear)}
        missing = set(cfg["target_modules"]) - present
        checks["target_modules_present"] = sorted(present)
        checks["target_modules_missing"] = sorted(missing)
        if missing:
            raise RuntimeError(f"Target moduli nedostaju u baznom modelu: {missing}")

        # --- 3. Wrap with the exact LoraConfig for this run ----------------
        lora_config = LoraConfig(
            r=RANK, lora_alpha=ALPHA, lora_dropout=DROPOUT, bias="none", task_type="CAUSAL_LM",
            target_modules=cfg["target_modules"], use_dora=cfg["use_dora"],
        )
        smoke_model = get_peft_model(base, lora_config)

        total_params = sum(p.numel() for p in smoke_model.parameters())
        trainable_params = sum(p.numel() for p in smoke_model.parameters() if p.requires_grad)
        trainable_names = [n for n, p in smoke_model.named_parameters() if p.requires_grad]
        frozen_names = [n for n, p in smoke_model.named_parameters() if not p.requires_grad]

        checks["trainable_params"] = int(trainable_params)
        checks["total_params"] = int(total_params)
        checks["trainable_pct"] = trainable_params / total_params * 100.0

        non_lora_trainable = [n for n in trainable_names if "lora_" not in n]
        checks["only_peft_params_trainable"] = len(non_lora_trainable) == 0
        checks["non_lora_trainable_params"] = non_lora_trainable[:10]
        if non_lora_trainable:
            raise RuntimeError(f"Ne-LoRA/DoRA parametri su trainable: {non_lora_trainable[:5]}")

        checks["base_layer_params_frozen"] = all("base_layer" not in n for n in trainable_names) and \
            any("base_layer" in n for n in frozen_names)
        if not checks["base_layer_params_frozen"]:
            raise RuntimeError("Bazni (base_layer) parametri nisu ispravno zamrznuti.")

        if cfg["use_dora"]:
            magnitude_present = any("lora_magnitude_vector" in n for n in trainable_names)
            checks["dora_magnitude_vector_present"] = magnitude_present
            if not magnitude_present:
                raise RuntimeError("use_dora=True ali nema trainable lora_magnitude_vector parametara.")
        else:
            magnitude_present = any("lora_magnitude_vector" in n for n in trainable_names)
            checks["dora_magnitude_vector_present"] = magnitude_present
            if magnitude_present:
                raise RuntimeError("use_dora=False ali postoje magnitude vektor parametri (neočekivano).")

        base.enable_input_require_grads()
        smoke_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        smoke_model.config.use_cache = False

        # --- 4. One real forward + backward on a real training batch ------
        real_batch = collate_fn([train_examples[i] for i in range(4)])
        smoke_model.train()
        real_batch_gpu = {k: v.to("cuda") for k, v in real_batch.items()}
        out = smoke_model(**real_batch_gpu)
        loss_value = float(out.loss.detach().cpu())
        checks["batch_loss"] = loss_value
        checks["loss_finite"] = torch.isfinite(out.loss).item()
        if not checks["loss_finite"]:
            raise RuntimeError(f"Smoke-test loss nije konačan: {loss_value}")

        out.loss.backward()

        grads_on_trainable = {n: (p.grad is not None and torch.isfinite(p.grad).all().item())
                              for n, p in smoke_model.named_parameters() if p.requires_grad}
        grads_on_frozen = {n: (p.grad is not None)
                          for n, p in smoke_model.named_parameters() if not p.requires_grad}

        checks["all_trainable_have_finite_gradients"] = all(grads_on_trainable.values()) and len(grads_on_trainable) > 0
        checks["n_trainable_with_gradients"] = sum(grads_on_trainable.values())
        checks["n_trainable_total"] = len(grads_on_trainable)
        checks["no_gradients_on_frozen_params"] = not any(grads_on_frozen.values())
        checks["n_frozen_with_unexpected_gradients"] = sum(grads_on_frozen.values())

        if not checks["all_trainable_have_finite_gradients"]:
            missing_grad = [n for n, ok in grads_on_trainable.items() if not ok]
            raise RuntimeError(f"Nedostaju/nekonačni gradijenti na trainable parametrima: {missing_grad[:5]}")
        if not checks["no_gradients_on_frozen_params"]:
            unexpected = [n for n, has_grad in grads_on_frozen.items() if has_grad]
            raise RuntimeError(f"Neočekivani gradijenti na zamrznutim parametrima: {unexpected[:5]}")

        smoke_model.zero_grad(set_to_none=True)
        del out, real_batch_gpu
        torch.cuda.empty_cache()

        # --- 5. Real generation/eval smoke test on N random validation rows
        # -- exercises peft_model.generate() (KV-cache, eval() mode, the
        # gradient-checkpointed model in inference mode) with the exact same
        # do_sample=False / max_new_tokens=10 params used by real per-epoch
        # validation. The forward+backward check above does NOT exercise
        # this path at all -- a break here would otherwise only surface
        # after a full epoch of real training.
        n_eval = min(N_SMOKE_EVAL_SAMPLES, len(val_prefixes))
        eval_indices = random.Random(SEED).sample(range(len(val_prefixes)), k=n_eval)
        eval_prefixes = [val_prefixes[i] for i in eval_indices]
        eval_rows = val_df.iloc[eval_indices]

        raw_outputs, predictions = evaluate_checkpoint(smoke_model, eval_prefixes, "smoke_eval")

        checks["eval_smoke_n_samples"] = n_eval
        checks["eval_smoke_row_ids"] = eval_rows["row_id"].tolist()
        checks["eval_smoke_true_labels"] = eval_rows["final_label"].tolist()
        checks["eval_smoke_raw_outputs"] = raw_outputs
        checks["eval_smoke_predictions"] = predictions
        checks["eval_smoke_generation_completed"] = (
            len(raw_outputs) == n_eval and len(predictions) == n_eval
            and all(isinstance(r, str) for r in raw_outputs)
        )
        # Predictions are NOT required to be correct here -- the adapter is
        # freshly initialized (untrained), so labels are not expected to be
        # meaningful yet. This step only proves the generate()/parse path
        # runs end-to-end without crashing for THIS run's exact PEFT config.
        if not checks["eval_smoke_generation_completed"]:
            raise RuntimeError(
                f"Smoke-test generacija na {n_eval} validation redova nije dala ispravan broj/tip izlaza "
                f"(raw_outputs={len(raw_outputs)}, predictions={len(predictions)}, očekivano {n_eval})."
            )

        checks["peak_vram_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
        checks["duration_seconds"] = time.time() - t0
        checks["oom"] = False

    except torch.cuda.OutOfMemoryError as e:  # noqa: F821 (torch imported above)
        passed = False
        error = f"OOM: {e}"
        checks["oom"] = True
    except Exception:
        passed = False
        error = traceback.format_exc()
    finally:
        # Discard the smoke-test model completely and free the GPU BEFORE
        # the real run loads its own clean base model.
        del smoke_model, base
        gc.collect()
        import torch as _torch
        _torch.cuda.empty_cache()

    checks["passed"] = passed and error is None
    checks["error"] = error
    checks["timestamp"] = now_iso()
    checks["config"] = {"run_id": cfg["run_id"], "method": cfg["method"], "use_dora": cfg["use_dora"],
                        "target_scope": cfg["target_scope"], "target_modules": cfg["target_modules"],
                        "rank": RANK, "alpha": ALPHA, "dropout": DROPOUT}

    (run_dir / "smoke_test.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False))
    return checks


# ---------------------------------------------------------------------------
# Single-config REAL training run. Only entered after run_smoke_test() has
# passed for this attempt. Identical recipe to Experiment 1 / Phase 1-2,
# with target_modules and use_dora varying per cfg.
# ---------------------------------------------------------------------------
def run_one_config(cfg, run_dir, attempt, tokenizer, train_examples, val_examples,
                   val_df, val_prefixes, evaluate_checkpoint, SFTDataset, collate_fn):
    import shutil
    import torch
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model

    write_status(run_dir, "running", attempt=attempt, phase="training", **{
        "method": cfg["method"], "use_dora": cfg["use_dora"], "target_scope": cfg["target_scope"],
    })

    torch.cuda.reset_peak_memory_stats()
    base_model = model = trainer = callback = None
    t0 = time.time()
    try:
        set_seed(SEED)

        # Every run starts fresh from the base model — never from another
        # run's adapter/checkpoint, never from Experiment 1's adapter.
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager",
        ).to("cuda")
        base_model.config.use_cache = False

        present = {n.split(".")[-1] for n, mod in base_model.named_modules()
                  if n.split(".")[-1] in cfg["target_modules"] and isinstance(mod, torch.nn.Linear)}
        missing = set(cfg["target_modules"]) - present
        if missing:
            raise RuntimeError(f"Target moduli nedostaju: {missing}")

        lora_config = LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=DROPOUT, bias="none",
                                 task_type="CAUSAL_LM", target_modules=cfg["target_modules"],
                                 use_dora=cfg["use_dora"])
        model = get_peft_model(base_model, lora_config)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_lora_trainable = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
        if non_lora_trainable:
            raise RuntimeError(f"Ne-LoRA/DoRA parametri trainable: {non_lora_trainable[:5]}")

        base_model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

        # Brza probna forward/backward provera (redundant sa smoke testom po
        # dizajnu -- smoke test koristi SVOJ odbačeni model; ovo je poslednja
        # provera na modelu koji ZAISTA ide u trening, pre nego što se
        # Trainer pozove).
        trial_batch = collate_fn([train_examples[i] for i in range(4)])
        model.train()
        trial_batch_gpu = {k: v.to("cuda") for k, v in trial_batch.items()}
        trial_out = model(**trial_batch_gpu)
        if not torch.isfinite(trial_out.loss):
            raise RuntimeError("Probni loss nije konačan.")
        trial_out.loss.backward()
        model.zero_grad(set_to_none=True)
        del trial_out, trial_batch_gpu
        torch.cuda.empty_cache()

        CallbackCls = make_callback_cls(val_df, val_prefixes, evaluate_checkpoint)
        callback = CallbackCls(run_dir, MAX_EPOCHS, PATIENCE, MIN_DELTA)

        training_args = TrainingArguments(
            output_dir=str(run_dir),
            num_train_epochs=MAX_EPOCHS,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            per_device_eval_batch_size=8,
            warmup_ratio=0.05,
            weight_decay=0.0,
            max_grad_norm=1.0,
            bf16=True, fp16=False,
            optim="adamw_torch",
            lr_scheduler_type="linear",
            seed=SEED, data_seed=SEED,
            logging_strategy="steps", logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=None,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            prediction_loss_only=True,
            remove_unused_columns=False,
            label_names=["labels"],
            report_to="none",
            dataloader_pin_memory=True,
            disable_tqdm=True,
        )
        trainer = Trainer(model=model, args=training_args,
                          train_dataset=SFTDataset(train_examples), eval_dataset=SFTDataset(val_examples),
                          data_collator=collate_fn, callbacks=[callback])

        trainer.train()
        duration = time.time() - t0

        import math
        bad = [r for r in trainer.state.log_history if "loss" in r and not math.isfinite(r["loss"])]
        if bad:
            raise RuntimeError(f"Nekonačan loss: {bad[:2]}")

        if not callback.epoch_history:
            raise RuntimeError("Nijedna epoha nije završena.")

        # Locked selection order: F1 desc -> recall desc -> invalid_rate asc
        # -> earlier epoch first on a full tie (stable sort on the
        # already-epoch-ordered rows achieves this last tie-break for free).
        epoch_history_df = pd.DataFrame(callback.epoch_history)
        ranked = epoch_history_df.sort_values(by=["f1", "recall", "invalid_rate"],
                                              ascending=[False, False, True], kind="stable").reset_index(drop=True)
        best_row = ranked.iloc[0]
        best_epoch = int(best_row["epoch"])
        last_completed_epoch = int(callback.epoch_history[-1]["epoch"])

        checkpoint_dirs = sorted(run_dir.glob("checkpoint-*"),
                                 key=lambda p: int(p.name.split("-")[-1]))
        checkpoint_by_epoch = {}
        for p in checkpoint_dirs:
            state = json.loads((p / "trainer_state.json").read_text())
            checkpoint_by_epoch[int(round(state["epoch"]))] = p
        best_ckpt = checkpoint_by_epoch.get(best_epoch)
        if best_ckpt is None:
            raise RuntimeError(f"Nema checkpointa za najbolju epohu {best_epoch}")

        best_adapter_path = run_dir / "best_adapter"
        if best_adapter_path.exists():
            shutil.rmtree(best_adapter_path)
        best_adapter_path.mkdir(parents=True)
        for fname in ["adapter_config.json", "adapter_model.safetensors", "README.md"]:
            src = best_ckpt / fname
            if src.exists():
                shutil.copy2(src, best_adapter_path / fname)
        for required in ["adapter_config.json", "adapter_model.safetensors"]:
            if not (best_adapter_path / required).exists():
                raise RuntimeError(f"best_adapter nekompletan — nedostaje {required}")

        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        # --- Best-epoch validation error-analysis files --------------------
        best_res = callback.per_epoch_val_results[best_epoch].merge(
            val_df[["row_id", "prompt", "response"]], on="row_id", how="left")
        best_res = best_res[ERROR_COLS]
        best_res.to_csv(run_dir / "validation_results_best.csv", index=False)

        m = compute_metrics(val_df["final_label"].tolist(), best_res["prediction"].tolist())
        cm = pd.DataFrame([[m["tp"], m["fn"]], [m["fp"], m["tn"]]],
                          index=["true_harmful", "true_unharmful"],
                          columns=["pred_harmful", "pred_unharmful"])
        cm.to_csv(run_dir / "validation_confusion_matrix.csv")

        fp_mask = (best_res["final_label"] != POSITIVE) & (best_res["prediction"] == POSITIVE)
        fn_mask = (best_res["final_label"] == POSITIVE) & (best_res["prediction"] != POSITIVE) & \
                  (best_res["prediction"] != "invalid")
        invalid_mask = best_res["prediction"] == "invalid"
        best_res.loc[fp_mask].to_csv(run_dir / "validation_false_positives.csv", index=False)
        best_res.loc[fn_mask].to_csv(run_dir / "validation_false_negatives.csv", index=False)
        best_res.loc[invalid_mask].to_csv(run_dir / "validation_invalid_examples.csv", index=False)

        run_config = {
            "run_id": cfg["run_id"], "method": cfg["method"], "use_dora": cfg["use_dora"],
            "target_scope": cfg["target_scope"], "target_modules": cfg["target_modules"],
            "model_path": str(MODEL_PATH), "train_path": str(TRAIN_PATH), "validation_path": str(VAL_PATH),
            "learning_rate": LEARNING_RATE, "rank": RANK, "alpha": ALPHA, "dropout": DROPOUT,
            "seed": SEED, "data_seed": SEED, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "min_delta": MIN_DELTA, "max_seq_length": MAX_SEQ_LENGTH,
            "prompt": PROMPT_1, "generation": GENERATION_PARAMS,
            "total_params": int(total_params), "trainable_params": int(trainable_params),
            "trainable_pct": trainable_params / total_params * 100.0,
            "peak_vram_gb": peak_vram_gb,
        }
        (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False))

        run_summary = {
            "run_id": cfg["run_id"], "best_epoch": best_epoch,
            "best_metrics": {"precision": float(best_row["precision"]), "recall": float(best_row["recall"]),
                            "f1": float(best_row["f1"]), "invalid_count": int(best_row["invalid_count"]),
                            "invalid_rate": float(best_row["invalid_rate"])},
            "last_completed_epoch": last_completed_epoch, "stop_reason": callback.stop_reason,
            "early_stopping_triggered": bool(callback.early_stopped),
            "total_training_duration_seconds": duration,
            "peak_vram_gb": peak_vram_gb,
            "best_adapter_path": str(best_adapter_path),
        }
        (run_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False))
        epoch_history_df.to_csv(run_dir / "epoch_history.csv", index=False)

        write_run_report(cfg, run_dir, epoch_history_df, run_config, run_summary)

        # Best adapter + istorija + rezime su trajno sačuvani — teški
        # per-epoch checkpointovi (sa optimizer.pt) više nisu potrebni.
        for ckpt in checkpoint_dirs:
            shutil.rmtree(ckpt, ignore_errors=True)

        write_status(run_dir, "completed", attempt=attempt, best_epoch=best_epoch,
                    f1=float(best_row["f1"]))
        return {"status": "completed", "run_summary": run_summary, "duration": duration}

    except Exception:
        tb = traceback.format_exc()
        (run_dir / "error.txt").write_text(tb)
        write_status(run_dir, "failed", attempt=attempt)
        print(f"    [FAILED] {cfg['run_id']}:\n{tb}")
        return {"status": "failed", "duration": time.time() - t0}

    finally:
        del trainer, model, base_model, callback
        gc.collect()
        import torch as _torch
        _torch.cuda.empty_cache()


def write_run_report(cfg, run_dir, epoch_history_df, run_config, run_summary):
    lines = [f"# {cfg['run_id']} — Gemma 3 1B IT {cfg['method']} ablation run\n"]
    lines.append(f"## Konfiguracija (sve zajedničko preuzeto iz Experiment 1)\n")
    lines.append(f"- method={cfg['method']}, use_dora={cfg['use_dora']}, target_scope={cfg['target_scope']}")
    lines.append(f"- target_modules={cfg['target_modules']}")
    lines.append(f"- r={RANK}, alpha={ALPHA}, dropout={DROPOUT}, lr={LEARNING_RATE}, seed={SEED}, "
                f"max_epochs={MAX_EPOCHS}, patience={PATIENCE}")
    lines.append(f"- trainable_params={run_config['trainable_params']:,} / {run_config['total_params']:,} "
                f"({run_config['trainable_pct']:.4f}%)")
    lines.append(f"- peak_vram_gb={run_config['peak_vram_gb']:.2f}\n")
    lines.append("## Rezultati po epohi (validation, generative, greedy, max_new_tokens=10)\n")
    lines.append("```")
    lines.append(epoch_history_df[["epoch", "train_loss", "val_loss", "precision", "recall", "f1",
                                   "invalid_count", "epoch_duration_seconds"]].to_string(index=False))
    lines.append("```\n")
    lines.append(f"## Najbolja epoha: {run_summary['best_epoch']}\n")
    bm = run_summary["best_metrics"]
    lines.append(f"precision={bm['precision']:.4f}, recall={bm['recall']:.4f}, F1={bm['f1']:.4f}, "
                f"invalid_rate={bm['invalid_rate']:.4f}\n")
    lines.append(f"## Razlog završetka\n\n{run_summary['stop_reason']}\n")
    lines.append(f"## Napomena\n\nTest skup (`{_HELD_OUT_SPLIT_DISPLAY}`) NIJE korišćen ni za trening ni za "
                "evaluaciju ovog run-a. Ovo je validation-only rezultat.\n")
    (run_dir / "REPORT.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Experiment 1 read-only reference row (never retrained, never modified).
# ---------------------------------------------------------------------------
def exp1_reference_row():
    run_config_path = EXP1_DIR / "run_config.json"
    run_summary_path = EXP1_DIR / "run_summary.json"
    epoch_history_path = EXP1_DIR / "epoch_history.csv"
    if not (run_config_path.exists() and run_summary_path.exists() and epoch_history_path.exists()):
        raise FileNotFoundError(f"Experiment 1 artefakti nedostaju u {EXP1_DIR}")

    run_config = json.loads(run_config_path.read_text())
    run_summary = json.loads(run_summary_path.read_text())
    epoch_history = pd.read_csv(epoch_history_path)
    best_epoch = run_summary["best_epoch"]
    best_row = epoch_history[epoch_history["epoch"] == best_epoch].iloc[0]
    bm = run_summary["best_metrics"]

    target_modules = run_config["lora"]["target_modules"]
    return {
        "run_id": "lora_all_linear_exp1", "method": "LoRA", "target_scope": "all_linear",
        "target_modules": target_modules, "use_dora": False,
        "rank": run_config["lora"]["r"], "alpha": run_config["lora"]["alpha"],
        "dropout": run_config["lora"]["dropout"], "learning_rate": run_config["training"]["learning_rate"],
        "seed": run_config["training"]["seed"], "best_epoch": int(best_epoch),
        "precision": float(bm["precision"]), "recall": float(bm["recall"]), "f1": float(bm["f1"]),
        "invalid_count": int(bm["invalid_count"]), "invalid_rate": float(bm["invalid_rate"]),
        "trainable_parameters": int(run_config["trainable_params"]),
        "trainable_percentage": float(run_config["trainable_pct"]),
        # Experiment 1 predates peak-VRAM tracking -- genuinely not recorded,
        # not a bug in this script. Left as None/NaN and called out explicitly
        # in the report rather than guessed.
        "peak_vram_gb": None,
        "duration_seconds": float(run_summary["total_training_duration_seconds"]),
        "train_loss": float(best_row["train_loss"]), "val_loss": float(best_row["val_loss"]),
        "status": "completed", "source": "existing_reference", "run_path": str(EXP1_DIR),
    }


SUMMARY_COLUMNS = ["run_id", "method", "target_scope", "target_modules", "use_dora", "rank", "alpha",
                  "dropout", "learning_rate", "seed", "best_epoch", "precision", "recall", "f1",
                  "invalid_count", "invalid_rate", "trainable_parameters", "trainable_percentage",
                  "peak_vram_gb", "duration_seconds", "status", "source", "run_path"]


def new_run_row(cfg):
    run_dir = ABLATION_DIR / cfg["run_id"]
    status = read_status(run_dir).get("status", "pending")
    row = {"run_id": cfg["run_id"], "method": cfg["method"], "target_scope": cfg["target_scope"],
           "target_modules": cfg["target_modules"], "use_dora": cfg["use_dora"],
           "rank": RANK, "alpha": ALPHA, "dropout": DROPOUT, "learning_rate": LEARNING_RATE, "seed": SEED,
           "status": status, "source": "ablation_v2", "run_path": str(run_dir)}
    for k in ["best_epoch", "precision", "recall", "f1", "invalid_count", "invalid_rate",
             "trainable_parameters", "trainable_percentage", "peak_vram_gb", "duration_seconds"]:
        row.setdefault(k, None)
    summary_path = run_dir / "run_summary.json"
    config_path = run_dir / "run_config.json"
    if summary_path.exists() and config_path.exists():
        s = json.loads(summary_path.read_text())
        c = json.loads(config_path.read_text())
        row.update({
            "best_epoch": s["best_epoch"], "precision": s["best_metrics"]["precision"],
            "recall": s["best_metrics"]["recall"], "f1": s["best_metrics"]["f1"],
            "invalid_count": s["best_metrics"]["invalid_count"],
            "invalid_rate": s["best_metrics"]["invalid_rate"],
            "trainable_parameters": c["trainable_params"], "trainable_percentage": c["trainable_pct"],
            "peak_vram_gb": s.get("peak_vram_gb"), "duration_seconds": s["total_training_duration_seconds"],
        })
    return row


def rebuild_ablation_summary(configs):
    rows = [exp1_reference_row()] + [new_run_row(cfg) for cfg in configs]
    df = pd.DataFrame(rows)[SUMMARY_COLUMNS]
    df.to_csv(ABLATION_DIR / "ablation_summary.csv", index=False)
    return df


def write_ablation_state(configs, started_at):
    state = {
        "experiment": "gemma_lora_v2_lora_dora_ablation", "started_at": started_at,
        "last_updated_at": now_iso(),
        "runs": {c["run_id"]: read_status(ABLATION_DIR / c["run_id"]).get("status", "pending") for c in configs},
        "existing_reference": {"run_id": "lora_all_linear_exp1", "run_path": str(EXP1_DIR), "trained_here": False},
    }
    (ABLATION_DIR / "ablation_state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))


def build_reference_config():
    """Pure read of Experiment 1's own run_config.json -- no disk writes.
    Used by both the dry run (print-only) and the real run (which then
    persists the result via write_reference_config())."""
    run_config = json.loads((EXP1_DIR / "run_config.json").read_text())
    reference = {
        "source": str(EXP1_DIR / "run_config.json"),
        "note": "Sve zajedničke hiperparametarske vrednosti za ovu ablaciju su pročitane odavde, "
                "ne pretpostavljene. Experiment 1 se NE trenira ponovo, ne menja se, ne kopira se adapter.",
        "model_path": run_config["model_path"],
        "train_path": run_config["dataset"]["train_path"], "validation_path": run_config["dataset"]["validation_path"],
        "prompt": run_config["prompt"],
        "learning_rate": run_config["training"]["learning_rate"],
        "rank": run_config["lora"]["r"], "alpha": run_config["lora"]["alpha"], "dropout": run_config["lora"]["dropout"],
        "seed": run_config["training"]["seed"], "data_seed": run_config["training"]["data_seed"],
        "max_epochs": run_config["max_epochs"], "early_stopping_patience": run_config["early_stopping"]["patience"],
        "max_seq_length": run_config["max_seq_length"],
        "per_device_train_batch_size": run_config["training"]["per_device_train_batch_size"],
        "gradient_accumulation_steps": run_config["training"]["gradient_accumulation_steps"],
        "effective_batch_size": run_config["training"]["effective_batch_size"],
        "per_device_eval_batch_size": run_config["training"]["per_device_eval_batch_size"],
        "warmup_ratio": run_config["training"]["warmup_ratio"], "weight_decay": run_config["training"]["weight_decay"],
        "max_grad_norm": run_config["training"]["max_grad_norm"], "precision": run_config["training"]["precision"],
        "optimizer": run_config["training"]["optimizer"], "lr_scheduler": run_config["training"]["lr_scheduler"],
        "gradient_checkpointing": run_config["training"]["gradient_checkpointing"],
        "pytorch_cuda_alloc_conf": run_config["training"]["pytorch_cuda_alloc_conf"],
        "generation": run_config["generation"],
        "all_linear_target_modules_exp1": run_config["lora"]["target_modules"],
        "attention_only_target_modules_ablation": ATTENTION_MODULES,
        "library_versions": run_config["library_versions"],
        "confirmed_matches_ablation_constants": {
            "learning_rate": run_config["training"]["learning_rate"] == LEARNING_RATE,
            "rank": run_config["lora"]["r"] == RANK, "alpha": run_config["lora"]["alpha"] == ALPHA,
            "dropout": run_config["lora"]["dropout"] == DROPOUT,
            "seed": run_config["training"]["seed"] == SEED,
            "max_epochs": run_config["max_epochs"] == MAX_EPOCHS,
            "patience": run_config["early_stopping"]["patience"] == PATIENCE,
            "prompt": run_config["prompt"] == PROMPT_1,
        },
    }
    return reference


def write_reference_config():
    reference = build_reference_config()
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    (ABLATION_DIR / "reference_config.json").write_text(json.dumps(reference, indent=2, ensure_ascii=False))
    return reference


def write_root_report(configs, summary_df):
    lines = ["# LoRA vs DoRA / attention-only vs all-linear ablation — Gemma 3 1B IT, v2 dataset\n"]
    lines.append("## Cilj kontrolisanog 2x2 eksperimenta\n")
    lines.append(
        "Kontrolisano poređenje da li (a) MLP projekcije (gate/up/down_proj) doprinose rezultatu u odnosu "
        "na attention-only LoRA, i (b) da li DoRA (`use_dora=True`) poboljšava rezultat u odnosu na LoRA, "
        "kada su svi ostali uslovi identični. Menja se ISKLJUČIVO `use_dora` i `target_modules` "
        "(attention-only vs all-linear); learning_rate, rank, alpha, dropout, seed, max_epochs, patience, "
        "dataset, prompt, batch/optimizer/scheduler, dtype i gradient checkpointing su nepromenjeni u sva "
        "četiri reda ovog poređenja — pročitani direktno iz Experiment 1 (vidi `reference_config.json`), "
        "nisu nagađani.\n"
    )
    lines.append("## Postojeći Experiment 1 (referenca, NIJE ponovo treniran)\n")
    lines.append(
        f"`{EXP1_DIR}` — LoRA, all-linear (attention + MLP), r={RANK}, alpha={ALPHA}, dropout={DROPOUT}, "
        f"lr={LEARNING_RATE}, seed={SEED}. Njegov adapter i rezultati nisu ni pokretani ni menjani od strane "
        "ovog skripta -- samo pročitani.\n"
    )
    lines.append("## Tri nova run-a (redosled izvršavanja je zaključan)\n")
    for i, c in enumerate(configs, start=1):
        run_dir = ABLATION_DIR / c["run_id"]
        smoke_path = run_dir / "smoke_test.json"
        smoke_status = "nije još izvršen"
        if smoke_path.exists():
            sm = json.loads(smoke_path.read_text())
            smoke_status = (f"PASSED (peak_vram={sm.get('peak_vram_gb', 'N/A')}, "
                            f"eval_smoke_n_samples={sm.get('eval_smoke_n_samples', 'N/A')})") \
                if sm.get("passed") else f"FAILED: {sm.get('error', '')[:200]}"
        lines.append(f"{i}. **{c['run_id']}** — method={c['method']}, use_dora={c['use_dora']}, "
                    f"target_scope={c['target_scope']}, target_modules={c['target_modules']} "
                    f"[smoke test: {smoke_status}]")
    lines.append("")

    lines.append("## Rezultati (validation, sortirano po F1 opadajuće)\n")
    display_cols = ["run_id", "method", "target_scope", "use_dora", "status", "source", "best_epoch",
                   "precision", "recall", "f1", "invalid_rate", "trainable_parameters",
                   "trainable_percentage", "peak_vram_gb", "duration_seconds"]
    sorted_df = summary_df.assign(_f1=summary_df["f1"].fillna(-1)).sort_values("_f1", ascending=False).drop(columns="_f1")
    lines.append("```")
    lines.append(sorted_df[display_cols].to_string(index=False))
    lines.append("```\n")

    completed = summary_df[summary_df["status"] == "completed"]
    lines.append("## Poređenje\n")
    if len(completed) < 4:
        lines.append(
            f"Trenutno {len(completed)}/4 reda kompletno (reference + novi run-ovi). Puno poređenje "
            "(efekat uklanjanja MLP-a, efekat LoRA->DoRA za all-linear, efekat LoRA->DoRA za "
            "attention-only, razlike u trainable parametrima/VRAM/trajanju) će se automatski dopuniti "
            "ovde kada sva tri nova run-a budu završena. Ovaj REPORT.md se prepisuje posle svakog "
            "završenog run-a.\n"
        )
    else:
        def get(run_id):
            return completed[completed["run_id"] == run_id].iloc[0]
        exp1 = get("lora_all_linear_exp1")
        lora_attn = get("lora_attention_only")
        dora_all = get("dora_all_linear")
        dora_attn = get("dora_attention_only")
        lines.append(f"- **Efekat uklanjanja MLP target modula (LoRA)**: all-linear F1={exp1['f1']:.4f} vs "
                    f"attention-only F1={lora_attn['f1']:.4f} (delta={lora_attn['f1']-exp1['f1']:+.4f}); "
                    f"trainable params {exp1['trainable_parameters']:,} -> {lora_attn['trainable_parameters']:,}.")
        lines.append(f"- **Efekat LoRA->DoRA (all-linear)**: F1={exp1['f1']:.4f} -> {dora_all['f1']:.4f} "
                    f"(delta={dora_all['f1']-exp1['f1']:+.4f}); peak VRAM {exp1['peak_vram_gb']} -> "
                    f"{dora_all['peak_vram_gb']:.2f} GB; duration {exp1['duration_seconds']:.0f}s -> "
                    f"{dora_all['duration_seconds']:.0f}s.")
        lines.append(f"- **Efekat LoRA->DoRA (attention-only)**: F1={lora_attn['f1']:.4f} -> {dora_attn['f1']:.4f} "
                    f"(delta={dora_attn['f1']-lora_attn['f1']:+.4f}); peak VRAM {lora_attn['peak_vram_gb']:.2f} -> "
                    f"{dora_attn['peak_vram_gb']:.2f} GB; duration {lora_attn['duration_seconds']:.0f}s -> "
                    f"{dora_attn['duration_seconds']:.0f}s.")
        best = completed.sort_values(by=["f1", "recall", "invalid_rate"], ascending=[False, False, True]).iloc[0]
        lines.append(f"\n**Najbolja validation konfiguracija (zaključani kriterijum: F1 -> recall -> "
                    f"invalid_rate)**: {best['run_id']} — precision={best['precision']:.4f}, "
                    f"recall={best['recall']:.4f}, F1={best['f1']:.4f}. Ovo je validation zaključak; NE "
                    "predstavlja test performanse.\n")

    lines.append("## Napomene\n")
    lines.append(
        f"- Test skup (`{_HELD_OUT_SPLIT_DISPLAY}`) NIJE učitan niti korišćen ni u jednom koraku ove "
        "ablacije -- ni za trening, ni za izbor konfiguracije, ni za metrike.\n"
        "- `lora_all_linear_exp1` (Experiment 1) je isključivo referenca pročitana sa diska -- adapter i "
        "rezultati nisu ponovo generisani ni menjani.\n"
        "- Rezultati u ovom izveštaju su validation-only i ne treba ih predstavljati kao test rezultate.\n"
    )
    (ABLATION_DIR / "REPORT.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def do_dry_run(configs):
    print("=" * 90)
    print("DRY RUN (LoRA/DoRA ablation) — nema treninga, nema GPU alokacije")
    print("=" * 90)

    print("\n[1] Provera putanja:")
    for label, p in [("model", MODEL_PATH), ("train.jsonl", TRAIN_PATH), ("validation.jsonl", VAL_PATH),
                     ("Experiment 1 dir", EXP1_DIR)]:
        print(f"    {label}: {p} -> {'POSTOJI' if p.exists() else 'NE POSTOJI'}")

    train_df, val_df = load_and_validate_dataset()

    print("\n[2] Self-audit: held-out split fajl se nigde ne pominje u ovom fajlu:")
    own_source = Path(__file__).read_text()
    held_out_filename = "test" + "." + "jsonl"
    assert held_out_filename not in own_source, \
        "Held-out split fajl je pronađen u izvornom kodu ovog skripta — PREKID."
    print(f"    [OK] Held-out split fajl se ne pojavljuje nigde u run_lora_dora_ablation_v2.py "
          f"(traženi obrazac: {held_out_filename!r}).")

    print("\n[3] Provera da nema logike za prekid zbog ukupnog vremena:")
    own_source_lower = own_source.lower()
    # Needles built via concatenation, exactly like the held-out-split check
    # above, so this self-audit's OWN diagnostic text (which necessarily
    # names what it's looking for) can never accidentally satisfy itself.
    needle_a = "time" + "_" + "budget"
    needle_b = "time" + "-" + "budget"
    assert needle_a not in own_source_lower and needle_b not in own_source_lower, \
        "Pronađena referenca na vremenski budžet u izvornom kodu — ovaj skript po specifikaciji ne treba da je ima."
    print("    [OK] Nema referenci na vremenski budžet u izvornom kodu -- skript se ne prekida zbog ukupnog vremena "
          "(nema odgovarajućeg argparse argumenta ni provere elapsed-time u main petlji).")

    print("\n[4] Provera da Experiment 1 NIJE u training queue-u ove ablacije:")
    exp1_ids = {c.get("run_id") for c in configs}
    assert "lora_all_linear_exp1" not in exp1_ids, "Experiment 1 je greškom u training queue-u!"
    print(f"    [OK] Experiment 1 (lora_all_linear_exp1) nije u build_grid() -- samo se čita kao referenca.")

    print(f"\n[5] Reference config (pročitan iz Experiment 1, ne nagađan; NIJE upisan na disk u dry-run-u):")
    reference = build_reference_config()
    for k in ["learning_rate", "rank", "alpha", "dropout", "seed", "max_epochs", "early_stopping_patience"]:
        print(f"    {k} = {reference[k]}")
    print(f"    confirmed_matches_ablation_constants: {reference['confirmed_matches_ablation_constants']}")
    assert all(reference["confirmed_matches_ablation_constants"].values()), \
        "Neki zaključani hiperparametar u skriptu se NE poklapa sa Experiment 1 -- PREKID."
    print("    [OK] Svi zaključani hiperparametri se poklapaju sa Experiment 1.")

    print(f"\n[6] {len(configs)} novih run-ova (redosled izvršavanja, zaključan):")
    assert len(configs) == 3, f"Očekivano tačno 3 nova run-a, dobijeno {len(configs)}"
    expected_order = ["lora_attention_only", "dora_all_linear", "dora_attention_only"]
    actual_order = [c["run_id"] for c in configs]
    assert actual_order == expected_order, f"Redosled run-ova je pogrešan: {actual_order} != {expected_order}"
    for i, c in enumerate(configs, start=1):
        run_dir = ABLATION_DIR / c["run_id"]
        status = read_status(run_dir).get("status", "pending") if run_dir.exists() else "pending"
        print(f"    {i}. {c['run_id']:22s} method={c['method']:5s} use_dora={c['use_dora']!s:6s} "
              f"target_scope={c['target_scope']:15s} target_modules={c['target_modules']} [status: {status}]")
    print("    [OK] Redosled i broj run-ova su tačno kao specificirano.")

    print("\n[7] Simulacija status/restart logike (bez treninga, bez GPU):")
    for c in configs:
        run_dir = ABLATION_DIR / c["run_id"]
        status_data = read_status(run_dir)
        status = status_data.get("status", "pending")
        attempt = status_data.get("attempt", 0)
        if status == "completed":
            action = "SKIP (već completed)"
        elif status == "failed_preflight":
            action = "SKIP (failed_preflight se ne ponavlja automatski)"
        elif status == "running":
            action = "ARCHIVE + RESTART (proces je bio prekinut u toku smoke testa ili treninga)"
        elif status == "failed" and attempt >= DEFAULT_MAX_ATTEMPTS:
            action = f"SKIP (već failed {attempt}x >= max-attempts)"
        else:
            action = f"RUN (pokušaj {attempt + 1}) -- smoke test, zatim pravi trening ako smoke test prođe"
        print(f"    {c['run_id']:22s} status={status:10s} attempt={attempt} -> {action}")

    print("\n[8] Provera da postojeći rezultati nisu dirani:")
    exp1_files_before = sorted(EXP1_DIR.glob("*")) if EXP1_DIR.exists() else []
    print(f"    Experiment 1 dir sadrži {len(exp1_files_before)} stavki -- ovaj dry-run ih NIJE izmenio "
          f"(samo pročitao run_config.json/run_summary.json/epoch_history.csv).")

    print(f"\n[OK] Dry run završen bez učitavanja modela na GPU, bez pokretanja treninga/smoke testa, "
          f"bez izmena Experiment 1/drugih rezultata, bez čitanja {_HELD_OUT_SPLIT_NAME}.")


# ---------------------------------------------------------------------------
# Main orchestration -- NO time budget: loops until every config is
# completed, failed_preflight, or exhausted its retry attempts.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                       help="Prikaži plan i provere bez učitavanja modela, treninga ili smoke testa.")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                       help=f"Maksimalan broj pokušaja PRAVOG treninga po konfiguraciji nakon uspešnog smoke "
                            f"testa, pre trajnog odustajanja (podrazumevano {DEFAULT_MAX_ATTEMPTS}). "
                            f"failed_preflight (neuspeo smoke test) se NIKAD ne ponavlja automatski, "
                            f"nezavisno od ove vrednosti.")
    args = parser.parse_args()

    configs = build_grid()

    if args.dry_run:
        do_dry_run(configs)
        return

    acquire_lock()
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    started_at = now_iso()
    print(f"[START] LoRA/DoRA ablation pokrenuta {started_at} (PID {os.getpid()}). Nema vremenskog budžeta.")

    write_reference_config()

    train_df, val_df = load_and_validate_dataset()
    tokenizer, end_of_turn_id, build_example, parse_label = build_tokenizer_and_helpers()
    train_examples, _ = build_split(train_df, "train", build_example)
    val_examples, _ = build_split(val_df, "validation", build_example)
    val_prefixes = build_val_prefixes(val_df, build_example)
    evaluate_checkpoint = make_evaluate_checkpoint(tokenizer, parse_label)
    SFTDataset = make_sft_dataset_cls()
    collate_fn = make_collate_fn(tokenizer)

    for cfg in configs:
        run_dir = ABLATION_DIR / cfg["run_id"]
        status_data = read_status(run_dir)
        status = status_data.get("status", "pending")
        attempt = status_data.get("attempt", 0)

        if status == "completed":
            print(f"[SKIP] {cfg['run_id']} već completed.")
            continue

        if status == "failed_preflight":
            print(f"[SKIP] {cfg['run_id']} je failed_preflight -- ne ponavlja se automatski "
                  f"(popraviti konfiguraciju/okruženje i obrisati status.json ručno da se ponovo proba).")
            continue

        if status == "running":
            print(f"[INTERRUPTED] {cfg['run_id']} je bio 'running' — prethodni proces je prekinut.")
            archive_stale_run(run_dir, attempt + 1)
            status = "interrupted"

        if status == "failed" and attempt >= args.max_attempts:
            print(f"[SKIP] {cfg['run_id']} već failed {attempt}x (max-attempts={args.max_attempts}).")
            continue

        next_attempt = attempt + 1
        print(f"\n[SMOKE TEST] {cfg['run_id']} (pokušaj {next_attempt}/{args.max_attempts})")
        write_status(run_dir, "running", attempt=next_attempt, phase="smoke_test")
        smoke_result = run_smoke_test(cfg, run_dir, train_examples, collate_fn,
                                      val_df, val_prefixes, evaluate_checkpoint)
        if not smoke_result["passed"]:
            print(f"    [FAILED_PREFLIGHT] {cfg['run_id']}: {smoke_result.get('error', '')[:500]}")
            write_status(run_dir, "failed_preflight", attempt=next_attempt, smoke_test_passed=False)
            summary_df = rebuild_ablation_summary(configs)
            write_root_report(configs, summary_df)
            write_ablation_state(configs, started_at)
            continue
        print(f"    [OK] Smoke test prošao (trainable={smoke_result['trainable_params']:,}, "
              f"eval_smoke_n_samples={smoke_result['eval_smoke_n_samples']}, "
              f"peak_vram={smoke_result['peak_vram_gb']:.2f} GB, "
              f"duration={smoke_result['duration_seconds']:.1f}s). Prelazim na pravi trening.")

        print(f"\n[RUN] {cfg['run_id']} (pokušaj {next_attempt}/{args.max_attempts})")
        result = run_one_config(cfg, run_dir, next_attempt, tokenizer, train_examples, val_examples,
                                val_df, val_prefixes, evaluate_checkpoint, SFTDataset, collate_fn)
        print(f"[{result['status'].upper()}] {cfg['run_id']} ({result['duration']:.0f}s)")

        summary_df = rebuild_ablation_summary(configs)
        write_root_report(configs, summary_df)
        write_ablation_state(configs, started_at)

    summary_df = rebuild_ablation_summary(configs)
    write_root_report(configs, summary_df)
    write_ablation_state(configs, started_at)

    remaining = [c for c in configs
                if read_status(ABLATION_DIR / c["run_id"]).get("status") not in ("completed", "failed_preflight")]
    print("\n" + "=" * 90)
    print(f"ABLATION {'NEPOTPUNA -- ima run-ova koji nisu ni completed ni failed_preflight' if remaining else 'KOMPLETNA'}")
    print("=" * 90)
    print("\nZbirna tabela (validation, sortirano po F1 opadajuće):")
    display_cols = ["run_id", "method", "target_scope", "use_dora", "status", "best_epoch",
                   "precision", "recall", "f1", "invalid_rate", "trainable_parameters", "peak_vram_gb"]
    sorted_df = summary_df.assign(_f1=summary_df["f1"].fillna(-1)).sort_values("_f1", ascending=False).drop(columns="_f1")
    print(sorted_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
