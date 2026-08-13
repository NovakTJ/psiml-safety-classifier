#!/usr/bin/env python
"""Standalone multi-seed stability check — Gemma 3 1B IT LoRA, v2 (no-refusal)
dataset. Follow-up to the Phase 1 (learning_rate x rank) and Phase 2
(dropout) sweeps.

Phase 2 found two close top all-linear-LoRA configs at dropout=0.0:

    Config A: learning_rate=3e-4, rank=8,  alpha=16, dropout=0.0  (F1=0.9684)
    Config B: learning_rate=1e-4, rank=16, alpha=32, dropout=0.0  (F1=0.9653)

Both were only ever trained at seed=42. This script checks whether either
config's edge over the other survives across seeds, by training each at
seed in {23, 41} (2 new runs each = 4 new training runs total) and comparing
against the existing seed=42 result — read read-only from Phase 2's results
directory, never retrained.

Grid (this script only):
    config_name in {config_a, config_b}
    seed         in {23, 41}
    data_seed = seed

That is 2 x 2 = 4 new training runs. Combined with the 2 read-only seed=42
references, the final comparison has 3 seeds x 2 configs = 6 rows.

Usage:
    ~/ccpp_env/bin/python sweep_lora_v2_multiseed.py --dry-run
    ~/ccpp_env/bin/python sweep_lora_v2_multiseed.py --time-budget-minutes 340

Safe to re-run: completed configs are skipped, failed configs are retried up
to --max-attempts times, and a config found "running" at startup (meaning
the previous process died mid-training) is restarted from scratch — this
script never resumes a Trainer from a mid-training checkpoint, since exact
optimizer/scheduler/callback-state reconstruction cannot be guaranteed
without that machinery, and the fallback of "restart the config from the
base model" was explicitly authorized instead (same policy as Phase 1/2).
"""

import argparse
import gc
import json
import os
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
# Offline mode defaults (the launcher also sets these; setting them here too
# means the script is self-sufficient if invoked directly, not only via the
# launcher).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed paths and constants
# ---------------------------------------------------------------------------
MODEL_PATH = Path("/data/models/gemma-3-1b-it")
V2_DATA_DIR = Path("/home/mls01/data/gemma_v2_no_refusal")
TRAIN_PATH = V2_DATA_DIR / "train.jsonl"
VAL_PATH = V2_DATA_DIR / "validation.jsonl"
# The held-out split is intentionally never named as a literal path constant
# here (see the self-audit in --dry-run, which greps this file's own source
# for the held-out filename and asserts it never appears).

SWEEP_DIR = Path("/home/mls01/scripts/model/results/gemma_lora_v2_multiseed")
LOCK_PATH = SWEEP_DIR / "multiseed.pid"

# Phase 2 results are a read-only input to this script (for the two seed=42
# reference rows). This script never creates, modifies, or deletes anything
# under PHASE2_SWEEP_DIR, and never touches Phase 1's directory either.
PHASE2_SWEEP_DIR = Path("/home/mls01/scripts/model/results/gemma_lora_v2_sweep_phase2")
PHASE1_SWEEP_DIR = Path("/home/mls01/scripts/model/results/gemma_lora_v2_sweep")
REFERENCE_SEED = 42

# The two configs under test — both already trained at seed=42 in Phase 2,
# both at dropout=0.0 (Phase 2's finding: less LoRA dropout won).
CONFIGS = [
    {"name": "config_a", "learning_rate": 3e-4, "rank": 8, "alpha": 16, "dropout": 0.0},
    {"name": "config_b", "learning_rate": 1e-4, "rank": 16, "alpha": 32, "dropout": 0.0},
]
NEW_SEEDS = [23, 41]
MAX_EPOCHS = 8
PATIENCE = 2
MIN_DELTA = 0.0
MAX_SEQ_LENGTH = 1024
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

DEFAULT_TIME_BUDGET_MINUTES = 340
DEFAULT_MAX_ATTEMPTS = 2
# Phase 2's 6 runs averaged ~32 min/run; these two configs (same lr/rank/
# alpha/dropout, only seed differs) should cost about the same per epoch.
# Used only until this sweep has at least one completed run to measure from.
FALLBACK_RUN_ESTIMATE_MINUTES = 32.0
RUN_ESTIMATE_SAFETY_FACTOR = 1.3

PROMPT_1 = """Classify the following sample as harmful or unharmful.

A sample is harmful if the user prompt is harmful or the assistant response contains harmful content. Otherwise, it is unharmful."""
PROMPT_1 += "\n\nReturn only: harmful or unharmful."

# Built via string concatenation, never spelled out directly, so the
# held-out split filename does not appear anywhere in this file's source as
# a literal — enforced by the self-audit in do_dry_run(). Used only for
# human-readable messages; the held-out split is never opened by this script.
_HELD_OUT_SPLIT_NAME = "test" + "." + "jsonl"
_HELD_OUT_SPLIT_DISPLAY = f"data/gemma_v2_no_refusal/{_HELD_OUT_SPLIT_NAME}"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------
def format_lr(lr):
    # 5e-05 -> 5e-5, 1e-04 -> 1e-4, 3e-04 -> 3e-4, etc.
    return f"{lr:.0e}".replace("e-0", "e-")


def config_id(name, lr, rank, alpha, dropout, seed):
    """This script's own naming, used for new run folders and for the
    'config_id' column of every row in multiseed_runs.csv (including the
    seed=42 reference rows, for consistency/joinability across seeds)."""
    return f"{name}_lr{format_lr(lr)}_r{rank}_alpha{alpha}_dropout{dropout}_seed{seed}"


def phase2_style_id(lr, rank, alpha, dropout, seed):
    """Phase 2's own naming (no config_a/config_b prefix) — used only to
    locate the seed=42 reference run directory inside PHASE2_SWEEP_DIR."""
    return f"lr{format_lr(lr)}_r{rank}_alpha{alpha}_dropout{dropout}_seed{seed}"


def build_grid():
    """Four new runs, in order: config_a seed 23, config_a seed 41,
    config_b seed 23, config_b seed 41."""
    configs = []
    for base in CONFIGS:
        for seed in NEW_SEEDS:
            configs.append({
                "name": base["name"],
                "config_id": config_id(base["name"], base["learning_rate"], base["rank"],
                                       base["alpha"], base["dropout"], seed),
                "learning_rate": base["learning_rate"], "rank": base["rank"],
                "alpha": base["alpha"], "dropout": base["dropout"],
                "seed": seed, "data_seed": seed,
            })
    return configs


# ---------------------------------------------------------------------------
# Concurrency lock. Only this script creates, checks, and removes
# LOCK_PATH ("multiseed.pid") — the launcher must never pre-write a PID
# into this file before starting python (same contract as Phase 2's fixed
# launcher; see launch_sweep_lora_v2_multiseed.sh).
# ---------------------------------------------------------------------------
def pid_is_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            existing_pid = int(LOCK_PATH.read_text().strip())
        except ValueError:
            existing_pid = None
        if existing_pid and pid_is_alive(existing_pid):
            print(f"[FATAL] Multi-seed sweep već aktivan (PID {existing_pid}, lock: {LOCK_PATH}). Izlazim bez pokretanja.")
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


def archive_stale_run(run_dir, attempt_number):
    """A config was found 'running' at startup: the previous process died
    mid-training. Archive its partial history (never silently overwrite) and
    remove partial checkpoints before a fresh attempt."""
    for fname in ["epoch_history.csv", "step_history.csv"]:
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
                 note="Prethodni proces prekinut usred treninga (detektovano pri sledećem pokretanju).")


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
    assert not (set(train_df["original_idx"]) & set(val_df["original_idx"]))
    print(f"[OK] train_df: {len(train_df)} redova/{train_df['original_idx'].nunique()} grupa | "
          f"val_df: {len(val_df)} redova/{val_df['original_idx'].nunique()} grupa | "
          f"held-out split nije učitan.")
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


def compute_metrics(y_true, y_pred, positive="harmful"):
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
            # do_sample=False -> no top_p/top_k passed (greedy decoding only).
            out = peft_model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                      do_sample=False, max_new_tokens=10)
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
# F1-based early-stopping callback — identical logic to Phase 1/2's
# ExplicitF1EarlyStoppingCallback: only a strictly greater F1 resets
# patience, a tied F1 does not. Writes status.json + history CSVs after
# every epoch (on_save fires once per epoch since save_strategy="epoch").
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
# Single-config training run — identical recipe to Phase 1/2's
# run_one_config; only the seed (and, across config_a/config_b, the
# lr/rank/alpha) varies across calls.
# ---------------------------------------------------------------------------
def run_one_config(cfg, run_dir, attempt, tokenizer, train_examples, val_examples,
                   val_df, val_prefixes, evaluate_checkpoint, SFTDataset, collate_fn):
    import shutil
    import torch
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model

    write_status(run_dir, "running", attempt=attempt, config_name=cfg["name"],
                learning_rate=cfg["learning_rate"], rank=cfg["rank"], alpha=cfg["alpha"],
                dropout=cfg["dropout"], seed=cfg["seed"])

    base_model = model = trainer = callback = None
    t0 = time.time()
    try:
        set_seed(cfg["seed"])

        # Every run starts fresh from the base model — never from a Phase 1,
        # Phase 2, or another multi-seed run's adapter/checkpoint.
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager",
        ).to("cuda")
        base_model.config.use_cache = False

        present = {n.split(".")[-1] for n, mod in base_model.named_modules()
                  if n.split(".")[-1] in TARGET_MODULES and isinstance(mod, torch.nn.Linear)}
        missing = set(TARGET_MODULES) - present
        if missing:
            raise RuntimeError(f"Target moduli nedostaju: {missing}")

        lora_config = LoraConfig(r=cfg["rank"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
                                 bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES)
        model = get_peft_model(base_model, lora_config)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_lora_trainable = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
        if non_lora_trainable:
            raise RuntimeError(f"Ne-LoRA parametri trainable: {non_lora_trainable[:5]}")

        base_model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

        # Brza probna forward/backward provera pre punog treninga.
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
            learning_rate=cfg["learning_rate"],
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            per_device_eval_batch_size=8,
            warmup_ratio=0.05,
            weight_decay=0.0,
            max_grad_norm=1.0,
            bf16=True, fp16=False,
            optim="adamw_torch",
            lr_scheduler_type="linear",
            seed=cfg["seed"], data_seed=cfg["data_seed"],
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

        train_output = trainer.train()
        duration = time.time() - t0

        import math
        bad = [r for r in trainer.state.log_history if "loss" in r and not math.isfinite(r["loss"])]
        if bad:
            raise RuntimeError(f"Nekonačan loss: {bad[:2]}")

        if not callback.epoch_history:
            raise RuntimeError("Nijedna epoha nije završena.")

        epoch_history_df = pd.DataFrame(callback.epoch_history)
        ranked = epoch_history_df.sort_values(by=["f1", "recall", "invalid_rate"],
                                              ascending=[False, False, True]).reset_index(drop=True)
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

        run_config = {
            "config_id": cfg["config_id"], "config_name": cfg["name"], "model_path": str(MODEL_PATH),
            "train_path": str(TRAIN_PATH), "validation_path": str(VAL_PATH),
            "learning_rate": cfg["learning_rate"], "rank": cfg["rank"], "alpha": cfg["alpha"],
            "dropout": cfg["dropout"], "seed": cfg["seed"], "data_seed": cfg["data_seed"],
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA,
            "max_seq_length": MAX_SEQ_LENGTH, "target_modules": TARGET_MODULES,
            "prompt": PROMPT_1, "generation": {"do_sample": False, "max_new_tokens": 10},
            "total_params": int(total_params), "trainable_params": int(trainable_params),
            "phase": "multiseed",
        }
        (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False))

        run_summary = {
            "config_id": cfg["config_id"], "config_name": cfg["name"], "best_epoch": best_epoch,
            "best_metrics": {"precision": float(best_row["precision"]), "recall": float(best_row["recall"]),
                            "f1": float(best_row["f1"]), "invalid_count": int(best_row["invalid_count"]),
                            "invalid_rate": float(best_row["invalid_rate"])},
            "last_completed_epoch": last_completed_epoch, "stop_reason": callback.stop_reason,
            "early_stopping_triggered": bool(callback.early_stopped),
            "total_training_duration_seconds": duration,
            "best_adapter_path": str(best_adapter_path),
        }
        (run_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False))
        epoch_history_df.to_csv(run_dir / "epoch_history.csv", index=False)

        # Best adapter + istorija + rezime su trajno sačuvani — teški
        # per-epoch checkpointovi (sa optimizer.pt) više nisu potrebni.
        # Uklanjaju se TEK nakon što je gore potvrđeno da best_adapter/
        # sadrži oba obavezna fajla.
        for ckpt in checkpoint_dirs:
            shutil.rmtree(ckpt, ignore_errors=True)

        write_status(run_dir, "completed", attempt=attempt, best_epoch=best_epoch,
                    f1=float(best_row["f1"]))
        return {"status": "completed", "run_summary": run_summary, "duration": duration}

    except Exception:
        tb = traceback.format_exc()
        (run_dir / "error.txt").write_text(tb)
        write_status(run_dir, "failed", attempt=attempt)
        print(f"    [FAILED] {cfg['config_id']}:\n{tb}")
        return {"status": "failed", "duration": time.time() - t0}

    finally:
        del trainer, model, base_model, callback
        gc.collect()
        import torch as _torch
        _torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Sweep-wide tables and REPORT.md
# ---------------------------------------------------------------------------
RUNS_COLUMNS = ["config_name", "config_id", "learning_rate", "rank", "alpha", "dropout", "seed",
               "status", "best_epoch", "precision", "recall", "f1", "invalid_count", "invalid_rate",
               "epochs_completed", "duration_seconds", "stop_reason", "run_path", "source"]

SUMMARY_COLUMNS = ["config_name", "learning_rate", "rank", "alpha", "dropout", "n_seeds", "seeds",
                   "mean_precision", "std_precision", "mean_recall", "std_recall",
                   "mean_f1", "std_f1", "min_f1", "max_f1", "mean_invalid_rate", "n_runs_zero_invalid"]


def phase2_reference_row(base):
    """Read-only: pulls the already-completed Phase 2 result for this config
    at seed=42. Never writes into PHASE2_SWEEP_DIR. Raises loudly (does not
    guess) if the values on disk don't match what was declared in this
    script's own docstring/spec, so a stale/mismatched Phase 2 result can
    never silently poison the multi-seed comparison."""
    run_dir = PHASE2_SWEEP_DIR / phase2_style_id(base["learning_rate"], base["rank"],
                                                 base["alpha"], base["dropout"], REFERENCE_SEED)
    summary_path = run_dir / "run_summary.json"
    history_path = run_dir / "epoch_history.csv"
    if not summary_path.exists() or not history_path.exists():
        raise FileNotFoundError(
            f"Phase 2 seed=42 referenca nedostaje ili je nekompletna: {run_dir} "
            f"(očekivano run_summary.json + epoch_history.csv)"
        )
    summary = json.loads(summary_path.read_text())
    history = pd.read_csv(history_path)
    best_epoch = summary["best_epoch"]
    best_row = history[history["epoch"] == best_epoch].iloc[0]
    return {
        "config_name": base["name"],
        "config_id": config_id(base["name"], base["learning_rate"], base["rank"],
                               base["alpha"], base["dropout"], REFERENCE_SEED),
        "learning_rate": base["learning_rate"], "rank": base["rank"], "alpha": base["alpha"],
        "dropout": base["dropout"], "seed": REFERENCE_SEED,
        "status": "completed", "best_epoch": int(best_epoch),
        "precision": float(best_row["precision"]), "recall": float(best_row["recall"]),
        "f1": float(best_row["f1"]), "invalid_count": int(best_row["invalid_count"]),
        "invalid_rate": float(best_row["invalid_rate"]),
        "epochs_completed": summary["last_completed_epoch"],
        "duration_seconds": summary["total_training_duration_seconds"],
        "stop_reason": summary["stop_reason"], "run_path": str(run_dir),
        "source": "phase2_reference",
    }


def rebuild_multiseed_runs(configs):
    """multiseed_runs.csv: the four new runs (whatever their current status)
    plus the two Phase 2 seed=42 references, six rows total."""
    rows = []
    for cfg in configs:
        run_dir = SWEEP_DIR / cfg["config_id"]
        status = read_status(run_dir).get("status", "pending")
        row = {"config_name": cfg["name"], "config_id": cfg["config_id"],
               "learning_rate": cfg["learning_rate"], "rank": cfg["rank"], "alpha": cfg["alpha"],
               "dropout": cfg["dropout"], "seed": cfg["seed"], "status": status,
               "run_path": str(run_dir), "source": "multiseed"}
        for k in ["best_epoch", "precision", "recall", "f1", "invalid_count", "invalid_rate",
                 "epochs_completed", "duration_seconds", "stop_reason"]:
            row.setdefault(k, None)
        summary_path = run_dir / "run_summary.json"
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            row.update({
                "best_epoch": s["best_epoch"], "precision": s["best_metrics"]["precision"],
                "recall": s["best_metrics"]["recall"], "f1": s["best_metrics"]["f1"],
                "invalid_count": s["best_metrics"]["invalid_count"],
                "invalid_rate": s["best_metrics"]["invalid_rate"],
                "epochs_completed": s["last_completed_epoch"],
                "duration_seconds": s["total_training_duration_seconds"],
                "stop_reason": s["stop_reason"],
            })
        rows.append(row)

    for base in CONFIGS:
        rows.append(phase2_reference_row(base))

    df = pd.DataFrame(rows)[RUNS_COLUMNS]
    sort_key_name = df["config_name"]
    sort_key_seed = df["seed"]
    df = df.assign(_name=sort_key_name, _seed=sort_key_seed) \
        .sort_values(by=["_name", "_seed"], ascending=[True, True]) \
        .drop(columns=["_name", "_seed"]).reset_index(drop=True)
    df.to_csv(SWEEP_DIR / "multiseed_runs.csv", index=False)
    return df


def rebuild_multiseed_summary(runs_df):
    """multiseed_summary.csv: one aggregated row per config, computed over
    whichever of the 3 seeds are currently 'completed' (so this is
    meaningful even mid-sweep, though the final n_seeds=3 comparison only
    happens once all 4 new runs are done)."""
    rows = []
    for base in CONFIGS:
        sub = runs_df[(runs_df["config_name"] == base["name"]) & (runs_df["status"] == "completed")]
        if len(sub) == 0:
            rows.append({
                "config_name": base["name"], "learning_rate": base["learning_rate"],
                "rank": base["rank"], "alpha": base["alpha"], "dropout": base["dropout"],
                "n_seeds": 0, "seeds": "", "mean_precision": None, "std_precision": None,
                "mean_recall": None, "std_recall": None, "mean_f1": None, "std_f1": None,
                "min_f1": None, "max_f1": None, "mean_invalid_rate": None, "n_runs_zero_invalid": None,
            })
            continue
        f1 = sub["f1"].astype(float)
        rows.append({
            "config_name": base["name"], "learning_rate": base["learning_rate"],
            "rank": base["rank"], "alpha": base["alpha"], "dropout": base["dropout"],
            "n_seeds": len(sub), "seeds": ",".join(str(s) for s in sorted(sub["seed"].tolist())),
            "mean_precision": float(sub["precision"].astype(float).mean()),
            "std_precision": float(sub["precision"].astype(float).std(ddof=1)) if len(sub) > 1 else 0.0,
            "mean_recall": float(sub["recall"].astype(float).mean()),
            "std_recall": float(sub["recall"].astype(float).std(ddof=1)) if len(sub) > 1 else 0.0,
            "mean_f1": float(f1.mean()),
            "std_f1": float(f1.std(ddof=1)) if len(sub) > 1 else 0.0,
            "min_f1": float(f1.min()), "max_f1": float(f1.max()),
            "mean_invalid_rate": float(sub["invalid_rate"].astype(float).mean()),
            "n_runs_zero_invalid": int((sub["invalid_count"].astype(float) == 0).sum()),
        })
    df = pd.DataFrame(rows)[SUMMARY_COLUMNS]
    df.to_csv(SWEEP_DIR / "multiseed_summary.csv", index=False)
    return df


def pick_winner(summary_df):
    """Selection order (never by single best seed):
    1. highest mean F1
    2. then highest mean recall
    3. then lowest F1 std
    4. then lowest mean invalid rate
    Returns None if no config has n_seeds > 0 yet."""
    complete = summary_df[summary_df["n_seeds"].fillna(0) > 0].copy()
    if len(complete) == 0:
        return None
    complete = complete.sort_values(
        by=["mean_f1", "mean_recall", "std_f1", "mean_invalid_rate"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return complete.iloc[0]


def write_multiseed_state(configs, time_budget_minutes, incomplete, started_at):
    state = {
        "phase": "multiseed", "time_budget_minutes": time_budget_minutes,
        "started_at": started_at, "last_updated_at": now_iso(),
        "incomplete": incomplete,
        "configs": {c["config_id"]: read_status(SWEEP_DIR / c["config_id"]).get("status", "pending")
                   for c in configs},
        "phase2_references": {
            config_id(base["name"], base["learning_rate"], base["rank"], base["alpha"],
                     base["dropout"], REFERENCE_SEED):
                str(PHASE2_SWEEP_DIR / phase2_style_id(base["learning_rate"], base["rank"],
                                                       base["alpha"], base["dropout"], REFERENCE_SEED))
            for base in CONFIGS
        },
    }
    (SWEEP_DIR / "multiseed_state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))


def write_report(runs_df, summary_df):
    lines = []
    lines.append("# Multi-seed stability check — Gemma 3 1B IT all-linear LoRA, v2 dataset (bez refusal-a)\n")
    lines.append("## Cilj\n")
    lines.append(
        "Phase 2 (`scripts/model/sweep_lora_v2_phase2.py`) je pronašao dve bliske najbolje all-linear "
        "LoRA konfiguracije, obe na `dropout=0.0`, ali obe trenirane samo na `seed=42`. Ovaj eksperiment "
        "proverava da li ta razlika (F1 0.9684 vs 0.9653) preživljava kroz više seedova, ili je šum jednog "
        "seed-a. Svaka konfiguracija je ponovo istrenirana na `seed in {23, 41}` (4 nova treninga); "
        "postojeći `seed=42` rezultat se ne ponavlja, samo se čita iz `scripts/model/results/"
        "gemma_lora_v2_sweep_phase2/` (Phase 1 i Phase 2 folderi ostaju netaknuti).\n"
    )

    lines.append("## Konfiguracije\n")
    lines.append(
        "- **Config A**: learning_rate=3e-4, rank=8, alpha=16, dropout=0.0 (Phase 2 seed=42 pobednik, F1=0.9684)\n"
        "- **Config B**: learning_rate=1e-4, rank=16, alpha=32, dropout=0.0 (Phase 2 seed=42 drugoplasirani, F1=0.9653)\n"
    )

    lines.append("## Svih šest pojedinačnih rezultata\n")
    cols = ["config_name", "config_id", "seed", "source", "status", "best_epoch",
           "precision", "recall", "f1", "invalid_rate", "epochs_completed", "duration_seconds"]
    lines.append("```")
    lines.append(runs_df[cols].to_string(index=False))
    lines.append("```\n")

    lines.append("## Agregirani rezultati (mean ± std preko seed-ova)\n")
    if summary_df["n_seeds"].fillna(0).max() == 0:
        lines.append("Nijedan run još nije završen — agregacija će se automatski popuniti nakon prvog završenog runa.\n")
    else:
        for _, row in summary_df.iterrows():
            if not row["n_seeds"]:
                lines.append(f"- **{row['config_name']}**: još nema završenih runova.\n")
                continue
            lines.append(
                f"- **{row['config_name']}** (n_seeds={int(row['n_seeds'])}, seeds={row['seeds']}): "
                f"F1 = {row['mean_f1']:.4f} ± {row['std_f1']:.4f} "
                f"(min={row['min_f1']:.4f}, max={row['max_f1']:.4f}), "
                f"precision = {row['mean_precision']:.4f} ± {row['std_precision']:.4f}, "
                f"recall = {row['mean_recall']:.4f} ± {row['std_recall']:.4f}, "
                f"mean invalid_rate = {row['mean_invalid_rate']:.4f}, "
                f"{int(row['n_runs_zero_invalid'])}/{int(row['n_seeds'])} runova bez invalid outputa.\n"
            )

    lines.append("## Izabrani stabilni pobednik\n")
    winner = pick_winner(summary_df)
    if winner is None:
        lines.append(
            "Nijedan config još nema završen nijedan seed — pobednik će se automatski odrediti "
            "čim bar jedan run bude završen za oba configa (puno poređenje traži sva 3 seed-a).\n"
        )
    else:
        lines.append(
            f"**{winner['config_name']}** — mean F1 = {winner['mean_f1']:.4f} ± {winner['std_f1']:.4f} "
            f"preko {int(winner['n_seeds'])} seed-a ({winner['seeds']}).\n"
        )
        lines.append(
            "**Kriterijum izbora (u ovom redosledu, pobednik nikad nije biran po najboljem "
            "pojedinačnom seedu)**: 1) najveći mean F1, 2) zatim najveći mean recall, "
            "3) zatim manji F1 std, 4) zatim manji mean invalid rate.\n"
        )
        others = summary_df[(summary_df["config_name"] != winner["config_name"]) & (summary_df["n_seeds"] > 0)]
        if len(others):
            other = others.iloc[0]
            gap = winner["mean_f1"] - other["mean_f1"]
            lines.append(
                f"Razlika u mean F1 naspram `{other['config_name']}`: {gap:+.4f} "
                f"({winner['mean_f1']:.4f} vs {other['mean_f1']:.4f}). "
                + ("Ovo je vrlo mala razlika — ne treba je preuveličavati kao jasnu pobedu.\n"
                   if abs(gap) < 0.005 else "\n")
            )
        lines.append(
            f"**Poređenje sa seed=42 zaključkom**: na seed=42 samom, Config A (F1=0.9684) je vodio "
            f"Config B (F1=0.9653) za +0.0031. "
            + (f"Multi-seed rezultat se slaže sa tim zaključkom (isti pobednik).\n"
               if winner["config_name"] == "config_a" else
               f"Multi-seed rezultat NE potvrđuje taj zaključak — pobednik preko seed-ova je drugačiji "
               f"config nego na samom seed=42, što znači da je seed=42 razlika verovatno bila šum.\n")
        )

    lines.append("## Napomene\n")
    lines.append(
        f"- Test skup (`{_HELD_OUT_SPLIT_DISPLAY}`) NIJE korišćen ni za trening ni za evaluaciju bilo kog runa.\n"
        "- Attention-only i DoRA eksperimenti NISU pokrenuti u ovoj fazi.\n"
        "- **Predlog za sledeći eksperiment**: attention-only ablation (LoRA samo na q_proj/k_proj/v_proj/"
        "o_proj, bez gate/up/down_proj) sa zaključanim pobedničkim configom iznad, da se proveri koliko "
        "MLP projekcije doprinose F1-u naspram broja trenable parametara.\n"
    )

    (SWEEP_DIR / "REPORT.md").write_text("\n".join(lines))


def estimate_next_run_minutes():
    runs_path = SWEEP_DIR / "multiseed_runs.csv"
    if runs_path.exists():
        df = pd.read_csv(runs_path)
        completed = df[(df["status"] == "completed") & (df["source"] == "multiseed")]
        if len(completed):
            avg_seconds = completed["duration_seconds"].mean()
            return (avg_seconds / 60.0) * RUN_ESTIMATE_SAFETY_FACTOR
    return FALLBACK_RUN_ESTIMATE_MINUTES


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def do_dry_run(configs):
    print("=" * 90)
    print("DRY RUN (multi-seed) — nema treninga, nema GPU alokacije")
    print("=" * 90)

    print("\n[1] Provera putanja:")
    for label, p in [("model", MODEL_PATH), ("train.jsonl", TRAIN_PATH), ("validation.jsonl", VAL_PATH)]:
        print(f"    {label}: {p} -> {'POSTOJI' if p.exists() else 'NE POSTOJI'}")

    train_df, val_df = load_and_validate_dataset()

    print("\n[2] Self-audit: held-out split fajl se nigde ne pominje u ovom fajlu:")
    own_source = Path(__file__).read_text()
    held_out_filename = "test" + "." + "jsonl"
    assert held_out_filename not in own_source, \
        f"Held-out split fajl je pronađen u izvornom kodu ovog sweep-a — PREKID."
    print(f"    [OK] Held-out split fajl se ne pojavljuje nigde u sweep_lora_v2_multiseed.py "
          f"(traženi obrazac: {held_out_filename!r}).")

    print("\n[3] Provera da Phase 1 i Phase 2 ostaju netaknuti (samo čitanje, ne pisanje):")
    for label, p in [("Phase 1 sweep dir", PHASE1_SWEEP_DIR), ("Phase 2 sweep dir", PHASE2_SWEEP_DIR)]:
        print(f"    {label}: {p} -> {'POSTOJI' if p.exists() else 'NE POSTOJI (!)'}")

    assert len(configs) == 4, f"Očekivano 4 nova multi-seed runa, dobijeno {len(configs)}"
    assert set(c["seed"] for c in configs) == {23, 41}, \
        f"Očekivani seedovi {{23, 41}}, dobijeno {set(c['seed'] for c in configs)}"
    for c in configs:
        assert c["seed"] == c["data_seed"], f"seed != data_seed za {c['config_id']}"
        assert c["dropout"] == 0.0, f"Neočekivan dropout != 0.0 za {c['config_id']}"
    print(f"    [OK] Tačno 4 nova runa, seedovi = {{23, 41}}, seed == data_seed za sve, dropout=0.0 za sve.")

    print(f"\n[4] {len(configs)} nova multi-seed runa (redosled izvršavanja):")
    for i, c in enumerate(configs, start=1):
        run_dir = SWEEP_DIR / c["config_id"]
        status = read_status(run_dir).get("status", "pending") if run_dir.exists() else "pending"
        print(f"    {i}. {c['config_id']:55s} name={c['name']:10s} lr={c['learning_rate']:<8g} "
              f"r={c['rank']:<3d} alpha={c['alpha']:<3d} seed={c['seed']:<3d} data_seed={c['data_seed']:<3d} "
              f"[status: {status}]")

    print(f"\n[5] Dve seed=42 reference (Phase 2, NEĆE se trenirati, samo se čitaju):")
    all_refs_found = True
    for base in CONFIGS:
        ref_id = phase2_style_id(base["learning_rate"], base["rank"], base["alpha"], base["dropout"], REFERENCE_SEED)
        ref_dir = PHASE2_SWEEP_DIR / ref_id
        found = ref_dir.exists() and (ref_dir / "run_summary.json").exists()
        all_refs_found = all_refs_found and found
        print(f"    - {base['name']} ({ref_id}) -> {ref_dir} [{'OK' if found else 'NEDOSTAJE'}]")
    if all_refs_found:
        try:
            refs = [phase2_reference_row(b) for b in CONFIGS]
            print("\n    Učitane referentne vrednosti (samo čitanje, ništa se ne piše u Phase 2 folder):")
            for r in refs:
                print(f"      {r['config_name']} seed=42: P={r['precision']:.6f} R={r['recall']:.6f} "
                      f"F1={r['f1']:.6f} best_epoch={r['best_epoch']} invalid_rate={r['invalid_rate']:.4f}")
            # Cross-check against the exact values the user specified in the spec.
            expected = {
                "config_a": {"best_epoch": 6, "precision": 0.980769, "recall": 0.956250, "f1": 0.968354},
                "config_b": {"best_epoch": 7, "precision": 0.974522, "recall": 0.956250, "f1": 0.965300},
            }
            for r in refs:
                exp = expected[r["config_name"]]
                ok = (r["best_epoch"] == exp["best_epoch"]
                      and abs(r["precision"] - exp["precision"]) < 1e-4
                      and abs(r["recall"] - exp["recall"]) < 1e-4
                      and abs(r["f1"] - exp["f1"]) < 1e-4)
                if not ok:
                    raise AssertionError(
                        f"NESLAGANJE za {r['config_name']} seed=42: fajl daje {r}, "
                        f"očekivano (iz spec-a) {exp} — PREKID, ne nagađam referencu."
                    )
            print("    [OK] Sve pročitane seed=42 vrednosti se poklapaju sa spec-om (na 1e-4).")
        except Exception as e:
            print(f"    [FATAL] {e}")
            raise
    else:
        print("    [WARN] Bar jedna Phase 2 referenca nedostaje — proveri da Phase 2 sweep nije premešten/obrisan.")

    print(f"\n[6] Procenjeno vreme po run-u: {estimate_next_run_minutes():.1f} min "
          f"(fallback konstanta dok ovaj sweep nema nijedan sopstveni završen run).")

    print("\n[OK] Dry run završen bez učitavanja modela na GPU, bez pokretanja treninga, "
          f"bez izmena u Phase 1/Phase 2 folderima, bez čitanja {_HELD_OUT_SPLIT_NAME}.")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                       help="Prikaži plan i provere bez učitavanja modela ili treninga.")
    parser.add_argument("--time-budget-minutes", type=float, default=DEFAULT_TIME_BUDGET_MINUTES,
                       help=f"Vremenski budžet za ovo pokretanje (podrazumevano {DEFAULT_TIME_BUDGET_MINUTES}).")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                       help=f"Maksimalan broj pokušaja po konfiguraciji pre trajnog odustajanja "
                            f"(podrazumevano {DEFAULT_MAX_ATTEMPTS}).")
    args = parser.parse_args()

    configs = build_grid()

    if args.dry_run:
        do_dry_run(configs)
        return

    acquire_lock()
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    # Fail loudly and immediately (before any GPU work) if the seed=42
    # references don't match the spec — never silently train against a
    # mismatched baseline.
    expected = {
        "config_a": {"best_epoch": 6, "precision": 0.980769, "recall": 0.956250, "f1": 0.968354},
        "config_b": {"best_epoch": 7, "precision": 0.974522, "recall": 0.956250, "f1": 0.965300},
    }
    for base in CONFIGS:
        r = phase2_reference_row(base)
        exp = expected[r["config_name"]]
        ok = (r["best_epoch"] == exp["best_epoch"]
              and abs(r["precision"] - exp["precision"]) < 1e-4
              and abs(r["recall"] - exp["recall"]) < 1e-4
              and abs(r["f1"] - exp["f1"]) < 1e-4)
        if not ok:
            print(f"[FATAL] Seed=42 referenca za {r['config_name']} se ne poklapa sa spec-om: {r} vs {exp}")
            sys.exit(1)
    print("[OK] Seed=42 reference potvrđene protiv spec-a pre pokretanja treninga.")

    invocation_start = time.time()
    started_at = now_iso()
    print(f"[START] Multi-seed sweep pokrenut {started_at} (PID {os.getpid()}), budžet {args.time_budget_minutes} min.")

    train_df, val_df = load_and_validate_dataset()
    tokenizer, end_of_turn_id, build_example, parse_label = build_tokenizer_and_helpers()
    train_examples, _ = build_split(train_df, "train", build_example)
    val_examples, _ = build_split(val_df, "validation", build_example)
    val_prefixes = build_val_prefixes(val_df, build_example)
    evaluate_checkpoint = make_evaluate_checkpoint(tokenizer, parse_label)
    SFTDataset = make_sft_dataset_cls()
    collate_fn = make_collate_fn(tokenizer)

    incomplete = False

    for cfg in configs:
        run_dir = SWEEP_DIR / cfg["config_id"]
        status_data = read_status(run_dir)
        status = status_data.get("status", "pending")
        attempt = status_data.get("attempt", 0)

        if status == "completed":
            print(f"[SKIP] {cfg['config_id']} već completed.")
            continue

        if status == "running":
            print(f"[INTERRUPTED] {cfg['config_id']} je bio 'running' — prethodni proces je prekinut.")
            archive_stale_run(run_dir, attempt + 1)
            status = "interrupted"

        if status == "failed" and attempt >= args.max_attempts:
            print(f"[SKIP] {cfg['config_id']} već failed {attempt}x (max-attempts={args.max_attempts}).")
            continue

        remaining_minutes = args.time_budget_minutes - (time.time() - invocation_start) / 60.0
        estimate = estimate_next_run_minutes()
        if remaining_minutes < estimate:
            print(f"[TIME BUDGET] Preostalo {remaining_minutes:.1f} min < procenjeno {estimate:.1f} min "
                  f"za sledeći run. Zaustavljam sweep pre {cfg['config_id']} — aktivni run (ako postoji) "
                  f"je već uredno završen. Ostatak nastavlja sledeće pokretanje.")
            incomplete = True
            break

        next_attempt = attempt + 1
        print(f"\n[RUN] {cfg['config_id']} (pokušaj {next_attempt}/{args.max_attempts})")
        result = run_one_config(cfg, run_dir, next_attempt, tokenizer, train_examples, val_examples,
                                val_df, val_prefixes, evaluate_checkpoint, SFTDataset, collate_fn)
        print(f"[{result['status'].upper()}] {cfg['config_id']} ({result['duration']:.0f}s)")

        runs_df = rebuild_multiseed_runs(configs)
        summary_df = rebuild_multiseed_summary(runs_df)
        write_report(runs_df, summary_df)
        write_multiseed_state(configs, args.time_budget_minutes, incomplete=False, started_at=started_at)
    else:
        remaining = [c for c in configs
                    if read_status(SWEEP_DIR / c["config_id"]).get("status") != "completed"]
        incomplete = len(remaining) > 0

    runs_df = rebuild_multiseed_runs(configs)
    summary_df = rebuild_multiseed_summary(runs_df)
    write_report(runs_df, summary_df)
    write_multiseed_state(configs, args.time_budget_minutes, incomplete=incomplete, started_at=started_at)

    print("\n" + "=" * 90)
    print(f"MULTI-SEED SWEEP {'NEPOTPUN — nastaviće se pri sledećem pokretanju' if incomplete else 'KOMPLETAN'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
