#!/usr/bin/env python
"""Standalone Phase 2 LoRA sweep — Gemma 3 1B IT, v2 (no-refusal) dataset.

Dropout follow-up to the completed Phase 1 sweep
(scripts/model/sweep_lora_v2.py -> scripts/model/results/gemma_lora_v2_sweep/).
Phase 1 swept learning_rate x rank at a fixed dropout=0.05 and found three
close top configs. Phase 2 takes exactly those three (learning_rate, rank,
alpha) combinations and checks whether dropout tuning improves them further.

Grid (Phase 2 only):
    (learning_rate, rank, alpha) in {
        (1e-4, 16, 32),
        (2e-4,  4,  8),
        (3e-4,  8, 16),
    }
    dropout in {0.0, 0.1}

That is 3 x 2 = 6 new training runs. The three dropout=0.05 cells for these
same (lr, rank, alpha) triples are NEVER retrained here — they already exist
as completed Phase 1 runs and are folded in as source=phase1_reference,
read-only, from scripts/model/results/gemma_lora_v2_sweep/. Phase 1's script,
results directory, and the separate Experiment 1 results directory are never
written to by this file.

Usage:
    ~/ccpp_env/bin/python sweep_lora_v2_phase2.py --dry-run
    ~/ccpp_env/bin/python sweep_lora_v2_phase2.py --time-budget-minutes 340

Safe to re-run: completed configs are skipped, failed configs are retried up
to --max-attempts times, and a config found "running" at startup (meaning
the previous process died mid-training) is restarted from scratch — this
script never resumes a Trainer from a mid-training checkpoint, since exact
optimizer/scheduler/callback-state reconstruction cannot be guaranteed
without that machinery, and the fallback of "restart the config from the
base model" was explicitly authorized instead (same policy as Phase 1).
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

SWEEP_DIR = Path("/home/mls01/scripts/model/results/gemma_lora_v2_sweep_phase2")
LOCK_PATH = SWEEP_DIR / "phase2.pid"

# Phase 1 results are read-only inputs to Phase 2 (for the three reference
# rows). This script never creates, modifies, or deletes anything under
# PHASE1_SWEEP_DIR or PHASE1_EXP1_DIR.
PHASE1_SWEEP_DIR = Path("/home/mls01/scripts/model/results/gemma_lora_v2_sweep")
PHASE1_EXP1_DIR = Path(
    "/home/mls01/scripts/model/results/gemma_lora_v2_exp1_r8_lr2e4_seed42_max8_es2"
)
PHASE1_DROPOUT = 0.05

# The three (learning_rate, rank, alpha) triples carried over from Phase 1 —
# these are Phase 1's top three F1 configs (all at dropout=0.05). Phase 2
# explores dropout for exactly these three, nothing else.
BASE_CONFIGS = [
    {"learning_rate": 1e-4, "rank": 16, "alpha": 32},
    {"learning_rate": 2e-4, "rank": 4, "alpha": 8},
    {"learning_rate": 3e-4, "rank": 8, "alpha": 16},
]
DROPOUTS = [0.0, 0.1]
SEED = 42
MAX_EPOCHS = 8
PATIENCE = 2
MIN_DELTA = 0.0
MAX_SEQ_LENGTH = 1024
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

DEFAULT_TIME_BUDGET_MINUTES = 340
DEFAULT_MAX_ATTEMPTS = 2
# Phase 1's 11 trained runs averaged ~27 min/run (299.9 min / 11); dropout
# does not change per-epoch compute cost, so the same ballpark applies here.
# Used only until this sweep has at least one completed run to measure from.
FALLBACK_RUN_ESTIMATE_MINUTES = 35.0
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
# Grid construction
# ---------------------------------------------------------------------------
def format_lr(lr):
    # 5e-05 -> 5e-5, 1e-04 -> 1e-4, etc. (matches the Phase 1 config_id style)
    return f"{lr:.0e}".replace("e-0", "e-")


def config_id(lr, rank, alpha, dropout, seed):
    return f"lr{format_lr(lr)}_r{rank}_alpha{alpha}_dropout{dropout}_seed{seed}"


def build_grid():
    """Six new Phase 2 configs, in run order: grouped by base (lr, rank,
    alpha) in the order listed in BASE_CONFIGS, dropout 0.0 before 0.1."""
    configs = []
    for base in BASE_CONFIGS:
        for dropout in DROPOUTS:
            configs.append({
                "config_id": config_id(base["learning_rate"], base["rank"], base["alpha"], dropout, SEED),
                "learning_rate": base["learning_rate"], "rank": base["rank"], "alpha": base["alpha"],
                "dropout": dropout, "seed": SEED,
            })
    return configs


def phase1_reference_config_id(base):
    return config_id(base["learning_rate"], base["rank"], base["alpha"], PHASE1_DROPOUT, SEED)


# ---------------------------------------------------------------------------
# Concurrency lock. Only this script creates, checks, and removes
# LOCK_PATH ("phase2.pid") — the launcher must never pre-write a PID into
# this file before starting python (see launch_sweep_lora_v2_phase2.sh for
# the corresponding launcher-side contract). This mirrors Phase 1's
# acquire_lock(), pointed at phase2.pid instead of sweep.pid.
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
            print(f"[FATAL] Phase 2 sweep već aktivan (PID {existing_pid}, lock: {LOCK_PATH}). Izlazim bez pokretanja.")
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
# F1-based early-stopping callback — identical logic to Phase 1's
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
# Single-config training run — identical recipe to Phase 1's run_one_config,
# only the LoRA dropout varies across calls (via cfg["dropout"]).
# ---------------------------------------------------------------------------
def run_one_config(cfg, run_dir, attempt, tokenizer, train_examples, val_examples,
                   val_df, val_prefixes, evaluate_checkpoint, SFTDataset, collate_fn):
    import shutil
    import torch
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model

    write_status(run_dir, "running", attempt=attempt, learning_rate=cfg["learning_rate"],
                rank=cfg["rank"], alpha=cfg["alpha"], dropout=cfg["dropout"])

    base_model = model = trainer = callback = None
    t0 = time.time()
    try:
        set_seed(cfg["seed"])

        # Every run starts fresh from the base model — never from a Phase 1
        # or another Phase 2 run's adapter/checkpoint.
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
            seed=cfg["seed"], data_seed=cfg["seed"],
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
            "config_id": cfg["config_id"], "model_path": str(MODEL_PATH),
            "train_path": str(TRAIN_PATH), "validation_path": str(VAL_PATH),
            "learning_rate": cfg["learning_rate"], "rank": cfg["rank"], "alpha": cfg["alpha"],
            "dropout": cfg["dropout"], "seed": cfg["seed"], "data_seed": cfg["seed"],
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA,
            "max_seq_length": MAX_SEQ_LENGTH, "target_modules": TARGET_MODULES,
            "prompt": PROMPT_1, "generation": {"do_sample": False, "max_new_tokens": 10},
            "total_params": int(total_params), "trainable_params": int(trainable_params),
            "phase": "phase2",
        }
        (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False))

        run_summary = {
            "config_id": cfg["config_id"], "best_epoch": best_epoch,
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
# Sweep-wide summary, dropout comparison, and REPORT.md
# ---------------------------------------------------------------------------
SUMMARY_COLUMNS = ["config_id", "learning_rate", "rank", "alpha", "dropout", "seed", "status",
                  "best_epoch", "precision", "recall", "f1", "invalid_count", "invalid_rate",
                  "train_loss", "val_loss", "epochs_completed", "duration_seconds", "stop_reason",
                  "run_path", "source"]


def phase1_reference_row(base):
    """Read-only: pulls the already-completed Phase 1 result for this
    (learning_rate, rank, alpha) triple at dropout=0.05. Never writes into
    PHASE1_SWEEP_DIR."""
    run_dir = PHASE1_SWEEP_DIR / phase1_reference_config_id(base)
    summary_path = run_dir / "run_summary.json"
    history_path = run_dir / "epoch_history.csv"
    if not summary_path.exists() or not history_path.exists():
        raise FileNotFoundError(
            f"Phase 1 referenca nedostaje ili je nekompletna: {run_dir} "
            f"(očekivano run_summary.json + epoch_history.csv)"
        )
    summary = json.loads(summary_path.read_text())
    history = pd.read_csv(history_path)
    best_epoch = summary["best_epoch"]
    best_row = history[history["epoch"] == best_epoch].iloc[0]
    return {
        "config_id": phase1_reference_config_id(base),
        "learning_rate": base["learning_rate"], "rank": base["rank"], "alpha": base["alpha"],
        "dropout": PHASE1_DROPOUT, "seed": SEED,
        "status": "completed", "best_epoch": int(best_epoch),
        "precision": float(best_row["precision"]), "recall": float(best_row["recall"]),
        "f1": float(best_row["f1"]), "invalid_count": int(best_row["invalid_count"]),
        "invalid_rate": float(best_row["invalid_rate"]),
        "train_loss": float(best_row["train_loss"]), "val_loss": float(best_row["val_loss"]),
        "epochs_completed": summary["last_completed_epoch"],
        "duration_seconds": summary["total_training_duration_seconds"],
        "stop_reason": summary["stop_reason"], "run_path": str(run_dir),
        "source": "phase1_reference",
    }


def _sort_rows_df(df):
    sort_key_f1 = df["f1"].fillna(-1)
    sort_key_recall = df["recall"].fillna(-1)
    sort_key_invalid = df["invalid_rate"].fillna(1e9)
    return df.assign(_f1=sort_key_f1, _recall=sort_key_recall, _invalid=sort_key_invalid) \
        .sort_values(by=["_f1", "_recall", "_invalid"], ascending=[False, False, True]) \
        .drop(columns=["_f1", "_recall", "_invalid"]).reset_index(drop=True)


def rebuild_phase2_summary(configs):
    """phase2_summary.csv: the six new Phase 2 runs only (source=sweep_phase2),
    whatever their current status (pending/running/completed/failed/interrupted)."""
    rows = []
    for cfg in configs:
        run_dir = SWEEP_DIR / cfg["config_id"]
        status = read_status(run_dir).get("status", "pending")
        row = {"config_id": cfg["config_id"], "learning_rate": cfg["learning_rate"],
               "rank": cfg["rank"], "alpha": cfg["alpha"], "dropout": cfg["dropout"],
               "seed": cfg["seed"], "status": status, "run_path": str(run_dir), "source": "sweep_phase2"}
        for k in ["best_epoch", "precision", "recall", "f1", "invalid_count", "invalid_rate",
                 "train_loss", "val_loss", "epochs_completed", "duration_seconds", "stop_reason"]:
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
            eh_path = run_dir / "epoch_history.csv"
            if eh_path.exists():
                eh = pd.read_csv(eh_path)
                best_r = eh[eh["epoch"] == s["best_epoch"]]
                if len(best_r):
                    row["train_loss"] = float(best_r.iloc[0]["train_loss"])
                    row["val_loss"] = float(best_r.iloc[0]["val_loss"])
        rows.append(row)

    df = _sort_rows_df(pd.DataFrame(rows)[SUMMARY_COLUMNS])
    df.to_csv(SWEEP_DIR / "phase2_summary.csv", index=False)
    return df


def write_dropout_comparison(configs):
    """dropout_comparison.csv: all nine rows — the six new Phase 2 runs plus
    the three Phase 1 dropout=0.05 references for the same (lr, rank, alpha)
    triples."""
    phase2_df = rebuild_phase2_summary(configs)
    ref_rows = [phase1_reference_row(base) for base in BASE_CONFIGS]
    ref_df = pd.DataFrame(ref_rows)[SUMMARY_COLUMNS]
    combined = _sort_rows_df(pd.concat([phase2_df, ref_df], ignore_index=True)[SUMMARY_COLUMNS])
    combined.to_csv(SWEEP_DIR / "dropout_comparison.csv", index=False)
    return combined


def write_phase2_state(configs, time_budget_minutes, incomplete, started_at):
    state = {
        "phase": "phase2", "time_budget_minutes": time_budget_minutes,
        "started_at": started_at, "last_updated_at": now_iso(),
        "incomplete": incomplete,
        "configs": {c["config_id"]: read_status(SWEEP_DIR / c["config_id"]).get("status", "pending")
                   for c in configs},
        "phase1_references": {phase1_reference_config_id(b): str(PHASE1_SWEEP_DIR / phase1_reference_config_id(b))
                              for b in BASE_CONFIGS},
    }
    (SWEEP_DIR / "phase2_state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))


def write_report(configs, comparison_df):
    lines = []
    lines.append("# Phase 2 dropout sweep — Gemma 3 1B IT LoRA, v2 dataset (bez refusal-a)\n")
    lines.append("## Cilj\n")
    lines.append(
        "Phase 1 (`scripts/model/sweep_lora_v2.py`) je pretražio `learning_rate x rank` na fiksnom "
        "`dropout=0.05` i pronašao tri najbolje (learning_rate, rank, alpha) kombinacije. Phase 2 uzima "
        "tačno te tri kombinacije i proverava da li podešavanje LoRA dropout-a "
        "(`dropout in {0.0, 0.1}`, naspram Phase 1 fiksnog 0.05) dalje poboljšava rezultat. "
        "Šest novih treninga; tri postojeća `dropout=0.05` rezultata iz Phase 1 se ne ponavljaju, "
        "samo se čitaju kao reference iz `scripts/model/results/gemma_lora_v2_sweep/` "
        "(Phase 1 folder ostaje netaknut).\n"
    )

    lines.append("## Svih devet rezultata (šest novih + tri Phase 1 reference)\n")
    lines.append(
        "Sortirano po F1 opadajuće, zatim recall opadajuće, zatim invalid rate rastuće. "
        "`status=pending` znači da taj run još nije izvršen.\n"
    )
    cols = ["config_id", "dropout", "source", "status", "best_epoch", "precision", "recall", "f1",
           "invalid_rate", "epochs_completed", "duration_seconds"]
    table_df = comparison_df[cols].copy()
    lines.append("```")
    lines.append(table_df.to_string(index=False))
    lines.append("```\n")

    completed = comparison_df[comparison_df["status"] == "completed"].reset_index(drop=True)
    lines.append("## Najbolja konfiguracija\n")
    if len(completed) == 0:
        lines.append(
            "Nijedan run još nije završen (sweep nije pokrenut ili je u toku) — nema šta da se "
            "proglasi najboljim još. Ova sekcija će se popuniti automatski nakon prvog završenog runa.\n"
        )
    else:
        best = completed.sort_values(by=["f1", "recall", "invalid_rate"],
                                     ascending=[False, False, True]).iloc[0]
        lines.append(
            f"**{best['config_id']}** ({best['source']}) — precision={best['precision']:.4f}, "
            f"recall={best['recall']:.4f}, F1={best['f1']:.4f}, invalid_rate={best['invalid_rate']:.4f}, "
            f"best_epoch={int(best['best_epoch'])}.\n"
        )

    lines.append("## Dve najbolje konfiguracije predložene za proveru na dodatnim seedovima\n")
    if len(completed) < 2:
        lines.append(
            "Nema dovoljno završenih runova još da se predlože dve najbolje konfiguracije za "
            "dodatni seed. Ova sekcija će se popuniti automatski čim bude završeno bar dva runa.\n"
        )
    else:
        top2 = completed.sort_values(by=["f1", "recall", "invalid_rate"],
                                     ascending=[False, False, True]).head(2)
        for _, row in top2.iterrows():
            lines.append(f"- **{row['config_id']}** — F1={row['f1']:.4f}, recall={row['recall']:.4f}")
        lines.append("")

    lines.append("## Tok najboljeg runa (po epohama)\n")
    if len(completed) == 0:
        lines.append("Nema još završenog runa čiju istoriju treninga bi trebalo prikazati.\n")
    else:
        best_row_path = Path(best["run_path"]) / "epoch_history.csv"
        if best_row_path.exists():
            hist = pd.read_csv(best_row_path)
            lines.append(f"Iz `{best_row_path}`:\n")
            lines.append("```")
            lines.append(hist[["epoch", "train_loss", "val_loss", "precision", "recall", "f1",
                              "invalid_count", "epoch_duration_seconds"]].to_string(index=False))
            lines.append("```\n")
        else:
            lines.append(f"Napomena: `{best_row_path}` ne postoji (Phase 1 referenca ili run bez sačuvane istorije).\n")

    lines.append("## Napomene\n")
    lines.append(
        f"- Test skup (`{_HELD_OUT_SPLIT_DISPLAY}`) NIJE korišćen ni za trening ni za "
        "evaluaciju bilo kog runa u ovoj fazi.\n"
        "- Dodatni seed-ovi za dve najbolje konfiguracije (predložene iznad) NISU automatski "
        "pokrenuti — to je namerno ostavljeno kao sledeći, ručno pokrenut korak.\n"
    )

    (SWEEP_DIR / "REPORT.md").write_text("\n".join(lines))


def estimate_next_run_minutes():
    summary_path = SWEEP_DIR / "phase2_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        completed = df[(df["status"] == "completed") & (df["source"] == "sweep_phase2")]
        if len(completed):
            avg_seconds = completed["duration_seconds"].mean()
            return (avg_seconds / 60.0) * RUN_ESTIMATE_SAFETY_FACTOR
    return FALLBACK_RUN_ESTIMATE_MINUTES


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def do_dry_run(configs):
    print("=" * 90)
    print("DRY RUN (Phase 2) — nema treninga, nema GPU alokacije")
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
    print(f"    [OK] Held-out split fajl se ne pojavljuje nigde u sweep_lora_v2_phase2.py "
          f"(traženi obrazac: {held_out_filename!r}).")

    print("\n[3] Provera da Phase 1 ostaje netaknut (samo čitanje, ne pisanje):")
    for label, p in [("Phase 1 sweep dir", PHASE1_SWEEP_DIR), ("Phase 1 Experiment 1 dir", PHASE1_EXP1_DIR)]:
        print(f"    {label}: {p} -> {'POSTOJI' if p.exists() else 'NE POSTOJI (!)'}")
    assert len(configs) == 6, f"Očekivano 6 novih Phase 2 runova, dobijeno {len(configs)}"

    print(f"\n[4] {len(configs)} novih Phase 2 runova (dropout sweep, redosled izvršavanja):")
    for i, c in enumerate(configs, start=1):
        run_dir = SWEEP_DIR / c["config_id"]
        status = read_status(run_dir).get("status", "pending") if run_dir.exists() else "pending"
        print(f"    {i}. {c['config_id']:45s} lr={c['learning_rate']:<8g} r={c['rank']:<3d} "
              f"alpha={c['alpha']:<3d} dropout={c['dropout']} [status: {status}]")

    print(f"\n[5] Tri Phase 1 reference (dropout=0.05, NEĆE se trenirati, samo se čitaju):")
    all_refs_found = True
    for base in BASE_CONFIGS:
        ref_id = phase1_reference_config_id(base)
        ref_dir = PHASE1_SWEEP_DIR / ref_id
        found = ref_dir.exists() and (ref_dir / "run_summary.json").exists()
        all_refs_found = all_refs_found and found
        print(f"    - {ref_id} -> {ref_dir} [{'OK' if found else 'NEDOSTAJE'}]")
    if all_refs_found:
        try:
            refs = [phase1_reference_row(b) for b in BASE_CONFIGS]
            print("\n    Učitane referentne vrednosti (samo čitanje, ništa se ne piše u Phase 1 folder):")
            for r in refs:
                print(f"      {r['config_id']}: P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} "
                      f"invalid_rate={r['invalid_rate']:.4f}")
        except Exception as e:
            print(f"    [WARN] Greška pri čitanju referenci: {e}")
    else:
        print("    [WARN] Bar jedna Phase 1 referenca nedostaje — proveri da Phase 1 sweep nije premešten/obrisan.")

    print(f"\n[6] Procenjeno vreme po run-u: {estimate_next_run_minutes():.1f} min "
          f"(fallback konstanta dok Phase 2 nema nijedan sopstveni završen run).")

    print(f"\n[OK] Dry run završen bez učitavanja modela na GPU, bez pokretanja treninga, "
          f"bez izmena u Phase 1 folderima, bez čitanja {_HELD_OUT_SPLIT_NAME}.")


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
    invocation_start = time.time()
    started_at = now_iso()
    print(f"[START] Phase 2 sweep pokrenut {started_at} (PID {os.getpid()}), budžet {args.time_budget_minutes} min.")

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

        comparison_df = write_dropout_comparison(configs)
        write_report(configs, comparison_df)
        write_phase2_state(configs, args.time_budget_minutes, incomplete=False, started_at=started_at)
    else:
        remaining = [c for c in configs
                    if read_status(SWEEP_DIR / c["config_id"]).get("status") != "completed"]
        incomplete = len(remaining) > 0

    comparison_df = write_dropout_comparison(configs)
    write_report(configs, comparison_df)
    write_phase2_state(configs, args.time_budget_minutes, incomplete=incomplete, started_at=started_at)

    print("\n" + "=" * 90)
    print(f"PHASE 2 SWEEP {'NEPOTPUN — nastaviće se pri sledećem pokretanju' if incomplete else 'KOMPLETAN'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
