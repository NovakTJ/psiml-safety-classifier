#!/usr/bin/env python
"""Standalone Phase 1 LoRA sweep — Gemma 3 1B IT, v2 (no-refusal) dataset.

Extracted and parametrized from the "Experiment 1" section of
scripts/model/gemma_lora.ipynb (r=8, lr=2e-4 config), which itself was
executed and verified as a standalone script before being embedded in the
notebook. Everything except learning_rate/rank/alpha is identical to that
proven Experiment 1 configuration.

Grid (Phase 1 only):
    learning_rate in {5e-5, 1e-4, 2e-4, 3e-4}
    rank          in {4, 8, 16}
    alpha = 2 * rank
    dropout = 0.05 (fixed for Phase 1)

The (lr=2e-4, rank=8) cell is the already-completed Experiment 1 run — it is
never retrained here, only folded into sweep_summary.csv as
source=existing_reference. That leaves exactly 11 new training runs.

Usage:
    ~/ccpp_env/bin/python sweep_lora_v2.py --dry-run
    ~/ccpp_env/bin/python sweep_lora_v2.py --time-budget-minutes 340

Safe to re-run: completed configs are skipped, failed configs are retried up
to --max-attempts times, and a config found "running" at startup (meaning
the previous process died mid-training) is restarted from scratch — this
script never resumes a Trainer from a mid-training checkpoint, since exact
optimizer/scheduler/callback-state reconstruction cannot be guaranteed
without that machinery, and the fallback of "restart the config from the
base model" was explicitly authorized instead.
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

SWEEP_DIR = Path("/home/mls01/scripts/model/results/gemma_lora_v2_sweep")
EXISTING_EXP1_DIR = Path(
    "/home/mls01/scripts/model/results/gemma_lora_v2_exp1_r8_lr2e4_seed42_max8_es2"
)
LOCK_PATH = SWEEP_DIR / "sweep.pid"

LEARNING_RATES = [5e-5, 1e-4, 2e-4, 3e-4]
RANKS = [4, 8, 16]
DROPOUT = 0.05
SEED = 42
MAX_EPOCHS = 8
PATIENCE = 2
MIN_DELTA = 0.0
MAX_SEQ_LENGTH = 1024
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

EXISTING_REFERENCE_LR = 2e-4
EXISTING_REFERENCE_RANK = 8

DEFAULT_TIME_BUDGET_MINUTES = 340
DEFAULT_MAX_ATTEMPTS = 2
# Used only until this sweep has at least one completed run to measure from.
FALLBACK_RUN_ESTIMATE_MINUTES = 45.0
RUN_ESTIMATE_SAFETY_FACTOR = 1.3

PROMPT_1 = """Classify the following sample as harmful or unharmful.

A sample is harmful if the user prompt is harmful or the assistant response contains harmful content. Otherwise, it is unharmful."""
PROMPT_1 += "\n\nReturn only: harmful or unharmful."

REQUIRED_LOCKED_VERSIONS = {
    "torch": "2.11.0+cu128",
    "transformers": "4.57.6",
    "huggingface_hub": "0.36.0",
}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------
def format_lr(lr):
    # 5e-05 -> 5e-5, 1e-04 -> 1e-4, etc. (matches the requested config_id style)
    return f"{lr:.0e}".replace("e-0", "e-")


def config_id(lr, rank, alpha, dropout, seed):
    return f"lr{format_lr(lr)}_r{rank}_alpha{alpha}_dropout{dropout}_seed{seed}"


def build_grid():
    configs = []
    for lr in LEARNING_RATES:
        for rank in RANKS:
            alpha = 2 * rank
            is_existing = (abs(lr - EXISTING_REFERENCE_LR) < 1e-12 and rank == EXISTING_REFERENCE_RANK)
            configs.append({
                "config_id": config_id(lr, rank, alpha, DROPOUT, SEED),
                "learning_rate": lr,
                "rank": rank,
                "alpha": alpha,
                "dropout": DROPOUT,
                "seed": SEED,
                "is_existing_reference": is_existing,
            })
    return configs


# ---------------------------------------------------------------------------
# Concurrency lock (defense in depth — the launcher also checks this file
# before ever starting python, but the script re-checks in case it is
# invoked directly).
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
            print(f"[FATAL] Sweep već aktivan (PID {existing_pid}, lock: {LOCK_PATH}). Izlazim bez pokretanja.")
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
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    assert (all_df["final_label"] == all_df["prompt_harm_label"]).all()
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
# F1-based early-stopping callback (identical logic to Experiment 1's
# ExplicitF1EarlyStoppingCallback — see gemma_lora.ipynb for the verified
# on_save hook-ordering rationale). Writes status.json after every epoch.
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
# Single-config training run
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
# Sweep-wide summary
# ---------------------------------------------------------------------------
SUMMARY_COLUMNS = ["config_id", "learning_rate", "rank", "alpha", "dropout", "seed", "status",
                  "best_epoch", "precision", "recall", "f1", "invalid_count", "invalid_rate",
                  "train_loss", "val_loss", "epochs_completed", "duration_seconds", "stop_reason",
                  "run_path", "source"]


def existing_reference_row():
    exp1_history = pd.read_csv(EXISTING_EXP1_DIR / "epoch_history.csv")
    exp1_summary = json.loads((EXISTING_EXP1_DIR / "run_summary.json").read_text())
    r2 = exp1_history[exp1_history["epoch"] == 2].iloc[0]
    return {
        "config_id": config_id(EXISTING_REFERENCE_LR, EXISTING_REFERENCE_RANK,
                               2 * EXISTING_REFERENCE_RANK, DROPOUT, SEED),
        "learning_rate": EXISTING_REFERENCE_LR, "rank": EXISTING_REFERENCE_RANK,
        "alpha": 2 * EXISTING_REFERENCE_RANK, "dropout": DROPOUT, "seed": SEED,
        "status": "completed", "best_epoch": 2,
        "precision": float(r2["precision"]), "recall": float(r2["recall"]), "f1": float(r2["f1"]),
        "invalid_count": int(r2["invalid_count"]), "invalid_rate": float(r2["invalid_rate"]),
        "train_loss": float(r2["train_loss"]), "val_loss": float(r2["val_loss"]),
        "epochs_completed": exp1_summary["last_completed_epoch"],
        "duration_seconds": exp1_summary["total_training_duration_seconds"],
        "stop_reason": exp1_summary["stop_reason"], "run_path": str(EXISTING_EXP1_DIR),
        "source": "existing_reference",
    }


def rebuild_sweep_summary(configs):
    rows = []
    for cfg in configs:
        if cfg["is_existing_reference"]:
            rows.append(existing_reference_row())
            continue
        run_dir = SWEEP_DIR / cfg["config_id"]
        status = read_status(run_dir).get("status", "pending")
        row = {"config_id": cfg["config_id"], "learning_rate": cfg["learning_rate"],
               "rank": cfg["rank"], "alpha": cfg["alpha"], "dropout": cfg["dropout"],
               "seed": cfg["seed"], "status": status, "run_path": str(run_dir), "source": "sweep_phase1"}
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

    df = pd.DataFrame(rows)[SUMMARY_COLUMNS]
    sort_key_f1 = df["f1"].fillna(-1)
    sort_key_recall = df["recall"].fillna(-1)
    sort_key_invalid = df["invalid_rate"].fillna(1e9)
    df = df.assign(_f1=sort_key_f1, _recall=sort_key_recall, _invalid=sort_key_invalid) \
        .sort_values(by=["_f1", "_recall", "_invalid"], ascending=[False, False, True]) \
        .drop(columns=["_f1", "_recall", "_invalid"]).reset_index(drop=True)
    df.to_csv(SWEEP_DIR / "sweep_summary.csv", index=False)
    return df


def write_sweep_state(configs, time_budget_minutes, incomplete, started_at):
    state = {
        "phase": "phase1", "time_budget_minutes": time_budget_minutes,
        "started_at": started_at, "last_updated_at": now_iso(),
        "incomplete": incomplete,
        "configs": {c["config_id"]: read_status(SWEEP_DIR / c["config_id"]).get("status", "pending")
                   if not c["is_existing_reference"] else "existing_reference" for c in configs},
    }
    (SWEEP_DIR / "sweep_state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))


def estimate_next_run_minutes():
    summary_path = SWEEP_DIR / "sweep_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        completed = df[(df["status"] == "completed") & (df["source"] == "sweep_phase1")]
        if len(completed):
            avg_seconds = completed["duration_seconds"].mean()
            return (avg_seconds / 60.0) * RUN_ESTIMATE_SAFETY_FACTOR
    return FALLBACK_RUN_ESTIMATE_MINUTES


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------
def do_dry_run(configs):
    print("=" * 90)
    print("DRY RUN — nema treninga, nema GPU alokacije")
    print("=" * 90)

    print("\n[1] Provera putanja:")
    for label, p in [("model", MODEL_PATH), ("train.jsonl", TRAIN_PATH), ("validation.jsonl", VAL_PATH)]:
        print(f"    {label}: {p} -> {'POSTOJI' if p.exists() else 'NE POSTOJI'}")

    train_df, val_df = load_and_validate_dataset()

    print("\n[2] Self-audit: held-out split fajl se nigde ne pominje u ovom fajlu:")
    own_source = Path(__file__).read_text()
    # Sastavljeno iz delova namerno — da bi sam ovaj audit-kod ostao izvan
    # sopstvene provere (inače bi ova linija samu sebe lažno okidala).
    held_out_filename = "test" + "." + "jsonl"
    assert held_out_filename not in own_source, \
        f"Held-out split fajl je pronađen u izvornom kodu ovog sweep-a — PREKID."
    print(f"    [OK] Held-out split fajl se ne pojavljuje nigde u sweep_lora_v2.py "
          f"(traženi obrazac: {held_out_filename!r}).")

    new_configs = [c for c in configs if not c["is_existing_reference"]]
    existing = [c for c in configs if c["is_existing_reference"]]
    assert len(new_configs) == 11, f"Očekivano 11 novih runova, dobijeno {len(new_configs)}"
    assert len(existing) == 1

    print(f"\n[3] Postojeća referenca (NEĆE se trenirati): {existing[0]['config_id']}")
    print(f"    Rezultat se čita iz: {EXISTING_EXP1_DIR}")

    print(f"\n[4] {len(new_configs)} novih runova (redosled izvršavanja):")
    for i, c in enumerate(new_configs, start=1):
        run_dir = SWEEP_DIR / c["config_id"]
        status = read_status(run_dir).get("status", "pending") if run_dir.exists() else "pending"
        print(f"    {i:2d}. {c['config_id']:45s} lr={c['learning_rate']:<8g} r={c['rank']:<3d} "
              f"alpha={c['alpha']:<3d} dropout={c['dropout']} [status: {status}]")

    print(f"\n[5] Procenjeno vreme po run-u: {estimate_next_run_minutes():.1f} min "
          f"(fallback konstanta dok nema završenih runova u ovom sweep-u).")

    print("\n[OK] Dry run završen bez učitavanja modela na GPU i bez pokretanja treninga.")


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
                            f"(podrazumevano {DEFAULT_MAX_ATTEMPTS}; nije eksplicitno traženo u specifikaciji, "
                            f"dodato da sweep ne troši ceo budžet na konfiguraciju koja stalno pada).")
    args = parser.parse_args()

    configs = build_grid()

    if args.dry_run:
        do_dry_run(configs)
        return

    acquire_lock()
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    invocation_start = time.time()
    started_at = now_iso()
    print(f"[START] Sweep pokrenut {started_at} (PID {os.getpid()}), budžet {args.time_budget_minutes} min.")

    train_df, val_df = load_and_validate_dataset()
    tokenizer, end_of_turn_id, build_example, parse_label = build_tokenizer_and_helpers()
    train_examples, _ = build_split(train_df, "train", build_example)
    val_examples, _ = build_split(val_df, "validation", build_example)
    val_prefixes = build_val_prefixes(val_df, build_example)
    evaluate_checkpoint = make_evaluate_checkpoint(tokenizer, parse_label)
    SFTDataset = make_sft_dataset_cls()
    collate_fn = make_collate_fn(tokenizer)

    new_configs = [c for c in configs if not c["is_existing_reference"]]
    incomplete = False

    for cfg in new_configs:
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

        rebuild_sweep_summary(configs)
        write_sweep_state(configs, args.time_budget_minutes, incomplete=False, started_at=started_at)
    else:
        remaining = [c for c in new_configs
                    if read_status(SWEEP_DIR / c["config_id"]).get("status") != "completed"]
        incomplete = len(remaining) > 0

    rebuild_sweep_summary(configs)
    write_sweep_state(configs, args.time_budget_minutes, incomplete=incomplete, started_at=started_at)

    print("\n" + "=" * 90)
    print(f"SWEEP {'NEPOTPUN — nastaviće se pri sledećem pokretanju' if incomplete else 'KOMPLETAN'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
