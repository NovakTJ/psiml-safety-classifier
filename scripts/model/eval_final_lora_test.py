#!/usr/bin/env python
"""Final, one-shot TEST evaluation of the locked all-linear LoRA config — Gemma 3
1B IT, v2 (no-refusal) dataset, seed=42.

This script does NOT retrain anything and does NOT run validation again. It:

  1. Locates and locks the existing seed=42 best_adapter for the multi-seed
     winner config (learning_rate=3e-4, rank=8, alpha=16, dropout=0.0) under
     scripts/model/results/gemma_lora_v2_sweep_phase2/, reading the exact path
     from that run's run_config.json / run_summary.json (never assumed).
  2. Hashes (SHA256) every adapter weight file before and after evaluation to
     prove the adapter was not mutated.
  3. Runs that adapter exactly once on data/gemma_v2_no_refusal/test.jsonl,
     reusing the tokenizer, PROMPT_1 text, chat template, head-tail truncation,
     and greedy-generation (do_sample=False, max_new_tokens=10) helpers
     imported directly from sweep_lora_v2_phase2.py — not reimplemented — so
     the inference pipeline is byte-for-byte identical to the one used during
     validation/training.
  4. Computes both "valid predictions only" and "end-to-end" (invalid counted
     as an error) metrics.
  5. Loads (never re-runs) the existing test predictions for the v2 zero-shot
     system and the Experiment 1 LoRA (seed=42) system, verifies all three
     systems share the identical 227 test row_id / final_label values, and
     recomputes their metrics with the exact same methodology used here.
  6. Saves everything under
     scripts/model/results/gemma_lora_v2_multiseed/final_test_seed42/ and
     prints terminal summary tables.

Usage:
    ~/ccpp_env/bin/python scripts/model/eval_final_lora_test.py
"""

import os

# ---------------------------------------------------------------------------
# Required env-var block — MUST run before importing torch/transformers.
# uid 1562 has no /etc/passwd entry in this container; without these,
# transformers crashes with "getpwuid(): uid not found: 1562".
# ---------------------------------------------------------------------------
os.environ.setdefault("USER", "mls01")
os.environ.setdefault("LOGNAME", "mls01")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/home/mls01/.cache/torchinductor")
os.environ.setdefault("TRITON_CACHE_DIR", "/home/mls01/.cache/triton")
os.environ.setdefault("XDG_CACHE_HOME", "/home/mls01/.cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import sweep_lora_v2_phase2 as p2  # noqa: E402  (reuse exact pipeline, not reimplemented)

# ---------------------------------------------------------------------------
# Fixed paths
# ---------------------------------------------------------------------------
PHASE2_RUN_DIR = SCRIPT_DIR / "results" / "gemma_lora_v2_sweep_phase2" / "lr3e-4_r8_alpha16_dropout0.0_seed42"
ADAPTER_DIR = PHASE2_RUN_DIR / "best_adapter"
RUN_CONFIG_PATH = PHASE2_RUN_DIR / "run_config.json"
RUN_SUMMARY_PATH = PHASE2_RUN_DIR / "run_summary.json"

MULTISEED_SUMMARY_PATH = SCRIPT_DIR / "results" / "gemma_lora_v2_multiseed" / "multiseed_summary.csv"

TEST_PATH = Path("/home/mls01/data/gemma_v2_no_refusal/test.jsonl")

ZEROSHOT_DIR = SCRIPT_DIR / "results" / "gemma_demo_zeroshot_v2_no_refusal"
ZEROSHOT_TEST_CSV = ZEROSHOT_DIR / "test_results_full.csv"

EXP1_DIR = SCRIPT_DIR / "results" / "gemma_lora_v2_exp1_r8_lr2e4_seed42_max8_es2"
EXP1_TEST_CSV = EXP1_DIR / "test_results_full.csv"

OUT_DIR = SCRIPT_DIR / "results" / "gemma_lora_v2_multiseed" / "final_test_seed42"

POSITIVE = "harmful"

SYSTEM_NAMES = {
    "zero_shot": "Zero-shot v2 (prompt_1)",
    "first_lora": "First LoRA — Experiment 1 (seed=42)",
    "final_lora": "Final LoRA — locked all-linear config (seed=42)",
}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_adapter_dir(adapter_dir):
    files = sorted(p.name for p in adapter_dir.iterdir() if p.is_file())
    return {name: sha256_file(adapter_dir / name) for name in files}


# ---------------------------------------------------------------------------
# Metrics — shared methodology for all three systems.
# "valid" metrics: precision/recall/f1 computed only over rows whose
# prediction is exactly "harmful" or "unharmful".
# "end-to-end" metrics: every invalid row is folded in as an error — invalid
# on a truly-harmful row becomes an FN, invalid on a truly-unharmful row
# becomes an FP. Invalid is never converted into a TN.
# ---------------------------------------------------------------------------
def _derive(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    balanced_accuracy = (recall + specificity) / 2
    mcc_denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": accuracy, "specificity": specificity,
        "fpr": fpr, "fnr": fnr, "balanced_accuracy": balanced_accuracy, "mcc": mcc,
    }


def compute_metrics_both(y_true, y_pred, positive=POSITIVE):
    n = len(y_true)
    tp = fp = fn = tn = 0
    e_tp = e_fp = e_fn = e_tn = 0
    invalid_count = 0
    for t, p in zip(y_true, y_pred):
        if p == "invalid":
            invalid_count += 1
            if t == positive:
                e_fn += 1
            else:
                e_fp += 1
            continue
        if t == positive and p == positive:
            tp += 1
            e_tp += 1
        elif t != positive and p == positive:
            fp += 1
            e_fp += 1
        elif t == positive and p != positive:
            fn += 1
            e_fn += 1
        else:
            tn += 1
            e_tn += 1

    valid_count = n - invalid_count
    valid_metrics = _derive(tp, fp, fn, tn)
    valid_metrics.update({"valid_count": valid_count, "invalid_count": invalid_count,
                           "invalid_rate": invalid_count / n if n else 0.0, "total": n})

    e2e_metrics = _derive(e_tp, e_fp, e_fn, e_tn)
    e2e_metrics.update({"invalid_count": invalid_count,
                         "invalid_rate": invalid_count / n if n else 0.0, "total": n})
    return valid_metrics, e2e_metrics


# ---------------------------------------------------------------------------
# Load an existing system's test predictions, verify alignment against the
# canonical test_df (same 227 row_id, same final_label), abort loudly if not.
# ---------------------------------------------------------------------------
def load_prior_system(name, csv_path, test_df):
    if not csv_path.exists():
        raise FileNotFoundError(f"[{name}] Nedostaje fajl sa test predikcijama: {csv_path}")
    df = pd.read_csv(csv_path)
    if "row_id" not in df.columns or "prediction" not in df.columns:
        raise ValueError(f"[{name}] {csv_path} nema očekivane kolone row_id/prediction.")

    test_ids = set(test_df["row_id"])
    sys_ids = set(df["row_id"])
    if sys_ids != test_ids:
        missing = test_ids - sys_ids
        extra = sys_ids - test_ids
        raise ValueError(
            f"[{name}] row_id skup iz {csv_path} se NE poklapa sa test.jsonl. "
            f"Nedostaje {len(missing)} (npr. {list(missing)[:5]}), višak {len(extra)} (npr. {list(extra)[:5]})."
        )
    if len(df) != len(test_df):
        raise ValueError(
            f"[{name}] {csv_path} ima {len(df)} redova, test.jsonl ima {len(test_df)} — broj redova se ne poklapa."
        )

    merged = test_df[["row_id", "final_label"]].merge(
        df[["row_id", "final_label", "prediction", "raw_output"]],
        on="row_id", suffixes=("_test", "_system"),
    )
    mismatched = merged[merged["final_label_test"] != merged["final_label_system"]]
    if len(mismatched):
        raise ValueError(
            f"[{name}] final_label se ne poklapa na {len(mismatched)} redova iz {csv_path}, "
            f"npr. row_id={mismatched['row_id'].tolist()[:5]} — PREKID poređenja."
        )

    merged = merged.set_index("row_id").loc[test_df["row_id"]].reset_index()
    print(f"    [OK] [{name}] {csv_path.name}: {len(merged)}/227 row_id poklapaju, final_label identičan.")
    return merged["final_label_test"].tolist(), merged["prediction"].tolist(), merged["raw_output"].tolist()


def fmt_pct(x):
    return f"{x * 100:.2f}%"


def print_table(rows, headers, col_widths=None):
    if col_widths is None:
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) + 2 for i, h in enumerate(headers)]
    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        print("".join(str(c).ljust(w) for c, w in zip(r, col_widths)))


def main():
    print("=" * 100)
    print("KONAČNA TEST EVALUACIJA — zaključani all-linear LoRA, seed=42")
    print("=" * 100)

    # ------------------------------------------------------------------
    # 1. Locate + verify the locked adapter from Phase 2's own records.
    # ------------------------------------------------------------------
    print("\n[1] Pronalaženje i provera zaključanog adaptera (seed=42)")
    if not (RUN_CONFIG_PATH.exists() and RUN_SUMMARY_PATH.exists() and ADAPTER_DIR.exists()):
        raise FileNotFoundError(
            f"Očekivani Phase 2 run nije pronađen na {PHASE2_RUN_DIR} "
            f"(run_config.json / run_summary.json / best_adapter/)."
        )
    run_config = json.loads(RUN_CONFIG_PATH.read_text())
    run_summary = json.loads(RUN_SUMMARY_PATH.read_text())

    expected = {"learning_rate": 3e-4, "rank": 8, "alpha": 16, "dropout": 0.0, "seed": 42}
    for k, v in expected.items():
        assert run_config[k] == v, f"run_config.json[{k}]={run_config[k]!r}, očekivano {v!r}"
    assert run_config["target_modules"] == [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ], f"target_modules se ne poklapaju: {run_config['target_modules']}"
    assert run_config["prompt"] == p2.PROMPT_1, "run_config.json prompt se ne poklapa sa PROMPT_1 u sweep_lora_v2_phase2.py"
    assert run_config["generation"] == {"do_sample": False, "max_new_tokens": 10}
    assert (ADAPTER_DIR / "adapter_config.json").exists() and (ADAPTER_DIR / "adapter_model.safetensors").exists()

    print(f"    [OK] Konfiguracija potvrđena iz {RUN_CONFIG_PATH}")
    print(f"    [OK] Adapter: {ADAPTER_DIR}")
    print(f"    [OK] best_epoch={run_summary['best_epoch']}, "
          f"validation F1={run_summary['best_metrics']['f1']:.4f}, "
          f"precision={run_summary['best_metrics']['precision']:.4f}, "
          f"recall={run_summary['best_metrics']['recall']:.4f}, "
          f"invalid_rate={run_summary['best_metrics']['invalid_rate']:.4f}")

    multiseed_context = None
    multiseed_context_b = None
    if MULTISEED_SUMMARY_PATH.exists():
        ms = pd.read_csv(MULTISEED_SUMMARY_PATH)
        row_a = ms[ms["config_name"] == "config_a"]
        row_b = ms[ms["config_name"] == "config_b"]
        if len(row_a):
            r = row_a.iloc[0]
            multiseed_context = {
                "config_name": "config_a", "seeds": r["seeds"],
                "mean_f1": float(r["mean_f1"]), "std_f1": float(r["std_f1"]),
                "mean_precision": float(r["mean_precision"]), "std_precision": float(r["std_precision"]),
                "mean_recall": float(r["mean_recall"]), "std_recall": float(r["std_recall"]),
            }
            print(f"    [OK] Multi-seed kontekst učitan: config_a mean F1={r['mean_f1']:.4f}±{r['std_f1']:.4f} "
                  f"preko seed-ova {r['seeds']}")
        if len(row_b):
            r = row_b.iloc[0]
            multiseed_context_b = {
                "config_name": "config_b", "seeds": r["seeds"],
                "mean_f1": float(r["mean_f1"]), "std_f1": float(r["std_f1"]),
            }

    hashes_before = hash_adapter_dir(ADAPTER_DIR)
    print("    [HASH] SHA256 adapter fajlova PRE evaluacije:")
    for name, h in sorted(hashes_before.items()):
        print(f"        {name}: {h}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Load + validate the test set.
    # ------------------------------------------------------------------
    print("\n[2] Učitavanje i provera test skupa")
    test_df = pd.read_json(TEST_PATH, lines=True)
    assert len(test_df) == 227, f"Očekivano 227 redova, dobijeno {len(test_df)}"
    assert test_df["row_id"].is_unique, "row_id nije jedinstven u test.jsonl"
    n_groups = test_df["original_idx"].nunique()
    assert n_groups == 100, f"Očekivano 100 original_idx grupa, dobijeno {n_groups}"
    label_dist = test_df["final_label"].value_counts().to_dict()
    print(f"    [OK] {TEST_PATH}")
    print(f"    [OK] {len(test_df)} redova, {n_groups} original_idx grupa, row_id jedinstveni")
    print(f"    [OK] final_label distribucija: {label_dist}")

    # ------------------------------------------------------------------
    # 3. Tokenizer + truncation + parser — reused unchanged from Phase 2.
    # ------------------------------------------------------------------
    print("\n[3] Priprema tokenizacije (isti pipeline kao validation, uvezen iz sweep_lora_v2_phase2.py)")
    tokenizer, end_of_turn_id, build_example, parse_label = p2.build_tokenizer_and_helpers()

    test_prefixes, truncated_flags = [], []
    for row in test_df.itertuples(index=False):
        ex = build_example(row.prompt, row.response, row.final_label)
        test_prefixes.append(ex["input_ids"][:ex["prefix_len"]])
        truncated_flags.append(ex["truncated"])
    n_truncated = int(sum(truncated_flags))
    print(f"    [OK] {len(test_prefixes)} test primera tokenizovano | skraćeno {n_truncated} "
          f"({n_truncated / len(test_prefixes) * 100:.2f}%)")

    # ------------------------------------------------------------------
    # 4. Load base model (local, non-quantized BF16) + locked adapter.
    # ------------------------------------------------------------------
    print("\n[4] Učitavanje baznog modela (BF16, nekvantizovan, lokalno) + zaključanog adaptera")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained(
        p2.MODEL_PATH, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager",
    ).to("cuda")
    peft_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    print(f"    [OK] {p2.MODEL_PATH} + {ADAPTER_DIR}")

    # ------------------------------------------------------------------
    # 5. Run inference exactly once (do_sample=False, max_new_tokens=10).
    # ------------------------------------------------------------------
    print("\n[5] Pokretanje inferencije na test skupu (jednom, greedy, max_new_tokens=10)")
    evaluate_checkpoint = p2.make_evaluate_checkpoint(tokenizer, parse_label)
    raw_outputs, predictions = evaluate_checkpoint(peft_model, test_prefixes, "test")

    del base_model, peft_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 6. Re-hash adapter, confirm unchanged.
    # ------------------------------------------------------------------
    print("\n[6] Ponovna SHA256 provera adaptera POSLE evaluacije")
    hashes_after = hash_adapter_dir(ADAPTER_DIR)
    hash_unchanged = hashes_before == hashes_after
    for name in sorted(hashes_before):
        status = "IDENTIČAN" if hashes_before[name] == hashes_after.get(name) else "PROMENJEN !!"
        print(f"        {name}: {status}")
    if not hash_unchanged:
        raise RuntimeError("Adapter fajlovi su se promenili tokom evaluacije — REZULTATI SE ODBACUJU.")
    print("    [OK] Adapter nije promenjen tokom evaluacije.")

    # ------------------------------------------------------------------
    # 7. Final LoRA — build results_full, metrics, error-analysis files.
    # ------------------------------------------------------------------
    print("\n[7] Računanje metrika za finalni LoRA model")
    final_true = test_df["final_label"].tolist()
    final_valid_metrics, final_e2e_metrics = compute_metrics_both(final_true, predictions)

    results_full = test_df.copy()
    results_full["prediction"] = predictions
    results_full["raw_output"] = raw_outputs
    results_full["is_valid"] = results_full["prediction"].isin(["harmful", "unharmful"])
    results_full["is_correct"] = results_full["prediction"] == results_full["final_label"]
    results_full["eval_truncated"] = truncated_flags
    results_full.to_csv(OUT_DIR / "test_results_full.csv", index=False)

    ERROR_COLS = ["row_id", "original_idx", "prompt", "response", "final_label",
                  "prediction", "raw_output", "language", "augmentation_type", "adversarial"]

    fp_mask = (results_full["final_label"] != POSITIVE) & (results_full["prediction"] == POSITIVE)
    fn_mask = (results_full["final_label"] == POSITIVE) & (results_full["prediction"] != POSITIVE) & \
              (results_full["prediction"] != "invalid")
    invalid_mask = results_full["prediction"] == "invalid"

    results_full.loc[fp_mask, ERROR_COLS].to_csv(OUT_DIR / "test_false_positives.csv", index=False)
    results_full.loc[fn_mask, ERROR_COLS].to_csv(OUT_DIR / "test_false_negatives.csv", index=False)
    results_full.loc[invalid_mask, ERROR_COLS].to_csv(OUT_DIR / "test_invalid_examples.csv", index=False)

    cm = pd.DataFrame(
        [[final_valid_metrics["tp"], final_valid_metrics["fn"]],
         [final_valid_metrics["fp"], final_valid_metrics["tn"]]],
        index=["true_harmful", "true_unharmful"], columns=["pred_harmful", "pred_unharmful"],
    )
    cm.to_csv(OUT_DIR / "test_confusion_matrix.csv")

    (OUT_DIR / "test_metrics_valid.json").write_text(json.dumps({
        "metrics_on_valid_predictions": final_valid_metrics,
        "dataset_summary": {
            "test_rows": len(test_df), "test_groups": n_groups,
            "final_label_distribution": label_dist,
            "eval_truncated": n_truncated,
            "eval_truncated_rate": n_truncated / len(test_df),
        },
    }, indent=2, ensure_ascii=False))
    (OUT_DIR / "test_metrics_end_to_end.json").write_text(
        json.dumps({"end_to_end_metrics": final_e2e_metrics}, indent=2, ensure_ascii=False)
    )

    print(f"    [OK] Final LoRA valid-only:  P={final_valid_metrics['precision']:.4f} "
          f"R={final_valid_metrics['recall']:.4f} F1={final_valid_metrics['f1']:.4f} "
          f"invalid={final_valid_metrics['invalid_count']}")
    print(f"    [OK] Final LoRA end-to-end:  P={final_e2e_metrics['precision']:.4f} "
          f"R={final_e2e_metrics['recall']:.4f} F1={final_e2e_metrics['f1']:.4f}")

    # ------------------------------------------------------------------
    # 8. locked_config.json / locked_adapter.json / adapter_hashes.json
    # ------------------------------------------------------------------
    locked_config = {
        "learning_rate": run_config["learning_rate"], "rank": run_config["rank"],
        "alpha": run_config["alpha"], "dropout": run_config["dropout"],
        "max_epochs": run_config["max_epochs"], "patience": run_config["patience"],
        "min_delta": run_config["min_delta"], "max_seq_length": run_config["max_seq_length"],
        "target_modules": run_config["target_modules"], "bias": "none", "task_type": "CAUSAL_LM",
        "seed": run_config["seed"], "data_seed": run_config["data_seed"],
        "prompt": run_config["prompt"], "generation": run_config["generation"],
        "selection_basis": ("Konfiguracija izabrana na osnovu multi-seed VALIDATION rezultata "
                             "(seeds 23, 41, 42) — vidi scripts/model/results/gemma_lora_v2_multiseed/REPORT.md. "
                             "Za konačnu test evaluaciju unapred je izabran isključivo seed=42 "
                             "(naučno uporedivo sa Experiment 1, koji je takođe seed=42). "
                             "Seedovi 23 i 41 NISU evaluirani na test skupu."),
        "multiseed_validation_context": multiseed_context,
        "source_run_config_path": str(RUN_CONFIG_PATH),
        "source_run_summary_path": str(RUN_SUMMARY_PATH),
        "locked_at": now_iso(),
    }
    (OUT_DIR / "locked_config.json").write_text(json.dumps(locked_config, indent=2, ensure_ascii=False))

    locked_adapter = {
        "base_model_path": str(p2.MODEL_PATH),
        "locked_adapter_path": str(ADAPTER_DIR),
        "seed": 42,
        "selected_epoch": run_summary["best_epoch"],
        "selection_criterion": (
            "seed=42 je unapred izabran za konačnu test evaluaciju (naučno uporediv sa Experiment 1 "
            "seed=42), na osnovu multi-seed validation rezultata koji potvrđuju da je ova konfiguracija "
            "(config A) stabilan pobednik preko seed-ova {23,41,42}. Unutar seed=42, best_epoch je izabran "
            "isključivo po maksimalnom validation harmful F1 tokom Phase 2 treninga (early stopping), bez "
            "ikakvog uvida u test skup."
        ),
        "validation_metrics": run_summary["best_metrics"],
        "multiseed_validation_context": multiseed_context,
        "lora_config": {
            "r": run_config["rank"], "alpha": run_config["alpha"], "dropout": run_config["dropout"],
            "bias": "none", "task_type": "CAUSAL_LM", "target_modules": run_config["target_modules"],
        },
        "prompt": run_config["prompt"],
        "parser": {"steps": ["strip()", "lowercase", "accept only exact 'harmful' or 'unharmful'",
                              "everything else -> 'invalid'", "raw_output always preserved"]},
        "generation": run_config["generation"],
        "max_seq_length": run_config["max_seq_length"],
    }
    (OUT_DIR / "locked_adapter.json").write_text(json.dumps(locked_adapter, indent=2, ensure_ascii=False))

    adapter_hashes = {
        "adapter_dir": str(ADAPTER_DIR), "algorithm": "sha256",
        "hashes_before_evaluation": hashes_before, "hashes_after_evaluation": hashes_after,
        "unchanged": hash_unchanged,
    }
    (OUT_DIR / "adapter_hashes.json").write_text(json.dumps(adapter_hashes, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------
    # 9. Load (never re-run) zero-shot + Experiment 1 test predictions,
    #    verify alignment, recompute metrics with identical methodology.
    # ------------------------------------------------------------------
    print("\n[8] Učitavanje postojećih test predikcija za zero-shot i First LoRA (Experiment 1) — provera poravnanja")
    zs_true, zs_pred, _ = load_prior_system("zero_shot", ZEROSHOT_TEST_CSV, test_df)
    e1_true, e1_pred, _ = load_prior_system("first_lora", EXP1_TEST_CSV, test_df)

    zs_valid, zs_e2e = compute_metrics_both(zs_true, zs_pred)
    e1_valid, e1_e2e = compute_metrics_both(e1_true, e1_pred)

    all_true = [t == final_true[i] for i, t in enumerate(zs_true)]
    assert all(all_true), "zero_shot final_label se ne poklapa sa test_df final_label na nekom redu."
    all_true2 = [t == final_true[i] for i, t in enumerate(e1_true)]
    assert all(all_true2), "first_lora final_label se ne poklapa sa test_df final_label na nekom redu."
    print("    [OK] Sva tri sistema koriste identičnih 227 row_id vrednosti i istu final_label kolonu.")

    # ------------------------------------------------------------------
    # 10. Comparison CSV (6 rows: 3 systems x 2 metric groups)
    # ------------------------------------------------------------------
    comp_rows = []
    for sys_key, valid_m, e2e_m in [
        ("zero_shot", zs_valid, zs_e2e),
        ("first_lora", e1_valid, e1_e2e),
        ("final_lora", final_valid_metrics, final_e2e_metrics),
    ]:
        for group, m in [("valid_only", valid_m), ("end_to_end", e2e_m)]:
            comp_rows.append({
                "system": SYSTEM_NAMES[sys_key], "metric_group": group,
                "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
                "accuracy": m["accuracy"], "specificity": m["specificity"],
                "fpr": m["fpr"], "fnr": m["fnr"], "balanced_accuracy": m["balanced_accuracy"],
                "mcc": m["mcc"], "invalid_count": m["invalid_count"], "invalid_rate": m["invalid_rate"],
            })
    comparison_df = pd.DataFrame(comp_rows)
    comparison_df.to_csv(OUT_DIR / "comparison_zero_shot_first_lora_final_lora.csv", index=False)

    # ------------------------------------------------------------------
    # 11. Terminal output
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("[A] PROVERA ADAPTER HASH-A")
    print("=" * 100)
    for name in sorted(hashes_before):
        print(f"    {name}: {hashes_before[name]}  [{'IDENTIČAN posle evaluacije' if hash_unchanged else 'PROMENJEN'}]")

    print("\n" + "=" * 100)
    print("[B] LOKACIJA SAČUVANIH REZULTATA")
    print("=" * 100)
    print(f"    {OUT_DIR}")

    print("\n" + "=" * 100)
    print("[C] VALID-ONLY METRIKE (precision/recall/F1 samo nad validnim predikcijama)")
    print("=" * 100)
    rows = []
    for sys_key, m in [("zero_shot", zs_valid), ("first_lora", e1_valid), ("final_lora", final_valid_metrics)]:
        rows.append([SYSTEM_NAMES[sys_key], f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}",
                     m["tp"], m["fp"], m["fn"], m["tn"], f"{m['accuracy']:.4f}", f"{m['mcc']:.4f}",
                     m["invalid_count"], fmt_pct(m["invalid_rate"])])
    print_table(rows, ["System", "Precision", "Recall", "F1", "TP", "FP", "FN", "TN",
                        "Accuracy", "MCC", "Invalid#", "InvalidRate"])

    print("\n" + "=" * 100)
    print("[D] END-TO-END METRIKE (invalid uvek greška)")
    print("=" * 100)
    rows = []
    for sys_key, m in [("zero_shot", zs_e2e), ("first_lora", e1_e2e), ("final_lora", final_e2e_metrics)]:
        rows.append([SYSTEM_NAMES[sys_key], f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}",
                     f"{m['fpr']:.4f}", f"{m['fnr']:.4f}", f"{m['accuracy']:.4f}", fmt_pct(m["invalid_rate"])])
    print_table(rows, ["System", "Precision", "Harmful recall", "F1", "FPR", "FNR", "Accuracy", "Invalid rate"])

    print("\n" + "=" * 100)
    print("[E] RAZLIKA: Final LoRA vs. Zero-shot v2  (end-to-end metrike)")
    print("=" * 100)
    d = {k: final_e2e_metrics[k] - zs_e2e[k] for k in
         ["f1", "recall", "fpr", "fnr", "accuracy", "invalid_rate"]}
    print(f"    F1 delta:            {d['f1']:+.4f}")
    print(f"    Harmful recall delta:{d['recall']:+.4f}")
    print(f"    FPR delta:           {d['fpr']:+.4f}")
    print(f"    FNR delta:           {d['fnr']:+.4f}")
    print(f"    Accuracy delta:      {d['accuracy']:+.4f}")
    print(f"    Invalid-rate delta:  {d['invalid_rate']:+.4f}")

    print("\n" + "=" * 100)
    print("[F] RAZLIKA: Final LoRA vs. First LoRA (Experiment 1, seed=42)  (end-to-end metrike)")
    print("=" * 100)
    d2 = {k: final_e2e_metrics[k] - e1_e2e[k] for k in
          ["f1", "recall", "fpr", "fnr", "accuracy", "invalid_rate"]}
    print(f"    F1 delta:            {d2['f1']:+.4f}")
    print(f"    Harmful recall delta:{d2['recall']:+.4f}")
    print(f"    FPR delta:           {d2['fpr']:+.4f}")
    print(f"    FNR delta:           {d2['fnr']:+.4f}")
    print(f"    Accuracy delta:      {d2['accuracy']:+.4f}")
    print(f"    Invalid-rate delta:  {d2['invalid_rate']:+.4f}")

    # ------------------------------------------------------------------
    # 12. REPORT.md
    # ------------------------------------------------------------------
    lines = []
    lines.append("# Konačna test evaluacija — zaključani all-linear LoRA (seed=42)\n")
    lines.append(
        "Ovo je **jednokratna, konačna** evaluacija zaključane all-linear LoRA konfiguracije na "
        "held-out test skupu. Test skup nije korišćen ni u jednom prethodnom koraku (Phase 1, Phase 2, "
        "multi-seed sweep) za izbor hiperparametara, konfiguracije, ili seed-a.\n"
    )

    lines.append("## Zaključana konfiguracija\n")
    lines.append("```json")
    lines.append(json.dumps({k: v for k, v in locked_config.items() if k not in
                             ("multiseed_validation_context",)}, indent=2, ensure_ascii=False))
    lines.append("```\n")
    if multiseed_context:
        vs_b = (f" (vs. config B mean F1 = {multiseed_context_b['mean_f1']:.4f} ± "
                f"{multiseed_context_b['std_f1']:.4f})") if multiseed_context_b else ""
        lines.append(
            "**Konfiguracija je izabrana na osnovu multi-seed VALIDATION rezultata** "
            "(`scripts/model/results/gemma_lora_v2_multiseed/REPORT.md`): config A "
            f"(`lr=3e-4, r=8, alpha=16, dropout=0.0`) je stabilan pobednik preko seedova "
            f"{{{multiseed_context['seeds']}}} sa mean F1 = {multiseed_context['mean_f1']:.4f} ± "
            f"{multiseed_context['std_f1']:.4f}{vs_b}."
        )
    else:
        lines.append("**Konfiguracija je izabrana na osnovu multi-seed VALIDATION rezultata.**")
    lines.append(
        "\n**Za konačnu test evaluaciju unapred je izabran isključivo `seed=42`** — razlog je naučno "
        "uporedivo poređenje sa Experiment 1 (First LoRA), koji je takođe treniran sa `seed=42`. "
        "**Seedovi 23 i 41 NISU evaluirani na test skupu** — ne postoji nikakav test rezultat za njih ni "
        "u ovom folderu ni bilo gde drugde u projektu.\n"
    )

    lines.append("## Validation rezultat seed=42 adaptera (Phase 2)\n")
    bm = run_summary["best_metrics"]
    lines.append(
        f"- best_epoch = {run_summary['best_epoch']} (od {run_summary['last_completed_epoch']} "
        f"završenih epoha, {run_summary['stop_reason']})\n"
        f"- precision = {bm['precision']:.4f}, recall = {bm['recall']:.4f}, F1 = {bm['f1']:.4f}, "
        f"invalid_rate = {bm['invalid_rate']:.4f}\n"
        f"- Adapter: `{ADAPTER_DIR}`\n"
    )

    lines.append("## Adapter hash provera\n")
    lines.append("SHA256 hash svakog adapter fajla, pre i posle test evaluacije — **identičan**, "
                  "što potvrđuje da adapter nije promenjen tokom evaluacije.\n")
    lines.append("```")
    for name in sorted(hashes_before):
        lines.append(f"{name}: {hashes_before[name]}")
    lines.append("```\n")

    lines.append("## Test skup\n")
    lines.append(
        f"`{TEST_PATH}` — {len(test_df)} redova, {n_groups} original_idx grupa, row_id jedinstveni.\n\n"
        f"Distribucija `final_label`: harmful={label_dist.get('harmful', 0)}, "
        f"unharmful={label_dist.get('unharmful', 0)}.\n\n"
        f"Tokenizacija (identičan pipeline kao validation): {n_truncated}/{len(test_df)} primera skraćeno "
        f"({n_truncated / len(test_df) * 100:.2f}%).\n"
    )

    lines.append("## A. Test metrike — samo nad validnim predikcijama\n")
    lines.append("```")
    lines.append(json.dumps(final_valid_metrics, indent=2))
    lines.append("```\n")

    lines.append("## B. Test metrike — end-to-end (invalid uvek pogrešan)\n")
    lines.append("```")
    lines.append(json.dumps(final_e2e_metrics, indent=2))
    lines.append("```\n")

    lines.append("## Confusion matrix (test, nad validnim predikcijama)\n")
    lines.append("```")
    lines.append(cm.to_string())
    lines.append("```\n")

    lines.append("## Poređenje tri sistema (end-to-end metrike)\n")
    lines.append(
        "Pre poređenja potvrđeno: sva tri sistema koriste identičnih 227 `row_id` vrednosti i istu "
        "`final_label` kolonu iz istog v2 test skupa (provera izvršena programski, vidi konzolni izlaz "
        "skripte `[8]`).\n\n"
        "**\"First LoRA\" = Experiment 1 (prvi pravi fine-tuned LoRA model, seed=42), NE raniji sanity "
        "pilot.**\n"
    )
    e2e_table = comparison_df[comparison_df["metric_group"] == "end_to_end"][
        ["system", "precision", "recall", "f1", "fpr", "fnr", "accuracy", "invalid_rate"]
    ]
    lines.append("```")
    lines.append(e2e_table.to_string(index=False))
    lines.append("```\n")

    lines.append("### Razlika: Final LoRA vs. Zero-shot v2 (end-to-end)\n")
    lines.append(f"- F1 delta: {d['f1']:+.4f}\n- Harmful recall delta: {d['recall']:+.4f}\n"
                  f"- FPR delta: {d['fpr']:+.4f}\n- FNR delta: {d['fnr']:+.4f}\n"
                  f"- Accuracy delta: {d['accuracy']:+.4f}\n- Invalid-rate delta: {d['invalid_rate']:+.4f}\n")

    lines.append("### Razlika: Final LoRA vs. First LoRA (Experiment 1, seed=42) (end-to-end)\n")
    lines.append(f"- F1 delta: {d2['f1']:+.4f}\n- Harmful recall delta: {d2['recall']:+.4f}\n"
                  f"- FPR delta: {d2['fpr']:+.4f}\n- FNR delta: {d2['fnr']:+.4f}\n"
                  f"- Accuracy delta: {d2['accuracy']:+.4f}\n- Invalid-rate delta: {d2['invalid_rate']:+.4f}\n")

    lines.append("## Napomene / ograničenja ovog zadatka\n")
    lines.append(
        "- Test skup NIJE korišćen za izbor konfiguracije, seed-a, ili bilo kog hiperparametra — svi "
        "izbori (config A, seed=42, best_epoch=6) su zaključani isključivo na osnovu validation rezultata "
        "PRE nego što je ovaj skript prvi put pročitao `test.jsonl`.\n"
        "- Seedovi 23 i 41 (multi-seed sweep) NISU evaluirani na test skupu — evaluiran je isključivo "
        "`seed=42`, radi uporedivosti sa Experiment 1.\n"
        "- Nije pravljen ensemble niti majority vote preko seedova.\n"
        "- Attention-only, DoRA i QLoRA eksperimenti NISU deo ovog zadatka.\n"
        "- Validation evaluacija NIJE ponovo pokretana u ovom skriptu — koriste se isključivo postojeći "
        "Phase 2 validation rezultati.\n"
        "- Raniji rezultati (Phase 1, Phase 2, multiseed, zero-shot, Experiment 1) nisu menjani ni brisani.\n"
    )

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))

    print("\n" + "=" * 100)
    print(f"[OK] Svi rezultati sačuvani u: {OUT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()
