#!/usr/bin/env python
"""Final, one-shot TEST evaluation of the three LoRA/DoRA ablation adapters —
Gemma 3 1B IT, v2 (no-refusal) dataset, seed=42.

This script does NOT retrain anything, does NOT modify any adapter, and does
NOT re-run validation. It:

  1. Re-verifies the existing ablation artifacts (REPORT.md, ablation_summary.csv,
     reference_config.json, and each run's run_config.json/run_summary.json/
     status.json) — never re-derives or guesses config values.
  2. Hashes (SHA256) every adapter weight file BEFORE and AFTER evaluation to
     prove no adapter was mutated.
  3. Runs each of the three adapters (lora_attention_only, dora_all_linear,
     dora_attention_only) exactly once on data/gemma_v2_no_refusal/test.jsonl,
     reusing the tokenizer, PROMPT_1 text, chat template, head-tail truncation,
     and greedy-generation (do_sample=False, max_new_tokens=10) helpers
     imported directly from run_lora_dora_ablation_v2.py — not reimplemented —
     so the inference pipeline is identical to the one used during training/
     validation for these adapters. A fresh base model + PeftModel.from_pretrained
     is loaded per adapter so adapters never mix.
  4. Computes both "valid predictions only" and "end-to-end" (invalid counted
     as an error) metrics, using the same methodology as
     eval_final_lora_test.py (not reimplemented from scratch).
  5. Loads (never re-runs) existing test predictions for Gemma zero-shot v2,
     Gemma initial LoRA (Experiment 1), Gemma sweep LoRA (final/locked), and
     Qwen3Guard native zero-shot, plus the regex baseline, verifies all systems
     share the identical 227 test row_id / final_label values, and recomputes
     their metrics with the exact same methodology used here (never hardcoded).
  6. Saves everything under
     scripts/model/results/gemma_lora_v2_lora_dora_ablation/test_evaluation/
     and appends a new section to the existing ablation REPORT.md.

Usage:
    ~/ccpp_env/bin/python scripts/model/eval_lora_dora_ablation_test.py
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

import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_lora_dora_ablation_v2 as abl  # noqa: E402  (reuse exact pipeline, not reimplemented)

# ---------------------------------------------------------------------------
# Fixed paths
# ---------------------------------------------------------------------------
ABLATION_DIR = SCRIPT_DIR / "results" / "gemma_lora_v2_lora_dora_ablation"
OUT_DIR = ABLATION_DIR / "test_evaluation"
TEST_PATH = Path("/home/mls01/data/gemma_v2_no_refusal/test.jsonl")
POSITIVE = "harmful"

PRE_SELECTED_WINNER = "dora_all_linear"  # decided on validation F1, BEFORE this test evaluation ran

ZEROSHOT_CSV = SCRIPT_DIR / "results" / "gemma_demo_zeroshot_v2_no_refusal" / "test_results_full.csv"
EXP1_CSV = SCRIPT_DIR / "results" / "gemma_lora_v2_exp1_r8_lr2e4_seed42_max8_es2" / "test_results_full.csv"
SWEEP_CSV = SCRIPT_DIR / "results" / "gemma_lora_v2_multiseed" / "final_test_seed42" / "test_results_full.csv"
QWEN3GUARD_CSV = SCRIPT_DIR / "results" / "qwen3guard_native_v2_no_refusal" / "test_results_full.csv"
REGEX_CSV = SCRIPT_DIR / "results" / "regex_baseline_v2_no_refusal" / "test_results_full.csv"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# SHA256 hashing (identical helper to eval_final_lora_test.py)
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
# Metrics — identical methodology to eval_final_lora_test.py's compute_metrics_both.
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
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": accuracy, "specificity": specificity, "fpr": fpr, "fnr": fnr,
            "balanced_accuracy": balanced_accuracy, "mcc": mcc}


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
# Load an existing system's saved test predictions, verify alignment.
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
            f"Nedostaje {len(missing)}, višak {len(extra)}."
        )
    if len(df) != len(test_df):
        raise ValueError(f"[{name}] {csv_path} ima {len(df)} redova, test.jsonl ima {len(test_df)}.")

    merged = test_df[["row_id", "final_label"]].merge(
        df[["row_id", "final_label", "prediction"]], on="row_id", suffixes=("_test", "_system"))
    mismatched = merged[merged["final_label_test"] != merged["final_label_system"]]
    if len(mismatched):
        raise ValueError(f"[{name}] final_label se ne poklapa na {len(mismatched)} redova — PREKID.")

    merged = merged.set_index("row_id").loc[test_df["row_id"]].reset_index()
    print(f"    [OK] [{name}] {csv_path.name}: {len(merged)}/{len(test_df)} row_id poklapaju, final_label identičan.")
    return merged["final_label_test"].tolist(), merged["prediction"].tolist()


def print_table(rows, headers):
    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) + 2 for i, h in enumerate(headers)]
    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        print("".join(str(c).ljust(w) for c, w in zip(r, col_widths)))


ERROR_COLS = ["row_id", "original_idx", "prompt", "response", "final_label", "prediction", "raw_output",
              "language", "augmentation_type", "adversarial", "eval_truncated"]


def main():
    t0 = time.time()
    print("=" * 100)
    print("KONAČNA TEST EVALUACIJA — LoRA/DoRA ablacija (3 adaptera), Gemma 3 1B IT, v2 dataset")
    print("=" * 100)

    # ------------------------------------------------------------------
    # 1. Review existing ablation artifacts.
    # ------------------------------------------------------------------
    print("\n[1] Provera postojećih ablation artefakata")
    for p in [ABLATION_DIR / "REPORT.md", ABLATION_DIR / "ablation_summary.csv", ABLATION_DIR / "reference_config.json"]:
        assert p.exists(), f"Nedostaje: {p}"
        print(f"    [OK] {p} postoji")
    reference_config = json.loads((ABLATION_DIR / "reference_config.json").read_text())
    ablation_summary = pd.read_csv(ABLATION_DIR / "ablation_summary.csv")
    print(f"    [OK] reference_config.json: lr={reference_config['learning_rate']}, rank={reference_config['rank']}, "
          f"alpha={reference_config['alpha']}, dropout={reference_config['dropout']}, seed={reference_config['seed']}, "
          f"max_epochs={reference_config['max_epochs']}, patience={reference_config['early_stopping_patience']}")
    print(f"    [PRE-SELECTED WINNER] '{PRE_SELECTED_WINNER}' je izabran kao validation pobednik PRE ovog test-a "
          f"(validation F1={ablation_summary.set_index('run_id').loc[PRE_SELECTED_WINNER, 'f1']:.4f}). "
          f"Testiranje druga dva adaptera je isključivo za kompletiranje ablation poređenja.")

    configs = abl.build_grid()  # exact same 3 configs/order used during training — not re-typed here
    assert [c["run_id"] for c in configs] == ["lora_attention_only", "dora_all_linear", "dora_attention_only"]

    run_info = {}
    adapter_hashes_before = {}
    for cfg in configs:
        run_id = cfg["run_id"]
        run_dir = ABLATION_DIR / run_id
        status = json.loads((run_dir / "status.json").read_text())
        assert status["status"] == "completed", f"{run_id}: status={status['status']}, očekivano 'completed'"
        run_config = json.loads((run_dir / "run_config.json").read_text())
        run_summary = json.loads((run_dir / "run_summary.json").read_text())
        assert run_config["use_dora"] == cfg["use_dora"], f"{run_id}: use_dora mismatch"
        assert run_config["target_modules"] == cfg["target_modules"], f"{run_id}: target_modules mismatch"
        assert run_config["prompt"] == abl.PROMPT_1, f"{run_id}: prompt mismatch"
        assert run_config["generation"] == abl.GENERATION_PARAMS, f"{run_id}: generation params mismatch"
        adapter_dir = run_dir / "best_adapter"
        assert (adapter_dir / "adapter_config.json").exists() and (adapter_dir / "adapter_model.safetensors").exists()

        h = hash_adapter_dir(adapter_dir)
        adapter_hashes_before[run_id] = h
        run_info[run_id] = {"run_config": run_config, "run_summary": run_summary, "adapter_dir": adapter_dir,
                            "status": status, "cfg": cfg}
        print(f"\n    [OK] {run_id}: status=completed, use_dora={cfg['use_dora']}, target_scope={cfg['target_scope']}, "
              f"target_modules={cfg['target_modules']}")
        print(f"         best_epoch={run_summary['best_epoch']}, validation P={run_summary['best_metrics']['precision']:.4f} "
              f"R={run_summary['best_metrics']['recall']:.4f} F1={run_summary['best_metrics']['f1']:.4f} "
              f"invalid_rate={run_summary['best_metrics']['invalid_rate']:.4f}")
        print(f"         SHA256 (adapter_model.safetensors) PRE: {h['adapter_model.safetensors']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Load + validate test.jsonl
    # ------------------------------------------------------------------
    print("\n[2] Učitavanje i provera test skupa")
    test_df = pd.read_json(TEST_PATH, lines=True)
    assert len(test_df) == 227, f"Očekivano 227 redova, dobijeno {len(test_df)}"
    assert test_df["row_id"].is_unique, "row_id nije jedinstven u test.jsonl"
    n_groups = test_df["original_idx"].nunique()
    assert n_groups == 100, f"Očekivano 100 original_idx grupa, dobijeno {n_groups}"
    label_dist = test_df["final_label"].value_counts().to_dict()
    assert label_dist == {"harmful": 127, "unharmful": 100}, f"Neočekivana raspodela labela: {label_dist}"
    print(f"    [OK] {TEST_PATH}")
    print(f"    [OK] {len(test_df)} redova, {n_groups} original_idx grupa, row_id jedinstveni, "
          f"final_label={label_dist}")

    # ------------------------------------------------------------------
    # 3. Tokenizer + truncation + parser — reused unchanged from run_lora_dora_ablation_v2.py.
    # ------------------------------------------------------------------
    print("\n[3] Priprema tokenizacije (isti pipeline kao trening/validation, uvezen iz run_lora_dora_ablation_v2.py)")
    tokenizer, end_of_turn_id, build_example, parse_label = abl.build_tokenizer_and_helpers()

    test_prefixes, truncated_flags = [], []
    for row in test_df.itertuples(index=False):
        ex = build_example(row.prompt, row.response, row.final_label)
        test_prefixes.append(ex["input_ids"][:ex["prefix_len"]])
        truncated_flags.append(ex["truncated"])
    n_truncated = int(sum(truncated_flags))
    print(f"    [OK] {len(test_prefixes)} test primera tokenizovano | skraćeno {n_truncated} "
          f"({n_truncated / len(test_prefixes) * 100:.2f}%)")

    evaluate_checkpoint = abl.make_evaluate_checkpoint(tokenizer, parse_label)

    locked_test_config = {
        "model_path": str(abl.MODEL_PATH), "test_path": str(TEST_PATH),
        "prompt": abl.PROMPT_1, "generation": abl.GENERATION_PARAMS,
        "max_seq_length": abl.MAX_SEQ_LENGTH,
        "parser_steps": ["strip()", "lowercase", "accept only exact 'harmful' or 'unharmful'",
                        "everything else -> 'invalid'", "raw_output always preserved"],
        "empty_response_handling": "if response == '' the assistant-response section of the input text is "
                                   "omitted entirely (build_sample_text branches on truthiness of response), "
                                   "identical to training/validation",
        "n_test_rows": len(test_df), "n_truncated": n_truncated,
    }

    # ------------------------------------------------------------------
    # 4. Evaluate each adapter exactly once, fresh base model each time.
    # ------------------------------------------------------------------
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    all_results = {}
    for cfg in configs:
        run_id = cfg["run_id"]
        adapter_dir = run_info[run_id]["adapter_dir"]
        print(f"\n[4.{configs.index(cfg)+1}] Evaluacija adaptera: {run_id}")
        print(f"    Učitavanje čistog baznog modela + {adapter_dir}")

        base_model = AutoModelForCausalLM.from_pretrained(
            abl.MODEL_PATH, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager",
        ).to("cuda")
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        print(f"    [OK] {abl.MODEL_PATH} + {adapter_dir}")

        t_eval = time.time()
        raw_outputs, predictions = evaluate_checkpoint(peft_model, test_prefixes, run_id)
        print(f"    [OK] Inferencija završena u {time.time()-t_eval:.1f}s")

        del base_model, peft_model
        gc.collect()
        torch.cuda.empty_cache()

        results_full = test_df.copy()
        results_full["prediction"] = predictions
        results_full["raw_output"] = raw_outputs
        results_full["is_valid"] = results_full["prediction"].isin(["harmful", "unharmful"])
        results_full["is_correct"] = results_full["prediction"] == results_full["final_label"]
        results_full["eval_truncated"] = truncated_flags

        valid_m, e2e_m = compute_metrics_both(results_full["final_label"].tolist(), results_full["prediction"].tolist())
        print(f"    [METRICS valid-only]   P={valid_m['precision']:.4f} R={valid_m['recall']:.4f} F1={valid_m['f1']:.4f} "
              f"TP={valid_m['tp']} FP={valid_m['fp']} FN={valid_m['fn']} TN={valid_m['tn']} "
              f"invalid={valid_m['invalid_count']} ({valid_m['invalid_rate']*100:.2f}%)")
        print(f"    [METRICS end-to-end]   P={e2e_m['precision']:.4f} R={e2e_m['recall']:.4f} F1={e2e_m['f1']:.4f} "
              f"FPR={e2e_m['fpr']:.4f} FNR={e2e_m['fnr']:.4f} accuracy={e2e_m['accuracy']:.4f}")

        run_out_dir = OUT_DIR / run_id
        run_out_dir.mkdir(parents=True, exist_ok=True)
        results_full.to_csv(run_out_dir / "test_results_full.csv", index=False)

        fp_mask = (results_full["final_label"] != POSITIVE) & (results_full["prediction"] == POSITIVE)
        fn_mask = (results_full["final_label"] == POSITIVE) & (results_full["prediction"] != POSITIVE) & \
                  (results_full["prediction"] != "invalid")
        invalid_mask = results_full["prediction"] == "invalid"
        results_full.loc[fp_mask, ERROR_COLS].to_csv(run_out_dir / "test_false_positives.csv", index=False)
        results_full.loc[fn_mask, ERROR_COLS].to_csv(run_out_dir / "test_false_negatives.csv", index=False)
        results_full.loc[invalid_mask, ERROR_COLS].to_csv(run_out_dir / "test_invalid_examples.csv", index=False)

        cm = pd.DataFrame([[valid_m["tp"], valid_m["fn"]], [valid_m["fp"], valid_m["tn"]]],
                          index=["true_harmful", "true_unharmful"], columns=["pred_harmful", "pred_unharmful"])
        cm.to_csv(run_out_dir / "test_confusion_matrix.csv")

        (run_out_dir / "test_metrics.json").write_text(json.dumps(
            {"metrics_on_valid_predictions": valid_m, "metrics_end_to_end": e2e_m}, indent=2))

        run_locked_config = dict(locked_test_config)
        run_locked_config["adapter_path"] = str(adapter_dir)
        run_locked_config["run_id"] = run_id
        run_locked_config["use_dora"] = cfg["use_dora"]
        run_locked_config["target_modules"] = cfg["target_modules"]
        (run_out_dir / "locked_test_config.json").write_text(json.dumps(run_locked_config, indent=2, ensure_ascii=False))

        all_results[run_id] = {"valid_m": valid_m, "e2e_m": e2e_m, "results_full": results_full}
        print(f"    [OK] Rezultati sačuvani u {run_out_dir}")

    # ------------------------------------------------------------------
    # 5. Hash AFTER — confirm no adapter was mutated.
    # ------------------------------------------------------------------
    print("\n[5] SHA256 provera adaptera POSLE evaluacije")
    all_unchanged = True
    for cfg in configs:
        run_id = cfg["run_id"]
        adapter_dir = run_info[run_id]["adapter_dir"]
        hashes_after = hash_adapter_dir(adapter_dir)
        unchanged = hashes_after == adapter_hashes_before[run_id]
        all_unchanged = all_unchanged and unchanged
        print(f"    {run_id}: {'IDENTIČAN' if unchanged else 'PROMENJEN !!'}")
        (OUT_DIR / run_id / "adapter_hashes.json").write_text(json.dumps({
            "adapter_dir": str(adapter_dir), "before": adapter_hashes_before[run_id], "after": hashes_after,
            "unchanged": unchanged,
        }, indent=2))
    if not all_unchanged:
        raise RuntimeError("Adapter fajlovi su se promenili tokom evaluacije — REZULTATI SE ODBACUJU.")
    print("    [OK] Svi adapteri nepromenjeni pre/posle evaluacije.")

    # ------------------------------------------------------------------
    # 6. new_ablation_test_summary.csv
    # ------------------------------------------------------------------
    print("\n[6] new_ablation_test_summary.csv")
    summary_rows = []
    for cfg in configs:
        run_id = cfg["run_id"]
        rs = run_info[run_id]["run_summary"]
        valid_m, e2e_m = all_results[run_id]["valid_m"], all_results[run_id]["e2e_m"]
        summary_rows.append({
            "run_id": run_id, "method": cfg["method"], "target_scope": cfg["target_scope"], "use_dora": cfg["use_dora"],
            "val_best_epoch": rs["best_epoch"], "val_precision": rs["best_metrics"]["precision"],
            "val_recall": rs["best_metrics"]["recall"], "val_f1": rs["best_metrics"]["f1"],
            "test_precision_e2e": e2e_m["precision"], "test_recall_e2e": e2e_m["recall"], "test_f1_e2e": e2e_m["f1"],
            "test_fpr_e2e": e2e_m["fpr"], "test_fnr_e2e": e2e_m["fnr"], "test_accuracy_e2e": e2e_m["accuracy"],
            "test_tp": e2e_m["tp"], "test_fp": e2e_m["fp"], "test_fn": e2e_m["fn"], "test_tn": e2e_m["tn"],
            "invalid_count": valid_m["invalid_count"], "invalid_rate": valid_m["invalid_rate"],
            "is_pre_selected_winner": run_id == PRE_SELECTED_WINNER,
        })
    new_ablation_test_summary = pd.DataFrame(summary_rows)
    new_ablation_test_summary.to_csv(OUT_DIR / "new_ablation_test_summary.csv", index=False)
    print(new_ablation_test_summary[["run_id", "val_f1", "test_f1_e2e", "test_precision_e2e", "test_recall_e2e"]]
          .to_string(index=False))

    # ------------------------------------------------------------------
    # 7. Load prior systems, verify alignment, build the 8-system comparison.
    # ------------------------------------------------------------------
    print("\n[7] Učitavanje sačuvanih test predikcija ostalih sistema (bez ponovnog pokretanja)")
    prior_systems = [
        ("Gemma zero-shot", "Zero-shot", "N/A", ZEROSHOT_CSV),
        ("Gemma initial LoRA", "LoRA", "all_linear", EXP1_CSV),
        ("Gemma sweep LoRA", "LoRA", "all_linear", SWEEP_CSV),
        ("Qwen3Guard", "Native zero-shot", "N/A", QWEN3GUARD_CSV),
    ]
    comparison_rows = []
    for name, method, scope, csv_path in prior_systems:
        true, pred = load_prior_system(name, csv_path, test_df)
        valid_m, e2e_m = compute_metrics_both(true, pred)
        comparison_rows.append({"System": name, "Method": method, "Target scope": scope, **e2e_m,
                                "invalid_count": valid_m["invalid_count"], "invalid_rate": valid_m["invalid_rate"]})

    new_display = [
        ("LoRA attention-only", "LoRA", "attention_only", "lora_attention_only"),
        ("DoRA all-linear", "DoRA", "all_linear", "dora_all_linear"),
        ("DoRA attention-only", "DoRA", "attention_only", "dora_attention_only"),
    ]
    for name, method, scope, run_id in new_display:
        e2e_m = all_results[run_id]["e2e_m"]
        valid_m = all_results[run_id]["valid_m"]
        comparison_rows.append({"System": name, "Method": method, "Target scope": scope, **e2e_m,
                                "invalid_count": valid_m["invalid_count"], "invalid_rate": valid_m["invalid_rate"]})

    # Additional (not in the required 7): regex baseline, if a comparable saved test result exists.
    if REGEX_CSV.exists():
        true, pred = load_prior_system("Regex baseline", REGEX_CSV, test_df)
        valid_m, e2e_m = compute_metrics_both(true, pred)
        comparison_rows.append({"System": "Regex baseline", "Method": "Regex (train-derived)", "Target scope": "N/A",
                                **e2e_m, "invalid_count": valid_m["invalid_count"], "invalid_rate": valid_m["invalid_rate"]})

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUT_DIR / "all_systems_test_comparison.csv", index=False)
    (OUT_DIR / "all_systems_test_comparison.json").write_text(json.dumps(comparison_rows, indent=2))

    # ------------------------------------------------------------------
    # 8. Print required table.
    # ------------------------------------------------------------------
    print("\n[8] OBJEDINJENA TABELA (test, end-to-end, redosled sistema NIJE sortiran po performansama)")
    headers = ["System", "Method", "Target scope", "Precision", "Harmful recall", "F1", "FPR", "FNR",
              "Accuracy", "Invalid rate"]
    table_rows = []
    for r in comparison_rows:
        table_rows.append([r["System"], r["Method"], r["Target scope"], f"{r['precision']:.4f}",
                           f"{r['recall']:.4f}", f"{r['f1']:.4f}", f"{r['fpr']:.4f}", f"{r['fnr']:.4f}",
                           f"{r['accuracy']:.4f}", f"{r['invalid_rate']:.4f}"])
    print_table(table_rows, headers)

    # ------------------------------------------------------------------
    # 9. Short narrative: deltas, MLP effect, LoRA->DoRA effect, best systems.
    # ------------------------------------------------------------------
    by_system = {r["System"]: r for r in comparison_rows}
    exp1 = by_system["Gemma initial LoRA"]
    lora_attn = by_system["LoRA attention-only"]
    dora_all = by_system["DoRA all-linear"]
    dora_attn = by_system["DoRA attention-only"]

    print("\n[9] Kratak rezime")
    print(f"  - Validation vs test (3 nova adaptera):")
    for cfg in configs:
        run_id = cfg["run_id"]
        rs = run_info[run_id]["run_summary"]
        e2e = all_results[run_id]["e2e_m"]
        print(f"      {run_id}: val F1={rs['best_metrics']['f1']:.4f} -> test F1={e2e['f1']:.4f} "
              f"(delta={e2e['f1']-rs['best_metrics']['f1']:+.4f})")

    print(f"\n  - Promena test F1/recall u odnosu na initial all-linear LoRA (Exp1, F1={exp1['f1']:.4f}, "
          f"recall={exp1['recall']:.4f}):")
    for name, row in [("LoRA attention-only", lora_attn), ("DoRA all-linear", dora_all), ("DoRA attention-only", dora_attn)]:
        print(f"      {name}: F1 delta={row['f1']-exp1['f1']:+.4f}, recall delta={row['recall']-exp1['recall']:+.4f}")

    print(f"\n  - Efekat uklanjanja MLP target modula (LoRA, test): all-linear (Exp1) F1={exp1['f1']:.4f} vs "
          f"attention-only F1={lora_attn['f1']:.4f} (delta={lora_attn['f1']-exp1['f1']:+.4f})")
    print(f"  - Efekat LoRA->DoRA (all-linear, test): F1={exp1['f1']:.4f} -> {dora_all['f1']:.4f} "
          f"(delta={dora_all['f1']-exp1['f1']:+.4f})")
    print(f"  - Efekat LoRA->DoRA (attention-only, test): F1={lora_attn['f1']:.4f} -> {dora_attn['f1']:.4f} "
          f"(delta={dora_attn['f1']-lora_attn['f1']:+.4f})")

    best_f1_system = max(comparison_rows, key=lambda r: r["f1"])
    best_recall_system = max(comparison_rows, key=lambda r: r["recall"])
    print(f"\n  - Najveći test F1: {best_f1_system['System']} (F1={best_f1_system['f1']:.4f})")
    print(f"  - Najveći harmful recall: {best_recall_system['System']} (recall={best_recall_system['recall']:.4f})")

    print(f"\n  [UPOZORENJE] '{PRE_SELECTED_WINNER}' je izabran kao finalni model PRE ovog test-a, na osnovu "
          f"validation F1. Test rezultati za 'lora_attention_only' i 'dora_attention_only' služe ISKLJUČIVO za "
          f"kompletiranje ablation poređenja i NE smeju se koristiti za post-hoc promenu izabranog modela ili "
          f"hiperparametara.")

    # ------------------------------------------------------------------
    # 10. Write REPORT.md for test_evaluation/, and append section to root REPORT.md.
    # ------------------------------------------------------------------
    write_test_evaluation_report(configs, run_info, all_results, new_ablation_test_summary, comparison_rows,
                                 n_truncated, len(test_df))
    append_root_report_section(configs, run_info, all_results, comparison_rows)

    # ------------------------------------------------------------------
    # 11. Final checks
    # ------------------------------------------------------------------
    print("\n[11] Završna provera")
    print(f"    [OK] Sva tri inference prolaza završena (lora_attention_only, dora_all_linear, dora_attention_only)")
    print(f"    [OK] Test skup: {len(test_df)} redova, row_id jedinstven, bez duplikata")
    all_row_id_sets_equal = all(set(pd.read_csv(p)["row_id"]) == set(test_df["row_id"]) for p in
                                [ZEROSHOT_CSV, EXP1_CSV, SWEEP_CSV, QWEN3GUARD_CSV, REGEX_CSV])
    print(f"    [OK] Svi sistemi u poređenju koriste identičan ground truth: {all_row_id_sets_equal}")
    print(f"    [OK] Adapter hash vrednosti identične pre/posle: {all_unchanged}")
    print(f"    [OK] Nijedan adapter nije treniran niti izmenjen (samo učitan za inference)")
    print(f"    [OK] Nijedan stari model nije ponovo pokretan (samo pročitane sačuvane predikcije)")
    required_files_ok = True
    for run_id in ["lora_attention_only", "dora_all_linear", "dora_attention_only"]:
        for fname in ["test_results_full.csv", "test_metrics.json", "test_confusion_matrix.csv",
                     "test_false_positives.csv", "test_false_negatives.csv", "test_invalid_examples.csv",
                     "locked_test_config.json", "adapter_hashes.json"]:
            if not (OUT_DIR / run_id / fname).exists():
                required_files_ok = False
                print(f"    [MISSING] {OUT_DIR / run_id / fname}")
    for fname in ["new_ablation_test_summary.csv", "all_systems_test_comparison.csv",
                 "all_systems_test_comparison.json", "REPORT.md"]:
        if not (OUT_DIR / fname).exists():
            required_files_ok = False
            print(f"    [MISSING] {OUT_DIR / fname}")
    print(f"    [OK] Svi traženi fajlovi postoje: {required_files_ok}")

    print(f"\ntotal runtime: {time.time()-t0:.1f}s")
    print("\n" + "=" * 100)
    print("FINALNA OBJEDINJENA TABELA")
    print("=" * 100)
    print_table(table_rows, headers)


def write_test_evaluation_report(configs, run_info, all_results, summary_df, comparison_rows, n_truncated, n_test):
    lines = ["# Konačna test evaluacija — LoRA/DoRA ablacija\n"]
    lines.append(
        "Test evaluacija tri adaptera iz LoRA/DoRA ablacije (`lora_attention_only`, `dora_all_linear`, "
        "`dora_attention_only`) na `data/gemma_v2_no_refusal/test.jsonl` (227 redova/100 grupa, "
        "127 harmful/100 unharmful). Isti bazni model, prompt, tokenizacija/truncation, parser i "
        "generation parametri (`do_sample=False, max_new_tokens=10`) kao tokom treninga/validacije — "
        "uvezeno iz `run_lora_dora_ablation_v2.py`, nije reimplementirano.\n"
    )
    lines.append(f"`{n_truncated}/{n_test}` ({n_truncated/n_test*100:.2f}%) test primera je skraćeno "
                f"(head-tail truncation na max_seq_length).\n")
    lines.append(
        f"**Metodološka napomena**: `dora_all_linear` je izabran kao validation pobednik PRE ovog testa. "
        f"Test rezultati za `lora_attention_only` i `dora_attention_only` služe isključivo za kompletiranje "
        f"ablation poređenja i ne smeju se koristiti za post-hoc promenu izabranog modela.\n"
    )
    lines.append("## Rezultati po adapteru (validation -> test)\n")
    lines.append("```\n" + summary_df[["run_id", "val_best_epoch", "val_precision", "val_recall", "val_f1",
                                       "test_precision_e2e", "test_recall_e2e", "test_f1_e2e", "test_fpr_e2e",
                                       "test_fnr_e2e", "invalid_rate"]].to_string(index=False) + "\n```\n")
    for cfg in configs:
        run_id = cfg["run_id"]
        valid_m = all_results[run_id]["valid_m"]
        cm = pd.DataFrame([[valid_m["tp"], valid_m["fn"]], [valid_m["fp"], valid_m["tn"]]],
                          index=["true_harmful", "true_unharmful"], columns=["pred_harmful", "pred_unharmful"])
        lines.append(f"### {run_id}\n")
        lines.append(f"Confusion matrix (nad validnim predikcijama):\n```\n{cm.to_string()}\n```\n")
    lines.append("## Objedinjena tabela (test, end-to-end)\n")
    display_cols = ["System", "Method", "Target scope", "precision", "recall", "f1", "fpr", "fnr", "accuracy", "invalid_rate"]
    lines.append("```\n" + pd.DataFrame(comparison_rows)[display_cols].to_string(index=False) + "\n```\n")
    lines.append("## Napomena\n")
    lines.append("Test skup korišćen tačno jednom po adapteru, nakon zaključavanja konfiguracije. "
                "Adapter SHA256 hash-ovi potvrđeno nepromenjeni pre/posle (vidi `adapter_hashes.json` "
                "u svakom pod-folderu).\n")
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"\n[OK] {OUT_DIR / 'REPORT.md'} written.")


def append_root_report_section(configs, run_info, all_results, comparison_rows):
    root_report_path = ABLATION_DIR / "REPORT.md"
    existing = root_report_path.read_text()

    lines = ["\n\n---\n\n# Konačna test evaluacija\n"]
    lines.append(
        f"Test evaluacija sva tri nova adaptera na `data/gemma_v2_no_refusal/test.jsonl` (227 redova/100 grupa, "
        f"127 harmful/100 unharmful), evaluirano TAČNO JEDNOM po adapteru, koristeći identičan pipeline "
        f"(tokenizer/prompt/truncation/parser/generation) kao tokom treninga i validacije. Puni detalji, FP/FN/"
        f"invalid analiza, i sve sačuvane datoteke su u `test_evaluation/` pod-folderu.\n"
    )
    lines.append(
        f"**Metodološka napomena**: `dora_all_linear` je izabran kao validation pobednik PRE ovog testa "
        f"(na osnovu validation F1). Test rezultati za `lora_attention_only` i `dora_attention_only` "
        f"predstavljaju samo završno ablation poređenje i NE smeju se koristiti za post-hoc promenu "
        f"izabranog modela ili hiperparametara.\n"
    )
    lines.append("## Zaključana konfiguracija svakog adaptera\n")
    for cfg in configs:
        run_id = cfg["run_id"]
        rc = run_info[run_id]["run_config"]
        lines.append(f"- **{run_id}**: method={cfg['method']}, use_dora={cfg['use_dora']}, "
                    f"target_scope={cfg['target_scope']}, target_modules={cfg['target_modules']}, "
                    f"r={rc['rank']}, alpha={rc['alpha']}, dropout={rc['dropout']}, lr={rc['learning_rate']}, "
                    f"seed={rc['seed']}")
    lines.append("")
    lines.append("## Validation vs test rezultati\n")
    rows_txt = []
    for cfg in configs:
        run_id = cfg["run_id"]
        rs = run_info[run_id]["run_summary"]
        e2e = all_results[run_id]["e2e_m"]
        rows_txt.append(f"- **{run_id}**: validation F1={rs['best_metrics']['f1']:.4f} (P={rs['best_metrics']['precision']:.4f}, "
                        f"R={rs['best_metrics']['recall']:.4f}) -> test F1={e2e['f1']:.4f} (P={e2e['precision']:.4f}, "
                        f"R={e2e['recall']:.4f}, FPR={e2e['fpr']:.4f}, FNR={e2e['fnr']:.4f})")
    lines.extend(rows_txt)
    lines.append("")
    lines.append("## Confusion matrice (test, nad validnim predikcijama)\n")
    for cfg in configs:
        run_id = cfg["run_id"]
        valid_m = all_results[run_id]["valid_m"]
        cm = pd.DataFrame([[valid_m["tp"], valid_m["fn"]], [valid_m["fp"], valid_m["tn"]]],
                          index=["true_harmful", "true_unharmful"], columns=["pred_harmful", "pred_unharmful"])
        lines.append(f"**{run_id}**:\n```\n{cm.to_string()}\n```\n")
    lines.append("## FP/FN/invalid analiza\n")
    for cfg in configs:
        run_id = cfg["run_id"]
        rf = all_results[run_id]["results_full"]
        n_fp = int(((rf["final_label"] != POSITIVE) & (rf["prediction"] == POSITIVE)).sum())
        n_fn = int(((rf["final_label"] == POSITIVE) & (rf["prediction"] != POSITIVE) & (rf["prediction"] != "invalid")).sum())
        n_invalid = int((rf["prediction"] == "invalid").sum())
        lines.append(f"- **{run_id}**: {n_fp} false positives, {n_fn} false negatives, {n_invalid} invalid "
                    f"(regex baseline/other systems' invalid handled identically -- invalid never becomes a "
                    f"valid label; on harmful rows it counts as FN, on unharmful rows as FP, end-to-end).")
    lines.append("")
    lines.append("## Objedinjena tabela svih sistema (test, end-to-end)\n")
    display_cols = ["System", "Method", "Target scope", "precision", "recall", "f1", "fpr", "fnr", "accuracy", "invalid_rate"]
    lines.append("```\n" + pd.DataFrame(comparison_rows)[display_cols].to_string(index=False) + "\n```\n")

    by_system = {r["System"]: r for r in comparison_rows}
    exp1 = by_system["Gemma initial LoRA"]
    lora_attn = by_system["LoRA attention-only"]
    dora_all = by_system["DoRA all-linear"]
    dora_attn = by_system["DoRA attention-only"]
    best_f1 = max(comparison_rows, key=lambda r: r["f1"])
    best_recall = max(comparison_rows, key=lambda r: r["recall"])

    lines.append("## Kratki zaključci\n")
    lines.append(
        f"- Efekat uklanjanja MLP target modula (LoRA, test): all-linear (Exp1) F1={exp1['f1']:.4f} -> "
        f"attention-only F1={lora_attn['f1']:.4f} (delta={lora_attn['f1']-exp1['f1']:+.4f}).\n"
        f"- Efekat LoRA->DoRA (all-linear, test): F1={exp1['f1']:.4f} -> {dora_all['f1']:.4f} "
        f"(delta={dora_all['f1']-exp1['f1']:+.4f}).\n"
        f"- Efekat LoRA->DoRA (attention-only, test): F1={lora_attn['f1']:.4f} -> {dora_attn['f1']:.4f} "
        f"(delta={dora_attn['f1']-lora_attn['f1']:+.4f}).\n"
        f"- Najveći test F1: **{best_f1['System']}** (F1={best_f1['f1']:.4f}).\n"
        f"- Najveći harmful recall: **{best_recall['System']}** (recall={best_recall['recall']:.4f}).\n"
    )
    lines.append(
        f"**Ponovljena metodološka napomena**: `dora_all_linear` je zaključan kao izabrani model PRE test "
        f"evaluacije, na osnovu validation F1. Test rezultati za `lora_attention_only` i "
        f"`dora_attention_only` NISU osnova za bilo kakvu naknadnu promenu te selekcije.\n"
    )

    root_report_path.write_text(existing.rstrip("\n") + "\n" + "\n".join(lines))
    print(f"[OK] Sekcija 'Konačna test evaluacija' dodata u {root_report_path} (postojeći sadržaj netaknut).")


if __name__ == "__main__":
    main()
