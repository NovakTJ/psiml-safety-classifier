"""
Hyperparameter sweep for the v2 linear probe (attempt 2).

Attempt 1 (scripts/model/results/linear_probe_v2_layer27_lasttoken/) used GUESSED
hyperparameters (StandardScaler -> LogisticRegression C=1.0, class_weight=balanced,
lbfgs) and reached validation F1 0.910 — but with train F1 = 1.000 (perfectly
memorized train set), i.e. clearly under-regularized for a 4096-dim feature with
1985 train rows. This sweep tunes the probe hyperparameters on the ALREADY-CACHED
activations (pure CPU, no GPU, no model load):

  - penalty:        l2 (lbfgs) and l1 (saga, sparse probes)
  - C:              log grid 1e-4 .. 3.0  (the big knob, given the overfit)
  - class_weight:   None / balanced / {0:1, 1:2}  (recall-oriented)

Feature, input format, layer (27), train split, and scaler (StandardScaler) are
all LOCKED from attempt 1 — only the LogisticRegression hyperparameters move.
Selection: max validation F1 at threshold 0.5 (comparable to attempt 1).
After selection, the decision threshold is tuned on validation (per the final
system design, validation selects the blocking threshold) and reported
separately. test.jsonl remains UNTOUCHED (not captured, not evaluated).

Run with ccpp_env (no GPU needed):

    /home/mls01/ccpp_env/bin/python scripts/model/sweep_probe_v2.py
"""

import csv
import json
import os
import time
import warnings

import numpy as np

import probe_v2_common as C

OUT_DIR = ("/home/mls01/scripts/model/results/"
           "linear_probe_v2_layer27_lasttoken_sweep")
ATTEMPT1_DIR = C.OUT_DIR  # attempt-1 experiment dir (read-only here)

SEED = 42
MAX_ITER = 5000

L2_CS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0]
L1_CS = [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
CLASS_WEIGHTS = [
    ("none", None),
    ("balanced", "balanced"),
    ("harmful_x2", {0: 1.0, 1: 2.0}),
]

# Locked from attempt 1 (recorded, not swept).
LOCKED = {
    "feature": "last-token residual stream, decoder layer 27 "
               "(second-to-last full-attention layer), bf16->fp32",
    "input_format": "teacher-forced v2 exchange; chat template with "
                    "add_generation_prompt=True, enable_thinking=False; "
                    "response rows get response tokens + <|im_end|> appended",
    "label": "final_label (harmful=1, unharmful=0)",
    "scaler": "StandardScaler",
    "train_split": "data/gemma_v2_no_refusal/train.jsonl",
    "seed": SEED,
}


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "run.log"), "a") as f:
        f.write(line + "\n")


def metrics_at(y, prob, threshold):
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else None,
        "accuracy": float(accuracy_score(y, pred)),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn),
                      "tn": int(tn)},
    }


def tune_threshold(y, prob):
    """Scan thresholds, maximize F1; ties -> higher recall -> higher threshold
    (fewer false alarms). Returns (threshold, metrics_dict)."""
    best = None
    for t in np.arange(0.01, 1.0, 0.005):
        m = metrics_at(y, prob, float(t))
        key = (m["f1"], m["recall"], m["threshold"])
        if best is None or key > (best[1]["f1"], best[1]["recall"],
                                  best[1]["threshold"]):
            best = (float(t), m)
    return best


def build_grid():
    grid = []
    for penalty, solver, cs in (("l2", "lbfgs", L2_CS),
                                ("l1", "saga", L1_CS)):
        for cw_name, cw in CLASS_WEIGHTS:
            for c in cs:
                grid.append({
                    "penalty": penalty, "solver": solver, "C": c,
                    "class_weight_name": cw_name, "class_weight": cw,
                })
    return grid


def fit_one(cfg, X_train, y_train):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            penalty=cfg["penalty"], solver=cfg["solver"], C=cfg["C"],
            class_weight=cfg["class_weight"], max_iter=MAX_ITER,
            random_state=SEED,
            n_jobs=-1 if cfg["solver"] == "saga" else None)),
    ])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_s = time.time() - t0
    converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    lr = clf.named_steps["lr"]
    return clf, {
        "fit_seconds": round(fit_s, 2),
        "lr_n_iter": int(lr.n_iter_[0]),
        "converged": bool(converged),
        "n_nonzero_coef": int((np.abs(lr.coef_) > 1e-8).sum()),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log("=== sweep_probe_v2 starting (CPU-only, cached activations) ===")

    # Guard: activations for train+validation must be fully cached; test must
    # stay uncaptured/unevaluated.
    for split in ("train", "validation"):
        n_done = len(C.done_row_ids(split))
        n_rows = len(C.load_rows(split))
        assert n_done == n_rows, \
            f"{split}: {n_done}/{n_rows} activations cached — run " \
            f"train_probe_v2.py / eval_probe_v2.py first"
    assert len(C.done_row_ids("test")) == 0, \
        "test activations exist — this sweep must not touch the test split"

    X_train, y_train, _, _ = C.load_activation_matrix("train")
    X_val, y_val, val_row_ids, _ = C.load_activation_matrix("validation")

    grid = build_grid()
    log(f"Grid: {len(grid)} configs "
        f"({len(L2_CS)} l2-Cs + {len(L1_CS)} l1-Cs) x "
        f"{len(CLASS_WEIGHTS)} class weights")

    results = []
    probs_by_key = {}
    for i, cfg in enumerate(grid):
        key = (f"{cfg['penalty']}_{cfg['solver']}_C{cfg['C']:g}_"
               f"cw_{cfg['class_weight_name']}")
        try:
            clf, fit_info = fit_one(cfg, X_train, y_train)
        except Exception as e:  # never let one config kill the sweep
            log(f"[{i+1}/{len(grid)}] {key}: FAILED ({e!r})")
            results.append({"config": key, **{k: cfg[k] for k in
                            ("penalty", "solver", "C", "class_weight_name")},
                            "error": repr(e)})
            continue
        val_prob = clf.predict_proba(X_val)[:, 1]
        train_prob = clf.predict_proba(X_train)[:, 1]
        vm = metrics_at(y_val, val_prob, 0.5)
        tm = metrics_at(y_train, train_prob, 0.5)
        row = {
            "config": key,
            "penalty": cfg["penalty"], "solver": cfg["solver"],
            "C": cfg["C"], "class_weight": cfg["class_weight_name"],
            "is_attempt1_config": bool(
                cfg["penalty"] == "l2" and cfg["C"] == 1.0
                and cfg["class_weight_name"] == "balanced"),
            **fit_info,
            "train_f1": tm["f1"],
            "val_precision": vm["precision"], "val_recall": vm["recall"],
            "val_f1": vm["f1"], "val_fpr": vm["fpr"],
            "val_accuracy": vm["accuracy"],
        }
        results.append(row)
        probs_by_key[key] = (clf, val_prob)
        log(f"[{i+1}/{len(grid)}] {key}: val F1 {vm['f1']:.4f} "
            f"(P {vm['precision']:.3f} R {vm['recall']:.3f}) "
            f"train F1 {tm['f1']:.4f} fit {fit_info['fit_seconds']}s "
            f"{'CONVERGED' if fit_info['converged'] else 'NOT-converged'}")

    # --- selection: max validation F1 at threshold 0.5 (attempt-1 protocol) ---
    ok = [r for r in results if "error" not in r]
    ok.sort(key=lambda r: (-r["val_f1"], -r["val_recall"], r["C"]))
    best = ok[0]
    log(f"Selected config: {best['config']} "
        f"(val F1 {best['val_f1']:.4f} at threshold 0.5)")

    best_clf, best_val_prob = probs_by_key[best["config"]]

    # --- threshold tuning on validation (blocking-threshold selection) ---
    thr, tuned = tune_threshold(y_val, best_val_prob)
    log(f"Tuned threshold on validation: {thr:.3f} -> "
        f"F1 {tuned['f1']:.4f} (P {tuned['precision']:.3f} "
        f"R {tuned['recall']:.3f})")
    default_m = metrics_at(y_val, best_val_prob, 0.5)
    train_m = metrics_at(y_train, best_clf.predict_proba(X_train)[:, 1], 0.5)

    # --- save everything ---
    csv_path = os.path.join(OUT_DIR, "sweep_results.csv")
    fieldnames = sorted({k for r in results for k in r})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    import joblib
    joblib.dump(best_clf, os.path.join(OUT_DIR, "probe.joblib"))

    probe_config = {
        **LOCKED,
        "model": {
            "type": "sklearn Pipeline(StandardScaler -> LogisticRegression)",
            "penalty": best["penalty"], "solver": best["solver"],
            "C": best["C"], "class_weight": best["class_weight"],
            "max_iter": MAX_ITER, "random_state": SEED,
        },
        "selection": {
            "metric": "validation F1 at threshold 0.5",
            "tie_break": "higher recall, then smaller C (stronger reg)",
            "n_configs_tried": len(results),
            "n_configs_failed": len(results) - len(ok),
        },
        "decision_threshold": {
            "default": 0.5,
            "tuned_on_validation": thr,
            "tuning": "max validation F1, ties -> higher recall -> "
                      "higher threshold",
        },
        "attempt": 2,
        "attempt1_dir": ATTEMPT1_DIR,
    }
    with open(os.path.join(OUT_DIR, "probe_config.json"), "w") as f:
        json.dump(probe_config, f, indent=2)

    metrics_val = {
        "split": "validation",
        "n_rows": int(len(y_val)), "n_harmful": int(y_val.sum()),
        "at_threshold_0.5": default_m,
        "at_tuned_threshold": tuned,
        "train_at_threshold_0.5": train_m,
        "fit_seconds": best["fit_seconds"],
        "lr_n_iter": best["lr_n_iter"],
        "n_nonzero_coef": best["n_nonzero_coef"],
    }
    with open(os.path.join(OUT_DIR, "metrics_validation.json"), "w") as f:
        json.dump(metrics_val, f, indent=2)

    pred = (best_val_prob >= 0.5).astype(int)
    with open(os.path.join(OUT_DIR, "predictions_validation.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "final_label", "prob_harmful",
                    "pred_label@0.5", "correct@0.5"])
        for rid, yi, pi, di in zip(val_row_ids, y_val, best_val_prob, pred):
            w.writerow([rid, "harmful" if yi else "unharmful",
                        f"{pi:.6f}", "harmful" if di else "unharmful",
                        int(yi == di)])

    # --- REPORT.md (auto-generated so it can't drift from the code) ---
    a1 = json.load(open(os.path.join(ATTEMPT1_DIR, "metrics_validation.json")))
    top = ok[:10]
    lines = [
        "# Linear Probe v2 — Hyperparameter Sweep (Attempt 2)",
        "",
        f"Date: {time.strftime('%Y-%m-%d')}. CPU-only sweep on the cached "
        "teacher-forced Qwen3.5-9B activations from attempt 1 "
        "(`linear_probe_v2_layer27_lasttoken/`). No GPU, no model load, no new "
        "captures. Feature (last-token residual @ decoder layer 27), input "
        "format, train split, and StandardScaler are locked from attempt 1; "
        "only the LogisticRegression hyperparameters moved.",
        "",
        "## Why",
        "",
        f"Attempt 1 used guessed hyperparameters (lbfgs, C=1.0, "
        f"class_weight=balanced) and hit validation F1 {a1['f1']:.4f} "
        f"(P {a1['precision']:.3f} / R {a1['recall']:.3f}) — but train F1 was "
        "1.000 (perfect memorization of 1985 rows in 4096 dims), i.e. clearly "
        "under-regularized. The sweep's main knob is therefore C, plus "
        "L1-vs-L2 and class_weight (the project prioritizes unsafe recall).",
        "",
        "## Grid",
        "",
        f"- penalty/solver: l2 (lbfgs) over C = {L2_CS}; "
        f"l1 (saga) over C = {L1_CS}",
        f"- class_weight: none / balanced / {{0:1, 1:2}}",
        f"- total {len(results)} configs, {len(results) - len(ok)} failed, "
        f"max_iter={MAX_ITER}, seed={SEED}",
        "- selection: max validation F1 at threshold 0.5 (same protocol as "
        "attempt 1; ties -> higher recall -> smaller C)",
        "- **test.jsonl untouched** (asserted uncaptured at sweep start).",
        "",
        "## Top 10 configs by validation F1 (threshold 0.5)",
        "",
        "| # | config | val P | val R | val F1 | val FPR | train F1 | fit s |",
        "|---|--------|------:|------:|-------:|--------:|---------:|------:|",
    ]
    for i, r in enumerate(top):
        lines.append(
            f"| {i+1} | {r['config']} | {r['val_precision']:.3f} | "
            f"{r['val_recall']:.3f} | **{r['val_f1']:.4f}** | "
            f"{r['val_fpr']:.3f} | {r['train_f1']:.4f} | {r['fit_seconds']} |")
    lines += [
        "",
        "## Selected config",
        "",
        f"`{best['config']}` — Pipeline(StandardScaler -> LogisticRegression("
        f"penalty={best['penalty']}, solver={best['solver']}, C={best['C']:g}, "
        f"class_weight={best['class_weight']}, max_iter={MAX_ITER})).",
        "",
        f"- Validation @0.5: P {default_m['precision']:.3f} / "
        f"R {default_m['recall']:.3f} / F1 {default_m['f1']:.4f} / "
        f"FPR {default_m['fpr']:.3f} (confusion {default_m['confusion']})",
        f"- Train @0.5: F1 {train_m['f1']:.4f} (attempt 1 had train F1 1.000; "
        f"the selected C {'still fully separates train' if train_m['f1'] == 1.0 else 'no longer fully separates train'})",
        f"- Nonzero coefficients: {best['n_nonzero_coef']}/4096"
        + (" (L1 sparse probe)" if best["penalty"] == "l1" else ""),
        "",
        "## Decision-threshold tuning (validation)",
        "",
        f"Scanning thresholds 0.01..0.99 for the selected config: best "
        f"threshold **{thr:.3f}** -> P {tuned['precision']:.3f} / "
        f"R {tuned['recall']:.3f} / F1 {tuned['f1']:.4f} / "
        f"FPR {tuned['fpr']:.3f}. "
        f"(At the default 0.5: F1 {default_m['f1']:.4f}.) This is the "
        "blocking-threshold selection step from the final system design; "
        "predictions/metrics above are reported at 0.5 for comparability "
        "with attempt 1.",
        "",
        "## Comparison on the same validation split (259 rows)",
        "",
        "| System | P | R | F1 |",
        "|--------|---:|---:|---:|",
        f"| Probe attempt 1 (guessed C=1.0 balanced) | {a1['precision']:.3f} "
        f"| {a1['recall']:.3f} | {a1['f1']:.4f} |",
        f"| **Probe attempt 2 (swept, @0.5)** | {default_m['precision']:.3f} "
        f"| {default_m['recall']:.3f} | **{default_m['f1']:.4f}** |",
        f"| Probe attempt 2 (swept, tuned thr {thr:.3f}) | "
        f"{tuned['precision']:.3f} | {tuned['recall']:.3f} | "
        f"{tuned['f1']:.4f} |",
        "| Gemma zero-shot (v2) | 0.776 | 0.910 | 0.838 |",
        "| Qwen3Guard native (v2) | 0.960 | 0.894 | 0.9256 |",
        "| Locked Gemma LoRA (v2) | 0.981 | 0.956 | 0.9684 |",
        "",
        "## Notes / limits",
        "",
        "- Validation-selected on 259 rows; expect some selection noise. The "
        "LoRA ablation already showed a validation winner can drop on test — "
        "treat the sweep ranking as noisy beyond the top few configs.",
        "- Only layer 27 / last-token features exist on disk. A layer sweep "
        "or token-window pooling needs NEW captures (GPU) — separate "
        "experiment.",
        "- Threshold tuning also happens on the same 259 validation rows, so "
        "the tuned-threshold F1 is mildly optimistic; the @0.5 number is the "
        "honest like-for-like comparison with attempt 1.",
        "- test.jsonl remains uncaptured and unevaluated; the one-shot test "
        "eval is a separate future step once the probe recipe is fully locked.",
        "",
        "## Artifacts",
        "",
        "- `sweep_results.csv` — all configs, fit stats, val/train metrics",
        "- `probe.joblib` — selected probe (StandardScaler + LogReg pipeline)",
        "- `probe_config.json` — full locked + selected config, thresholds",
        "- `metrics_validation.json`, `predictions_validation.csv`",
        "- `run.log`",
    ]
    with open(os.path.join(OUT_DIR, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    log(f"Saved sweep_results.csv, probe.joblib, probe_config.json, "
        f"metrics_validation.json, predictions_validation.csv, REPORT.md "
        f"to {OUT_DIR}")
    log("=== sweep_probe_v2 DONE ===")


if __name__ == "__main__":
    main()
