"""
Layer x pooling screen for the v2 linear probe, on top of the multi-layer
activation cache built by capture_probe_multilayer.py
(results/linear_probe_v3_multilayer_pooling/).

Motivation: attempt 1/2 (results/linear_probe_v2_layer27_lasttoken[_sweep]/)
locked layer 27 + last-token by heuristic, never compared against another
layer or pooling, and its 51-config LogisticRegression grid landed on a tight
F1 plateau (0.908-0.923) with validation recall stuck at ~0.900 in almost
every good cell -- the signature of a feature ceiling, not a classifier one.
This script checks whether ANY of the other 31 (layer, pooling) combinations
captured alongside layer27/last break that ceiling.

Two phases, CPU-only, no GPU, no model load (pure numpy/sklearn over the
cached .npz activations):

  Phase 1 (screen, cheap): for all 32 (layer, pooling) combos, fit ONE fixed
  LogisticRegression config -- C=0.03, class_weight=balanced, lbfgs -- the
  attempt-2 sweep winner, so the 32 numbers are an apples-to-apples
  comparison against the OLD layer27_last F1 of 0.9231 (same hyperparameters,
  only the feature changes). ~32 fast L2/lbfgs fits.

  Phase 2 (refine, still cheap): take the top N combos from phase 1 by val
  F1, run a small L2/lbfgs-only grid (attempt-2 already showed L1/saga never
  wins and is up to ~100x slower -- one saga fit took 420s vs 3.9s for the
  L2 winner -- so it is excluded here entirely) over C x class_weight on
  just those combos.

Selection: max validation F1 at threshold 0.5 (same protocol as attempt 1/2),
tie-break higher recall then smaller C. Decision threshold is then tuned on
validation for the overall winner (blocking-threshold step from the final
system design). test.jsonl is NEVER touched (asserted uncaptured at start,
matching attempt 2's guard).

Run with ccpp_env (no GPU needed):

    /home/mls01/ccpp_env/bin/python scripts/model/probe_v3/sweep_probe_v3_layer_pooling.py
"""

import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# probe_v2_common.py (teammate's, shared) lives one level up in scripts/model/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import capture_probe_multilayer as CM
import probe_v2_common as C

CAPTURE_DIR = CM.DEFAULT_OUT_DIR
OUT_DIR = "/home/mls01/scripts/model/results/linear_probe_v3_multilayer_pooling_sweep"

SEED = 42
MAX_ITER = 5000
TOP_N_PHASE2 = 5

# Attempt-2 sweep winner, reused here as the phase-1 fixed reference config.
FIXED_CONFIG = {"C": 0.03, "class_weight": "balanced", "class_weight_name": "balanced"}
FIXED_VAL_F1_LAYER27_LAST = 0.9231  # attempt-2 sweep result, for comparison only

# Phase-2 grid: L2/lbfgs only (L1/saga excluded, see docstring).
PHASE2_CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
PHASE2_CLASS_WEIGHTS = [
    ("none", None),
    ("balanced", "balanced"),
    ("harmful_x2", {0: 1.0, 1: 2.0}),
]

FEATURE_KEYS = [f"layer{L}_{p}" for L in CM.CAPTURE_LAYERS for p in CM.POOLINGS]


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
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }


def tune_threshold(y, prob):
    best = None
    for t in np.arange(0.01, 1.0, 0.005):
        m = metrics_at(y, prob, float(t))
        key = (m["f1"], m["recall"], m["threshold"])
        if best is None or key > (best[1]["f1"], best[1]["recall"], best[1]["threshold"]):
            best = (float(t), m)
    return best


def fit_one(C_val, class_weight, X_train, y_train):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(penalty="l2", solver="lbfgs", C=C_val,
                                  class_weight=class_weight, max_iter=MAX_ITER,
                                  random_state=SEED)),
    ])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_s = time.time() - t0
    converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    return clf, round(fit_s, 2), converged


def load_all_features(split):
    """One pass over the .npz cache: {feature_key: X (n,4096) float32}, y, row_ids."""
    rows = C.load_rows(split)
    act_dir = CM.act_dir_for(CAPTURE_DIR, split)
    per_key = {k: np.empty((len(rows), CM.HIDDEN_SIZE), dtype=np.float32)
               for k in FEATURE_KEYS}
    y = np.empty(len(rows), dtype=np.int64)
    row_ids = []
    for i, r in enumerate(rows):
        path = os.path.join(act_dir, f"{r['row_id']}.npz")
        assert os.path.exists(path), f"missing activation for {r['row_id']}"
        d = np.load(path)
        for k in FEATURE_KEYS:
            per_key[k][i] = d[k].astype(np.float32)
        y[i] = 1 if r["final_label"] == "harmful" else 0
        row_ids.append(r["row_id"])
    log(f"[{split}] assembled {len(FEATURE_KEYS)} feature matrices: "
        f"{len(rows)} rows, harmful {int(y.sum())}/{len(y)}")
    return per_key, y, row_ids


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log("=== sweep_probe_v3_layer_pooling starting (CPU-only, cached activations) ===")

    for split in ("train", "validation"):
        n_done = len(CM.done_row_ids(CAPTURE_DIR, split))
        n_rows = len(C.load_rows(split))
        assert n_done == n_rows, \
            f"{split}: {n_done}/{n_rows} multilayer activations cached — " \
            f"run capture_probe_multilayer.py --split {split} first"
    assert len(CM.done_row_ids(CAPTURE_DIR, "test")) == 0, \
        "test activations exist in the v3 capture dir — this sweep must not touch test"

    log(f"Loading train/validation features for {len(FEATURE_KEYS)} "
        f"(layer, pooling) combos ...")
    t0 = time.time()
    train_feat, y_train, _ = load_all_features("train")
    val_feat, y_val, val_row_ids = load_all_features("validation")
    log(f"Loaded in {time.time()-t0:.1f}s")

    # --- Phase 1: fixed config across all 32 combos ---------------------
    log(f"Phase 1: {len(FEATURE_KEYS)} combos x fixed config "
        f"(C={FIXED_CONFIG['C']}, class_weight={FIXED_CONFIG['class_weight_name']})")
    phase1_rows = []
    for i, key in enumerate(FEATURE_KEYS):
        clf, fit_s, converged = fit_one(FIXED_CONFIG["C"], FIXED_CONFIG["class_weight"],
                                        train_feat[key], y_train)
        prob = clf.predict_proba(val_feat[key])[:, 1]
        m = metrics_at(y_val, prob, 0.5)
        phase1_rows.append({"feature_key": key, "fit_seconds": fit_s,
                            "converged": converged, **m})
        log(f"[phase1 {i+1}/{len(FEATURE_KEYS)}] {key}: "
            f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
            f"({fit_s:.1f}s)")

    phase1_rows.sort(key=lambda r: r["f1"], reverse=True)
    with open(os.path.join(OUT_DIR, "phase1_screen.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_key", "precision", "recall", "f1", "fpr", "fnr",
                    "accuracy", "fit_seconds", "converged"])
        for r in phase1_rows:
            w.writerow([r["feature_key"], r["precision"], r["recall"], r["f1"],
                        r["fpr"], r["fnr"], r["accuracy"], r["fit_seconds"],
                        r["converged"]])
    best1 = phase1_rows[0]
    log(f"Phase 1 best: {best1['feature_key']} F1={best1['f1']:.4f} "
        f"(old layer27_last reference: F1={FIXED_VAL_F1_LAYER27_LAST:.4f})")

    top_keys = [r["feature_key"] for r in phase1_rows[:TOP_N_PHASE2]]
    log(f"Phase 2 candidates (top {TOP_N_PHASE2} by phase-1 F1): {top_keys}")

    # --- Phase 2: small L2 grid on the top candidates --------------------
    phase2_rows = []
    probs_by_config = {}
    for key in top_keys:
        for C_val in PHASE2_CS:
            for cw_name, cw in PHASE2_CLASS_WEIGHTS:
                clf, fit_s, converged = fit_one(C_val, cw, train_feat[key], y_train)
                prob = clf.predict_proba(val_feat[key])[:, 1]
                m = metrics_at(y_val, prob, 0.5)
                config_id = f"{key}__C{C_val:g}_cw_{cw_name}"
                phase2_rows.append({"feature_key": key, "C": C_val,
                                    "class_weight": cw_name, "config_id": config_id,
                                    "fit_seconds": fit_s, "converged": converged,
                                    **m})
                probs_by_config[config_id] = (clf, prob)
        log(f"[phase2] {key}: {len(PHASE2_CS)*len(PHASE2_CLASS_WEIGHTS)} configs done")

    phase2_rows.sort(key=lambda r: (r["f1"], r["recall"], -r["C"]), reverse=True)
    with open(os.path.join(OUT_DIR, "phase2_grid.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config_id", "feature_key", "C", "class_weight", "precision",
                    "recall", "f1", "fpr", "fnr", "accuracy", "fit_seconds",
                    "converged"])
        for r in phase2_rows:
            w.writerow([r["config_id"], r["feature_key"], r["C"], r["class_weight"],
                        r["precision"], r["recall"], r["f1"], r["fpr"], r["fnr"],
                        r["accuracy"], r["fit_seconds"], r["converged"]])

    winner = phase2_rows[0]
    winner_clf, winner_prob = probs_by_config[winner["config_id"]]
    log(f"Overall winner: {winner['config_id']} "
        f"P={winner['precision']:.4f} R={winner['recall']:.4f} F1={winner['f1']:.4f}")

    thr, tuned = tune_threshold(y_val, winner_prob)
    log(f"Tuned threshold on validation: {thr:.3f} -> "
        f"P={tuned['precision']:.4f} R={tuned['recall']:.4f} F1={tuned['f1']:.4f}")

    import joblib
    joblib.dump(winner_clf, os.path.join(OUT_DIR, "probe.joblib"))

    with open(os.path.join(OUT_DIR, "predictions_validation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "label", "prob_harmful", "pred_at_0.5",
                    f"pred_at_{thr:.3f}"])
        for rid, label, p in zip(val_row_ids, y_val, winner_prob):
            w.writerow([rid, int(label), float(p), int(p >= 0.5), int(p >= thr)])

    summary = {
        "capture_dir": CAPTURE_DIR,
        "feature_keys_tested": FEATURE_KEYS,
        "phase1_fixed_config": FIXED_CONFIG,
        "phase1_best": best1,
        "old_attempt2_reference_f1_layer27_last": FIXED_VAL_F1_LAYER27_LAST,
        "phase2_top_candidates": top_keys,
        "phase2_grid_size": len(phase2_rows),
        "winner": {
            "config_id": winner["config_id"], "feature_key": winner["feature_key"],
            "C": winner["C"], "class_weight": winner["class_weight"],
            "at_threshold_0.5": {k: winner[k] for k in
                                 ("precision", "recall", "f1", "fpr", "fnr", "accuracy")},
            "at_tuned_threshold": tuned,
        },
        "seed": SEED,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Linear probe v3 -- layer x pooling screen",
        "",
        f"Multi-layer capture: {CAPTURE_DIR}",
        f"8 full-attention layers ({CM.CAPTURE_LAYERS}) x 4 poolings "
        f"({CM.POOLINGS}) = {len(FEATURE_KEYS)} feature keys, all from the "
        "SAME forward pass as attempt 1/2's layer27/last-token capture.",
        "",
        "## Phase 1 -- fixed config (C=0.03, class_weight=balanced) across all "
        f"{len(FEATURE_KEYS)} combos",
        "",
        f"Reference: attempt-2 sweep's locked layer27_last, same hyperparameters, "
        f"scored val F1 **{FIXED_VAL_F1_LAYER27_LAST:.4f}**.",
        "",
        "| Rank | feature_key | precision | recall | f1 | fpr |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(phase1_rows[:10]):
        lines.append(f"| {i+1} | {r['feature_key']} | {r['precision']:.4f} | "
                     f"{r['recall']:.4f} | {r['f1']:.4f} | {r['fpr']:.4f} |")
    lines += [
        "",
        f"## Phase 2 -- L2/lbfgs grid on top {TOP_N_PHASE2}: {top_keys}",
        "",
        f"{len(PHASE2_CS)} C values x {len(PHASE2_CLASS_WEIGHTS)} class weights x "
        f"{TOP_N_PHASE2} feature keys = {len(phase2_rows)} fits.",
        "",
        "| Rank | config_id | precision | recall | f1 |",
        "|---|---|---:|---:|---:|",
    ]
    for i, r in enumerate(phase2_rows[:10]):
        lines.append(f"| {i+1} | {r['config_id']} | {r['precision']:.4f} | "
                     f"{r['recall']:.4f} | {r['f1']:.4f} |")
    lines += [
        "",
        f"## Winner: `{winner['config_id']}`",
        "",
        f"- At threshold 0.5: P={winner['precision']:.4f} R={winner['recall']:.4f} "
        f"F1={winner['f1']:.4f} FPR={winner['fpr']:.4f}",
        f"- Tuned threshold {thr:.3f}: P={tuned['precision']:.4f} "
        f"R={tuned['recall']:.4f} F1={tuned['f1']:.4f}",
        f"- vs old attempt-2 winner (layer27_last, F1 {FIXED_VAL_F1_LAYER27_LAST:.4f}): "
        f"delta = {winner['f1']-FIXED_VAL_F1_LAYER27_LAST:+.4f}",
        "",
        "test.jsonl was NOT captured or evaluated in this run.",
    ]
    with open(os.path.join(OUT_DIR, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    log(f"Saved phase1_screen.csv, phase2_grid.csv, summary.json, REPORT.md, "
        f"probe.joblib, predictions_validation.csv to {OUT_DIR}")
    log("=== sweep_probe_v3_layer_pooling DONE ===")


if __name__ == "__main__":
    main()
