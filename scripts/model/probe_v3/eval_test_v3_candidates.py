"""
One-shot test.jsonl evaluation of the v3 probe candidates -- the FIRST time
test is touched anywhere in the linear-probe experiment (attempt 1, attempt
2, and both v3 sweeps deliberately left it uncaptured/unevaluated).

Candidates are picked from sweep_probe_v3_response_aware.py's
grid_results.csv by a rule fixed BEFORE this script ever sees test data (no
cherry-picking after the fact):
  - winner_combined:      max combined_score (0.5*val_f1 + 0.5*switched_recall)
  - best_val_f1:          max val_f1 alone, for reference
  - best_switched_recall: max switched_recall alone, for reference
(duplicates collapsed if the same config wins more than one criterion)

Plus one fixed reference, NOT selected from this sweep: v2's locked
attempt-2 winner (layer27_last, results/linear_probe_v2_layer27_lasttoken_sweep/
probe.joblib) -- the pre-existing baseline this whole v3 investigation is
trying to beat.

Requires: capture_probe_multilayer.py --split test already run (GPU).

Run with ccpp_env (CPU-only, no GPU needed):

    /home/mls01/ccpp_env/bin/python scripts/model/probe_v3/eval_test_v3_candidates.py
"""

import csv
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np

# probe_v2_common.py (teammate's, shared) lives one level up in scripts/model/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import capture_probe_multilayer as CM
import probe_v2_common as C

RESPONSE_SWEEP_DIR = ("/home/mls01/scripts/model/results/"
                      "linear_probe_v3_multilayer_pooling_sweep_response_aware")
V2_SWEEP_DIR = "/home/mls01/scripts/model/results/linear_probe_v2_layer27_lasttoken_sweep"
V3_CAPTURE_DIR = CM.DEFAULT_OUT_DIR
OUT_DIR = os.path.join(RESPONSE_SWEEP_DIR, "test_evaluation")

CW_MAP = {"none": None, "balanced": "balanced", "harmful_x2": {0: 1.0, 1: 2.0}}
SEED = 42


def load_grid():
    with open(os.path.join(RESPONSE_SWEEP_DIR, "grid_results.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("C", "val_precision", "val_recall", "val_f1", "val_fpr",
                  "switched_recall", "combined_score", "fit_seconds"):
            r[k] = float(r[k])
    return rows


def pick_candidates(rows):
    by_combined = max(rows, key=lambda r: r["combined_score"])
    by_valf1 = max(rows, key=lambda r: r["val_f1"])
    by_switch = max(rows, key=lambda r: r["switched_recall"])
    cands = {}
    for label, r in [("winner_combined", by_combined), ("best_val_f1_alone", by_valf1),
                     ("best_switched_recall_alone", by_switch)]:
        cands.setdefault(r["config_id"], {"labels": [], "row": r})
        cands[r["config_id"]]["labels"].append(label)
    return cands


def load_split_feature(split, key):
    rows = C.load_rows(split)
    act_dir = CM.act_dir_for(V3_CAPTURE_DIR, split)
    X = np.empty((len(rows), CM.HIDDEN_SIZE), dtype=np.float32)
    y = np.empty(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        d = np.load(os.path.join(act_dir, f"{r['row_id']}.npz"))
        X[i] = d[key].astype(np.float32)
        y[i] = 1 if r["final_label"] == "harmful" else 0
    return X, y


def fit_probe(feature_key, C_val, cw_name):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_train, y_train = load_split_feature("train", feature_key)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C_val, class_weight=CW_MAP[cw_name],
                                  max_iter=5000, random_state=SEED)),
    ])
    clf.fit(X_train, y_train)
    return clf


def metrics(y, pred):
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "accuracy": float(accuracy_score(y, pred)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main():
    n_test = len(C.load_rows("test"))
    n_done = len(CM.done_row_ids(V3_CAPTURE_DIR, "test"))
    assert n_done == n_test, (
        f"test: {n_done}/{n_test} v3 activations cached — run "
        f"capture_probe_multilayer.py --split test first")

    grid = load_grid()
    cands = pick_candidates(grid)

    results = []
    for config_id, info in cands.items():
        r = info["row"]
        clf = fit_probe(r["feature_key"], r["C"], r["class_weight"])
        X_test, y_test = load_split_feature("test", r["feature_key"])
        pred = clf.predict(X_test)
        m = metrics(y_test, pred)
        results.append({"name": config_id, "labels": ",".join(info["labels"]),
                        "feature_key": r["feature_key"], "C": r["C"],
                        "class_weight": r["class_weight"], **m})

    v2_locked = joblib.load(os.path.join(V2_SWEEP_DIR, "probe.joblib"))
    X_test27, y_test27 = load_split_feature("test", "layer27_last")
    pred27 = v2_locked.predict(X_test27)
    m27 = metrics(y_test27, pred27)
    results.append({"name": "v2_locked_layer27_last", "labels": "pre_existing_baseline",
                    "feature_key": "layer27_last", "C": None, "class_weight": None,
                    **m27})

    results.sort(key=lambda r: r["f1"], reverse=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT_DIR, "test_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "labels", "feature_key", "C", "class_weight",
                    "precision", "recall", "f1", "fpr", "accuracy",
                    "tp", "fp", "fn", "tn"])
        for r in results:
            w.writerow([r["name"], r["labels"], r["feature_key"], r["C"],
                        r["class_weight"], r["precision"], r["recall"], r["f1"],
                        r["fpr"], r["accuracy"], r["tp"], r["fp"], r["fn"], r["tn"]])

    print(f"\n{'name':45s} {'labels':30s} {'P':>7s} {'R':>7s} {'F1':>7s} {'FPR':>7s}")
    for r in results:
        print(f"{r['name']:45s} {r['labels']:30s} {r['precision']:7.4f} "
              f"{r['recall']:7.4f} {r['f1']:7.4f} {r['fpr']:7.4f}")
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
