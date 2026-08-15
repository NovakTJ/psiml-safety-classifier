"""
Locks the v3 linear-probe investigation's final winner as the new probe
baseline, superseding attempt-2's layer27_last (results/
linear_probe_v2_layer27_lasttoken_sweep/), which is kept untouched as
historical record.

Winner: layer15_mean_response, C=0.003, class_weight={0:1.0, 1:2.0}
("harmful_x2"), selected by sweep_probe_v3_response_aware.py's
combined_score = 0.5*val_f1 + 0.5*switched_prompt_recall (NOT val_f1 alone
-- see that script's docstring for why: val F1 alone rewards prompt-only
behavior on gemma_v2_no_refusal, since final_label == prompt_harm_label on
every row there).

Journey (full detail in the three prior REPORT.md files):
  1. capture_probe_multilayer.py: one GPU pass captures 8 full-attention
     layers x 4 poolings (32 features/row) instead of attempt-1/2's single
     layer27/last-token.
  2. sweep_probe_v3_layer_pooling.py (phase1+2, F1-only selection): won by
     layer15_mean_prompt, val F1 0.9779 -- but recall 0.0196 (1/51) on
     data/switched_prompt_dataset.jsonl (benign prompt + harmful response),
     i.e. a prompt-only classifier in disguise. NOT locked.
  3. sweep_probe_v3_response_aware.py: restricted to mean_response pooling,
     added switched-prompt recall as a second selection axis. Won by
     layer15_mean_response (this file's winner).
  4. eval_test_v3_candidates.py: one-shot test.jsonl (227 rows) evaluation,
     confirmed the win holds on real held-out data (not just validation
     noise): F1 0.9091 vs attempt-2's 0.8528 on the same test rows.

Run with ccpp_env (CPU-only, needs the already-cached v3 train activations):

    /home/mls01/ccpp_env/bin/python scripts/model/lock_probe_v3.py
"""

import json
import os

import joblib
import numpy as np

import capture_probe_multilayer as CM
import probe_v2_common as C

OUT_DIR = "/home/mls01/scripts/model/results/linear_probe_v3_locked"
V3_CAPTURE_DIR = CM.DEFAULT_OUT_DIR

FEATURE_KEY = "layer15_mean_response"
C_VAL = 0.003
CLASS_WEIGHT = {0: 1.0, 1: 2.0}
SEED = 42

# Recorded from prior runs (not recomputed here -- see docstring for source).
VAL_METRICS = {"precision": 0.9107142857142857, "recall": 0.95625,
              "f1": 0.9329268292682927, "fpr": 0.15151515151515152}
SWITCHED_RECALL = 1.0
TEST_METRICS = {"precision": 0.8446, "recall": 0.9843, "f1": 0.9091, "fpr": 0.2300}
BASELINE_TEST_METRICS = {  # attempt-2 layer27_last, same 227 test rows
    "precision": 0.8188, "recall": 0.8898, "f1": 0.8528, "fpr": 0.2500}


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


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    os.makedirs(OUT_DIR, exist_ok=True)

    X_train, y_train = load_split_feature("train", FEATURE_KEY)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C_VAL, class_weight=CLASS_WEIGHT,
                                  max_iter=5000, random_state=SEED)),
    ])
    clf.fit(X_train, y_train)
    joblib.dump(clf, os.path.join(OUT_DIR, "probe.joblib"))

    probe_config = {
        "feature": "mean-pooled residual stream over RESPONSE tokens "
                   "(incl. trailing <|im_end|>), decoder layer 15, bf16->fp32. "
                   "For empty-response rows (~58% of v2), falls back to the "
                   "last-token vector (see capture_probe_multilayer.py "
                   "pool_row()).",
        "layer": 15,
        "pooling": "mean_response",
        "input_format": "teacher-forced v2 exchange; chat template with "
                        "add_generation_prompt=True, enable_thinking=False; "
                        "response rows get response tokens + <|im_end|> appended "
                        "(same as attempt-1/2 -- locked from probe_v2_common.py)",
        "label": "final_label (harmful=1, unharmful=0)",
        "model": {
            "type": "sklearn Pipeline(StandardScaler -> LogisticRegression)",
            "C": C_VAL, "class_weight": CLASS_WEIGHT, "max_iter": 5000,
            "random_state": SEED, "solver": "lbfgs", "penalty": "l2",
        },
        "selection_method": "sweep_probe_v3_response_aware.py, "
                            "combined_score = 0.5*val_f1(threshold 0.5) + "
                            "0.5*switched_prompt_recall -- NOT val_f1 alone "
                            "(see docstring)",
        "supersedes": "results/linear_probe_v2_layer27_lasttoken_sweep/ "
                      "(layer27_last, attempt-2, kept untouched as reference)",
        "train_split": "data/gemma_v2_no_refusal/train.jsonl",
        "validation_metrics_threshold_0.5": VAL_METRICS,
        "switched_prompt_recall": SWITCHED_RECALL,
        "test_metrics_threshold_0.5": TEST_METRICS,
        "baseline_test_metrics_layer27_last": BASELINE_TEST_METRICS,
    }
    with open(os.path.join(OUT_DIR, "probe_config.json"), "w") as f:
        json.dump(probe_config, f, indent=2)

    lines = [
        "# Linear probe v3 -- LOCKED",
        "",
        f"**Winner: `{FEATURE_KEY}`, C={C_VAL}, class_weight={CLASS_WEIGHT}`** "
        "-- supersedes attempt-2's layer27_last as the project's probe baseline.",
        "",
        "## Why this and not the higher-F1 layer15_mean_prompt",
        "",
        "sweep_probe_v3_layer_pooling.py's F1-only sweep picked "
        "layer15_mean_prompt (val F1 0.9779) -- but that pooling ignores the "
        "response entirely, and gemma_v2_no_refusal's final_label collapses "
        "to prompt_harm_label on every row, so val F1 alone can't tell a "
        "true exchange classifier from a prompt-only one in disguise. "
        "Confirmed on data/switched_prompt_dataset.jsonl (51 benign-prompt + "
        "harmful-response rows): layer15_mean_prompt recall = **0.0196** "
        "(1/51). layer15_mean_response, selected with switched-prompt recall "
        "as a second axis, scores **1.0** (51/51) on the same stress test.",
        "",
        "## Validation (threshold 0.5, 259 rows)",
        f"P={VAL_METRICS['precision']:.4f} R={VAL_METRICS['recall']:.4f} "
        f"F1={VAL_METRICS['f1']:.4f} FPR={VAL_METRICS['fpr']:.4f}",
        "",
        "## Switched-prompt stress test (51 rows, all harmful by construction)",
        f"recall = {SWITCHED_RECALL:.4f} (51/51)",
        "",
        "## Test (threshold 0.5, 227 rows, one-shot, first time test.jsonl "
        "was touched in the probe experiment)",
        "",
        "| System | precision | recall | F1 | FPR |",
        "|---|---:|---:|---:|---:|",
        f"| **layer15_mean_response (this, locked)** | "
        f"{TEST_METRICS['precision']:.4f} | {TEST_METRICS['recall']:.4f} | "
        f"{TEST_METRICS['f1']:.4f} | {TEST_METRICS['fpr']:.4f} |",
        f"| layer27_last (attempt-2, old baseline) | "
        f"{BASELINE_TEST_METRICS['precision']:.4f} | "
        f"{BASELINE_TEST_METRICS['recall']:.4f} | "
        f"{BASELINE_TEST_METRICS['f1']:.4f} | "
        f"{BASELINE_TEST_METRICS['fpr']:.4f} |",
        "",
        f"F1 delta vs old baseline: "
        f"{TEST_METRICS['f1']-BASELINE_TEST_METRICS['f1']:+.4f}",
        "",
        "## Journey",
        "",
        "1. `capture_probe_multilayer.py` -- one GPU pass, 8 layers x 4 "
        "poolings instead of attempt-1/2's single layer27/last-token "
        "(results/linear_probe_v3_multilayer_pooling/).",
        "2. `sweep_probe_v3_layer_pooling.py` -- F1-only phase1+2 sweep, won "
        "by layer15_mean_prompt (val F1 0.9779), rejected after the "
        "switched-prompt check (results/"
        "linear_probe_v3_multilayer_pooling_sweep/).",
        "3. `sweep_probe_v3_response_aware.py` -- mean_response-only grid, "
        "combined_score selection, won by this config (results/"
        "linear_probe_v3_multilayer_pooling_sweep_response_aware/).",
        "4. `eval_test_v3_candidates.py` -- one-shot test.jsonl evaluation, "
        "confirmed the win on held-out data (results/"
        "linear_probe_v3_multilayer_pooling_sweep_response_aware/"
        "test_evaluation/).",
        "5. This file: refit on the full train split, saved as the locked "
        "artifact.",
        "",
        "## Files",
        "- `probe.joblib` -- sklearn Pipeline(StandardScaler, LogisticRegression), "
        "fit on data/gemma_v2_no_refusal/train.jsonl",
        "- `probe_config.json` -- full config + all metrics above, machine-readable",
    ]
    with open(os.path.join(OUT_DIR, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Locked probe saved to {OUT_DIR}")
    print(f"  feature={FEATURE_KEY} C={C_VAL} class_weight={CLASS_WEIGHT}")
    print(f"  val F1={VAL_METRICS['f1']:.4f}  switched_recall={SWITCHED_RECALL:.4f}  "
          f"test F1={TEST_METRICS['f1']:.4f}")


if __name__ == "__main__":
    main()
