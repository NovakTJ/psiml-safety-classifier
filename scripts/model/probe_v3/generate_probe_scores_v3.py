"""
Generates the probe score files eval_ensemble_v2.py needs (--probe-scores):
CSV with row_id,score for validation + test, score = raw logit (log-odds)
from the LOCKED v3 probe (results/linear_probe_v3_locked/probe.joblib,
layer15_mean_response, C=0.003, class_weight={0:1,1:2}) -- decision_function()
output, not predict_proba(), per eval_ensemble_v2.py's own docstring
recommendation ("logit(p) is the cleaner choice").

No retraining, no re-evaluation of the probe's classification metrics --
those are already locked in lock_probe_v3.py's output. This just re-scores
the same already-captured validation/test activations in continuous form
instead of the hard 0/1 predictions eval_test_v3_candidates.py used.

Run with ccpp_env (CPU-only, needs the already-cached v3 activations):

    /home/mls01/ccpp_env/bin/python scripts/model/probe_v3/generate_probe_scores_v3.py
"""

import csv
import os
import sys
from pathlib import Path

import joblib
import numpy as np

# probe_v2_common.py (teammate's, shared) lives one level up in scripts/model/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import capture_probe_multilayer as CM
import probe_v2_common as C

LOCKED_DIR = "/home/mls01/scripts/model/results/linear_probe_v3_locked"
V3_CAPTURE_DIR = CM.DEFAULT_OUT_DIR
FEATURE_KEY = "layer15_mean_response"


def load_split_feature(split, key):
    rows = C.load_rows(split)
    act_dir = CM.act_dir_for(V3_CAPTURE_DIR, split)
    X = np.empty((len(rows), CM.HIDDEN_SIZE), dtype=np.float32)
    row_ids = []
    for i, r in enumerate(rows):
        d = np.load(os.path.join(act_dir, f"{r['row_id']}.npz"))
        X[i] = d[key].astype(np.float32)
        row_ids.append(r["row_id"])
    return X, row_ids


def main():
    clf = joblib.load(os.path.join(LOCKED_DIR, "probe.joblib"))

    for split in ("validation", "test"):
        n_done = len(CM.done_row_ids(V3_CAPTURE_DIR, split))
        n_rows = len(C.load_rows(split))
        assert n_done == n_rows, f"{split}: {n_done}/{n_rows} activations cached"

        X, row_ids = load_split_feature(split, FEATURE_KEY)
        scores = clf.decision_function(X)  # raw logit, higher = more harmful

        out_path = os.path.join(LOCKED_DIR, f"scores_{split}.csv")
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row_id", "score"])
            for rid, s in zip(row_ids, scores):
                w.writerow([rid, float(s)])
        print(f"[{split}] {len(row_ids)} rows -> {out_path} "
              f"(score range [{scores.min():.3f}, {scores.max():.3f}])")


if __name__ == "__main__":
    main()
