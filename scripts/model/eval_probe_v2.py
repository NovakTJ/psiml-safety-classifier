"""
Evaluate the trained linear probe on a held-out v2 split.

Same protocol as training: teacher-forced capture of last-token layer-27
activations for the requested split (cache-aware — if train_probe_v2.py or a
previous eval run already captured the split, this is instant), then probe
inference + metrics.

Usage (qwen35_env):

    /home/mls01/.conda/envs/qwen35_env/bin/python scripts/model/eval_probe_v2.py \
        --split validation          # default
    ... --split test                # one-shot held-out test
    ... --split all                 # validation then test

Writes predictions_{split}.csv + metrics_{split}.json under the experiment dir.
"""

import argparse
import csv
import json
import os

import probe_v2_common as C


def evaluate_split(clf, split, threshold=0.5):
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score)

    X, y, row_ids, rows = C.load_activation_matrix(split)
    prob = clf.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = {
        "split": split,
        "n_rows": int(len(y)),
        "n_harmful": int(y.sum()),
        "threshold": threshold,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else None,
        "accuracy": float(accuracy_score(y, pred)),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn),
                      "tn": int(tn)},
    }
    C.log(f"[{split}] metrics: {json.dumps(metrics)}")

    csv_path = os.path.join(C.OUT_DIR, f"predictions_{split}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "final_label", "prob_harmful", "pred_label",
                    "correct"])
        for rid, yi, pi, di in zip(row_ids, y, prob, pred):
            w.writerow([rid, "harmful" if yi else "unharmful",
                        f"{pi:.6f}", "harmful" if di else "unharmful",
                        int(yi == di)])
    with open(os.path.join(C.OUT_DIR, f"metrics_{split}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    C.log(f"[{split}] wrote {csv_path} and metrics_{split}.json")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation",
                    choices=["validation", "test", "all"])
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    splits = ["validation", "test"] if args.split == "all" else [args.split]

    os.makedirs(C.OUT_DIR, exist_ok=True)
    C.log(f"=== eval_probe_v2 starting (splits={splits}, "
          f"threshold={args.threshold}) ===")

    import joblib
    probe_path = os.path.join(C.OUT_DIR, "probe.joblib")
    assert os.path.exists(probe_path), \
        f"no trained probe at {probe_path} — run train_probe_v2.py first"
    clf = joblib.load(probe_path)
    C.log(f"Loaded probe from {probe_path}")

    # Capture any missing activations for the requested splits.
    todo = [s for s in splits
            if len(C.done_row_ids(s)) < len(C.load_rows(s))]
    if todo:
        model, tokenizer = C.load_model_and_tokenizer()
        for s in todo:
            C.capture_split(model, tokenizer, s)
        del model
        import torch
        torch.cuda.empty_cache()
        C.log("Model unloaded.")
    else:
        C.log("All requested splits fully cached; no GPU work needed.")

    for s in splits:
        evaluate_split(clf, s, threshold=args.threshold)
    C.log("=== eval_probe_v2 DONE ===")


if __name__ == "__main__":
    main()
