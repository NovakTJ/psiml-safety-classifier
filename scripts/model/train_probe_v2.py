"""
Train the attempt-1 linear probe on teacher-forced Qwen3.5-9B activations
(last token, layer 27) over the gemma_v2 train split.

Steps:
1. Capture activations for the train split (cache-aware: re-running after a
   kill resumes where it left off; activations are the expensive part and are
   kept so probe hyperparameters can be tweaked later without re-extraction).
2. Train a logistic-regression probe (attempt-1 hyperparameters are GUESSED,
   per instructions — only the pipeline shape matters now).
3. Save probe + config + train metrics.

Run with qwen35_env (only env with transformers 5.x for qwen3_5):

    /home/mls01/.conda/envs/qwen35_env/bin/python scripts/model/train_probe_v2.py
"""

import json
import os
import time

import probe_v2_common as C


def main():
    os.makedirs(C.OUT_DIR, exist_ok=True)
    C.log("=== train_probe_v2 starting ===")

    model, tokenizer = C.load_model_and_tokenizer()
    C.capture_split(model, tokenizer, "train")

    # Free the GPU before the (CPU-only) probe fit, so a kill during training
    # leaves a machine someone else can use and a complete activation cache.
    del model
    import torch
    torch.cuda.empty_cache()
    C.log("Model unloaded; training probe on CPU.")

    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, row_ids, _ = C.load_activation_matrix("train")

    # Attempt-1 hyperparameters: guessed, not tuned. Recorded in probe_config.
    probe_config = {
        "feature": "last-token residual stream, decoder layer 27 "
                   "(second-to-last full-attention layer), bf16->fp32",
        "input_format": "teacher-forced v2 exchange; chat template with "
                        "add_generation_prompt=True, enable_thinking=False; "
                        "response rows get response tokens + <|im_end|> appended",
        "label": "final_label (harmful=1, unharmful=0)",
        "model": {
            "type": "sklearn Pipeline(StandardScaler -> LogisticRegression)",
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": 42,
            "solver": "lbfgs",
        },
        "train_split": "data/gemma_v2_no_refusal/train.jsonl",
        "seed": 42,
        "attempt": 1,
    }
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=probe_config["model"]["C"],
            class_weight=probe_config["model"]["class_weight"],
            max_iter=probe_config["model"]["max_iter"],
            random_state=probe_config["model"]["random_state"],
            solver=probe_config["model"]["solver"])),
    ])

    t0 = time.time()
    clf.fit(X, y)
    fit_s = time.time() - t0
    C.log(f"Probe fit in {fit_s:.1f}s on X{X.shape}")

    # Train-set metrics as a sanity check only (selection happens on
    # validation via eval_probe_v2.py).
    prob = clf.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = {
        "split": "train",
        "n_rows": int(len(y)),
        "n_harmful": int(y.sum()),
        "threshold": 0.5,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "accuracy": float(accuracy_score(y, pred)),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn),
                      "tn": int(tn)},
        "fit_seconds": fit_s,
        "lr_n_iter": int(clf.named_steps["lr"].n_iter_[0]),
    }
    C.log(f"Train metrics (sanity only): {json.dumps(metrics)}")

    joblib.dump(clf, os.path.join(C.OUT_DIR, "probe.joblib"))
    with open(os.path.join(C.OUT_DIR, "probe_config.json"), "w") as f:
        json.dump(probe_config, f, indent=2)
    with open(os.path.join(C.OUT_DIR, "metrics_train.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    C.log(f"Saved probe.joblib, probe_config.json, metrics_train.json "
          f"to {C.OUT_DIR}")
    C.log("=== train_probe_v2 DONE ===")


if __name__ == "__main__":
    main()
