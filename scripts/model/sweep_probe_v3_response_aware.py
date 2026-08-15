"""
Follow-up to sweep_probe_v3_layer_pooling.py, after eval_switched_prompt_probes.py
showed its winner (layer15_mean_prompt, val F1 0.9779) catches only 1/51
(recall 0.0196) benign-prompt+harmful-response exchanges -- vs.
layer11_mean_response's 43/51 (0.8431) using only the phase-1 fixed config.
gemma_v2_no_refusal validation F1 rewards prompt-only behavior (final_label
== prompt_harm_label on every row -- see CLAUDE.md), so selecting by F1
alone silently throws away response sensitivity. This sweep restricts the
search to mean_response pooling (the pooling family that actually looks at
the response) and adds switched-prompt recall as a SECOND selection axis.

Grid: mean_response pooling x all 8 full-attention layers x L2/lbfgs
C in {0.003,0.01,0.03,0.1,0.3,1.0} x class_weight in {none, balanced,
harmful_x2} = 144 fits, all on the ALREADY-CACHED activations (train from
capture_probe_multilayer.py, switched-prompt captured fresh here once and
cached to disk for reuse). CPU-only after the one-time switched-prompt
capture.

Selection: combined_score = 0.5*val_f1(threshold 0.5) + 0.5*switched_recall,
equal weight since neither axis alone is trustworthy (val F1 is
prompt-confounded; switched recall is only 51 rows, all positive class, no
precision signal). Both axes reported in full for every config so the
tradeoff is visible, not hidden behind one number.

test.jsonl is NOT touched by this script.

Run with ccpp_env for a cache-only re-run, or qwen35_env if the
switched-prompt activation cache does not exist yet (first run needs the
model):

    /home/mls01/.conda/envs/qwen35_env/bin/python \\
        scripts/model/sweep_probe_v3_response_aware.py
"""

import csv
import json
import os
import time
import warnings

import numpy as np

import capture_probe_multilayer as CM
import probe_v2_common as C

OUT_DIR = ("/home/mls01/scripts/model/results/"
           "linear_probe_v3_multilayer_pooling_sweep_response_aware")
V3_CAPTURE_DIR = CM.DEFAULT_OUT_DIR
SWITCHED_DATASET_PATH = "/home/mls01/data/switched_prompt_dataset.jsonl"
SWITCHED_CACHE_DIR = os.path.join(V3_CAPTURE_DIR, "activations", "switched_prompt")

SEED = 42
MAX_ITER = 5000
POOLING = "mean_response"
CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
CLASS_WEIGHTS = [
    ("none", None),
    ("balanced", "balanced"),
    ("harmful_x2", {0: 1.0, 1: 2.0}),
]
FEATURE_KEYS = [f"layer{L}_{POOLING}" for L in CM.CAPTURE_LAYERS]


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
        "accuracy": float(accuracy_score(y, pred)),
    }


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


def load_switched_rows():
    rows = [json.loads(l) for l in open(SWITCHED_DATASET_PATH) if l.strip()]
    for r in rows:
        assert r["prompt_harm_label"] == "unharmful"
        assert r["response_harm_label"] == "harmful"
        r["final_label"] = "harmful"
    return rows


def ensure_switched_cache():
    """Cache-aware: capture switched-prompt activations to disk once (GPU),
    reused by every future run of this script (CPU-only after that)."""
    rows = load_switched_rows()
    os.makedirs(SWITCHED_CACHE_DIR, exist_ok=True)
    done = {os.path.splitext(f)[0] for f in os.listdir(SWITCHED_CACHE_DIR)
            if f.endswith(".npz")}
    todo = [r for r in rows if r["row_id"] not in done]
    if not todo:
        log(f"switched-prompt cache complete ({len(rows)} rows), skipping GPU capture")
        return

    log(f"switched-prompt cache: {len(done)}/{len(rows)} cached, "
        f"capturing {len(todo)} on GPU ...")
    import torch  # noqa: E402 (deferred: only needed on the GPU path)

    model, tokenizer = CM.load_model_and_tokenizer(log)
    span_by_idx = {}
    for i, r in enumerate(todo):
        ids, n_prompt, n_response = CM.build_input_ids_with_spans(
            tokenizer, r["prompt"], r.get("response") or "")
        span_by_idx[i] = (ids, n_prompt, n_response)
    batches = C.plan_batches([(i, len(span_by_idx[i][0])) for i in range(len(todo))])

    pad_id = tokenizer.pad_token_id
    for b_idx, batch in enumerate(batches):
        seqs = [span_by_idx[i][0] for i in batch]
        max_len = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for j, s in enumerate(seqs):
            input_ids[j, max_len - len(s):] = torch.tensor(s)
            attn[j, max_len - len(s):] = 1
        input_ids = input_ids.to(model.device)
        attn = attn.to(model.device)
        with torch.inference_mode():
            try:
                out = model(input_ids=input_ids, attention_mask=attn,
                            output_hidden_states=True, logits_to_keep=1)
            except TypeError:
                out = model(input_ids=input_ids, attention_mask=attn,
                            output_hidden_states=True)
        hs_by_layer = {
            L: out.hidden_states[L + 1].to(torch.float16).cpu().numpy()
            for L in CM.CAPTURE_LAYERS
        }
        del out
        for j, i in enumerate(batch):
            r = todo[i]
            ids, n_prompt, n_response = span_by_idx[i]
            L_j = len(ids)
            start = max_len - L_j
            vecs = CM.pool_row(hs_by_layer, j, start, max_len, n_prompt, n_response)
            tmp = os.path.join(SWITCHED_CACHE_DIR, f".{r['row_id']}.tmp")
            final = os.path.join(SWITCHED_CACHE_DIR, f"{r['row_id']}.npz")
            with open(tmp, "wb") as vf:
                np.savez(vf, **vecs)
            os.replace(tmp, final)
        del hs_by_layer, input_ids, attn
        log(f"switched-prompt capture batch {b_idx+1}/{len(batches)} done")

    del model
    torch.cuda.empty_cache()
    log("switched-prompt capture complete; model unloaded")


def load_switched_feature(key):
    rows = load_switched_rows()
    X = np.empty((len(rows), CM.HIDDEN_SIZE), dtype=np.float32)
    for i, r in enumerate(rows):
        d = np.load(os.path.join(SWITCHED_CACHE_DIR, f"{r['row_id']}.npz"))
        X[i] = d[key].astype(np.float32)
    y = np.ones(len(rows), dtype=np.int64)  # all harmful by construction
    return X, y


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log("=== sweep_probe_v3_response_aware starting ===")

    for split in ("train", "validation"):
        n_done = len(CM.done_row_ids(V3_CAPTURE_DIR, split))
        n_rows = len(C.load_rows(split))
        assert n_done == n_rows, f"{split}: {n_done}/{n_rows} cached"
    assert len(CM.done_row_ids(V3_CAPTURE_DIR, "test")) == 0, \
        "test activations exist — this sweep must not touch test"

    ensure_switched_cache()

    log(f"Grid: {len(FEATURE_KEYS)} layers x {len(CS)} Cs x "
        f"{len(CLASS_WEIGHTS)} class weights = "
        f"{len(FEATURE_KEYS)*len(CS)*len(CLASS_WEIGHTS)} fits, pooling={POOLING}")

    results = []
    for key in FEATURE_KEYS:
        X_train, y_train = load_split_feature("train", key)
        X_val, y_val = load_split_feature("validation", key)
        X_switch, y_switch = load_switched_feature(key)
        for C_val in CS:
            for cw_name, cw in CLASS_WEIGHTS:
                clf, fit_s, converged = fit_one(C_val, cw, X_train, y_train)
                val_prob = clf.predict_proba(X_val)[:, 1]
                val_m = metrics_at(y_val, val_prob, 0.5)
                switch_prob = clf.predict_proba(X_switch)[:, 1]
                switch_recall = float((switch_prob >= 0.5).mean())
                combined = 0.5 * val_m["f1"] + 0.5 * switch_recall
                config_id = f"{key}__C{C_val:g}_cw_{cw_name}"
                results.append({
                    "config_id": config_id, "feature_key": key, "C": C_val,
                    "class_weight": cw_name, "val_precision": val_m["precision"],
                    "val_recall": val_m["recall"], "val_f1": val_m["f1"],
                    "val_fpr": val_m["fpr"], "switched_recall": switch_recall,
                    "combined_score": combined, "fit_seconds": fit_s,
                    "converged": converged,
                })
        log(f"[{key}] {len(CS)*len(CLASS_WEIGHTS)} configs done")

    results.sort(key=lambda r: r["combined_score"], reverse=True)
    with open(os.path.join(OUT_DIR, "grid_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config_id", "feature_key", "C", "class_weight",
                    "val_precision", "val_recall", "val_f1", "val_fpr",
                    "switched_recall", "combined_score", "fit_seconds",
                    "converged"])
        for r in results:
            w.writerow([r["config_id"], r["feature_key"], r["C"], r["class_weight"],
                        r["val_precision"], r["val_recall"], r["val_f1"],
                        r["val_fpr"], r["switched_recall"], r["combined_score"],
                        r["fit_seconds"], r["converged"]])

    winner = results[0]
    log(f"Winner (by combined_score): {winner['config_id']} "
        f"val_f1={winner['val_f1']:.4f} switched_recall={winner['switched_recall']:.4f} "
        f"combined={winner['combined_score']:.4f}")

    by_val_f1 = sorted(results, key=lambda r: r["val_f1"], reverse=True)[0]
    by_switch = sorted(results, key=lambda r: r["switched_recall"], reverse=True)[0]

    lines = [
        "# Linear probe v3 -- response-aware sweep (mean_response pooling only)",
        "",
        "Motivation: sweep_probe_v3_layer_pooling.py's winner (layer15_mean_prompt) "
        "scored val F1 0.9779 but recall 0.0196 (1/51) on the switched-prompt "
        "stress test (benign prompt + harmful response). This sweep restricts "
        f"to `{POOLING}` pooling (looks at the response) across all 8 layers, "
        "and selects by combined_score = 0.5*val_f1 + 0.5*switched_recall "
        "instead of val_f1 alone.",
        "",
        f"{len(results)} fits total.",
        "",
        "## Top 10 by combined score",
        "",
        "| Rank | config_id | val_precision | val_recall | val_f1 | switched_recall | combined |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(results[:10]):
        lines.append(f"| {i+1} | {r['config_id']} | {r['val_precision']:.4f} | "
                     f"{r['val_recall']:.4f} | {r['val_f1']:.4f} | "
                     f"{r['switched_recall']:.4f} | {r['combined_score']:.4f} |")
    lines += [
        "",
        f"## Winner: `{winner['config_id']}`",
        f"- val: P={winner['val_precision']:.4f} R={winner['val_recall']:.4f} "
        f"F1={winner['val_f1']:.4f}",
        f"- switched-prompt recall: {winner['switched_recall']:.4f}",
        "",
        f"For reference -- best by val_f1 alone: `{by_val_f1['config_id']}` "
        f"(val_f1={by_val_f1['val_f1']:.4f}, switched_recall="
        f"{by_val_f1['switched_recall']:.4f}); best by switched_recall alone: "
        f"`{by_switch['config_id']}` (val_f1={by_switch['val_f1']:.4f}, "
        f"switched_recall={by_switch['switched_recall']:.4f}).",
        "",
        "test.jsonl was NOT captured or evaluated in this run.",
    ]
    with open(os.path.join(OUT_DIR, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    log(f"Saved grid_results.csv, REPORT.md to {OUT_DIR}")
    log("=== sweep_probe_v3_response_aware DONE ===")


if __name__ == "__main__":
    main()
