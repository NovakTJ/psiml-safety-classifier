"""
Stacked ensemble: instead of the fixed 2-signal cascade formula in
eval_ensemble_v2.py (probe gates, alpha-blend with ONE classifier), fit a
multi-feature logistic regression on VALIDATION over probe + all 4
LoRA/DoRA ablation classifier scores (5 raw logits total), letting the
model learn how much to trust each signal instead of a manual alpha/tau
sweep on a hand-picked pair. This is also a direct answer to "would better
calibration help": a multi-feature LogisticRegression IS joint calibration
+ combination (each raw logit gets its own learned weight, on the probability
scale) -- generally stronger than calibrating each source independently
(Platt/isotonic) and then hand-blending with one alpha.

VALIDATION ONLY. Uses 5-fold stratified CV (out-of-fold predictions) to
pick the regularization strength C and to get a less-optimistic F1 estimate
than a plain in-sample fit would give -- 259 rows x up to 5 features is
small enough that in-sample fit alone risks overfitting the stacker itself,
which the rest of this project's simpler (1-2 parameter) selections didn't
need to worry about as much.

Does NOT touch test.jsonl. Prints CV-estimated F1 for several feature
subsets (single best classifier, all 4 classifiers, all 4 + probe) so an
informed decision can be made about whether stacking is worth a further
test look, instead of assuming more signals automatically helps.

Run with ccpp_env (CPU-only):

    /home/mls01/ccpp_env/bin/python scripts/model/stack_ensemble_v2.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = "/home/mls01/scripts/model"
DATA_DIR = "/home/mls01/data/gemma_v2_no_refusal"

SCORE_SOURCES = {
    "probe": f"{SCRIPT_DIR}/results/linear_probe_v3_locked/scores_validation.csv",
    "lora_all_linear": f"{SCRIPT_DIR}/results/gemma_lora_v2_logit_scores/validation_scores.csv",
    "lora_attention_only": f"{SCRIPT_DIR}/results/gemma_lora_v2_logit_scores_attention_only/validation_scores.csv",
    "dora_attention_only": f"{SCRIPT_DIR}/results/gemma_lora_v2_logit_scores_dora_attention_only/validation_scores.csv",
    "dora_all_linear": f"{SCRIPT_DIR}/results/gemma_lora_v2_logit_scores_dora_all_linear/validation_scores.csv",
}
C_GRID = [0.003, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8,
         1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0]
SEED = 42


def load_labels():
    rows = [json.loads(l) for l in open(f"{DATA_DIR}/validation.jsonl") if l.strip()]
    df = pd.DataFrame(rows)[["row_id", "final_label"]]
    df["y"] = (df["final_label"] == "harmful").astype(int)
    return df


def load_features(labels):
    df = labels[["row_id", "y"]].copy()
    for name, path in SCORE_SOURCES.items():
        s = pd.read_csv(path)[["row_id", "score"]].rename(columns={"score": name})
        df = df.merge(s, on="row_id", how="left")
        assert df[name].isna().sum() == 0, f"{name}: missing rows"
    return df


def best_threshold_f1(y, prob):
    best = (0.5, -1)
    for t in np.arange(0.01, 1.0, 0.005):
        f1 = f1_score(y, (prob >= t).astype(int), zero_division=0)
        if f1 > best[1]:
            best = (float(t), f1)
    return best


def cv_evaluate(X, y, feature_names, label, poly_degree=1, interaction_only=False):
    from sklearn.preprocessing import PolynomialFeatures

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_c, best_cv_f1, best_oof = None, -1, None
    for c in C_GRID:
        pipe_prob = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            scaler = StandardScaler().fit(X[tr])
            Xtr_s, Xte_s = scaler.transform(X[tr]), scaler.transform(X[te])
            if poly_degree > 1:
                poly = PolynomialFeatures(degree=poly_degree, include_bias=False,
                                          interaction_only=interaction_only).fit(Xtr_s)
                Xtr_s, Xte_s = poly.transform(Xtr_s), poly.transform(Xte_s)
            lr = LogisticRegression(C=c, max_iter=2000, random_state=SEED)
            lr.fit(Xtr_s, y[tr])
            pipe_prob[te] = lr.predict_proba(Xte_s)[:, 1]
        thr, f1 = best_threshold_f1(y, pipe_prob)
        if f1 > best_cv_f1:
            best_c, best_cv_f1, best_oof = c, f1, pipe_prob
    thr, f1 = best_threshold_f1(y, best_oof)
    pred = (best_oof >= thr).astype(int)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    print(f"[{label}] features={feature_names} poly_degree={poly_degree} "
          f"interaction_only={interaction_only} best_C={best_c} "
          f"CV-out-of-fold: threshold={thr:.3f} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
    return {"label": label, "features": feature_names, "poly_degree": poly_degree,
            "interaction_only": interaction_only, "C": best_c,
            "threshold": thr, "precision": prec, "recall": rec, "f1": f1}


def main():
    labels = load_labels()
    df = load_features(labels)
    y = df["y"].to_numpy()
    print(f"validation: {len(df)} rows ({int(y.sum())} harmful / {int((1-y).sum())} unharmful)\n")

    results = []

    # Reference: current locked 2-signal cascade result (in-sample, not CV -- for context only).
    print("=== Reference (already known, in-sample selection like the rest of the "
          "project, NOT cross-validated) ===")
    print("ensemble_v2_final_maxf1 (probe + lora_all_linear cascade): val F1=0.9843\n")

    print("=== Stacked logistic regression, 5-fold CV (out-of-fold), various feature sets ===")
    feature_sets = [
        ("probe_only", ["probe"]),
        ("best_single_classifier", ["probe", "lora_all_linear"]),
        ("probe_+_all_4_classifiers", ["probe", "lora_all_linear", "lora_attention_only",
                                       "dora_attention_only", "dora_all_linear"]),
        ("all_4_classifiers_no_probe", ["lora_all_linear", "lora_attention_only",
                                        "dora_attention_only", "dora_all_linear"]),
    ]
    for label, feats in feature_sets:
        X = df[feats].to_numpy(dtype=float)
        r = cv_evaluate(X, y, feats, label)
        results.append(r)

    print("\n=== Interaction terms (PolynomialFeatures), finer C grid "
         f"({len(C_GRID)} values) ===")
    interaction_sets = [
        ("probe_x_lora__interaction_only", ["probe", "lora_all_linear"], 2, True),
        ("probe_x_lora__full_degree2", ["probe", "lora_all_linear"], 2, False),
        ("probe_x_lora_ao__interaction_only",
         ["probe", "lora_all_linear", "lora_attention_only"], 2, True),
        ("probe_x_all4__interaction_only",
         ["probe", "lora_all_linear", "lora_attention_only",
          "dora_attention_only", "dora_all_linear"], 2, True),
    ]
    for label, feats, deg, inter_only in interaction_sets:
        X = df[feats].to_numpy(dtype=float)
        r = cv_evaluate(X, y, feats, label, poly_degree=deg, interaction_only=inter_only)
        results.append(r)

    print()
    best = max(results, key=lambda r: r["f1"])
    print(f"Best CV config: {best['label']} (F1={best['f1']:.4f})")

    out_dir = f"{SCRIPT_DIR}/results/ensemble_v2_stacked_cv"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/cv_results.json", "w") as f:
        json.dump(results, f, indent=2)
    df.to_csv(f"{out_dir}/validation_features.csv", index=False)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
