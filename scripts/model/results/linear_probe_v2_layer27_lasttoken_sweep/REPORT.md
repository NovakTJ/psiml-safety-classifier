# Linear Probe v2 — Hyperparameter Sweep (Attempt 2)

Date: 2026-08-14. CPU-only sweep on the cached teacher-forced Qwen3.5-9B activations from attempt 1 (`linear_probe_v2_layer27_lasttoken/`). No GPU, no model load, no new captures. Feature (last-token residual @ decoder layer 27), input format, train split, and StandardScaler are locked from attempt 1; only the LogisticRegression hyperparameters moved.

## Why

Attempt 1 used guessed hyperparameters (lbfgs, C=1.0, class_weight=balanced) and hit validation F1 0.9097 (P 0.940 / R 0.881) — but train F1 was 1.000 (perfect memorization of 1985 rows in 4096 dims), i.e. clearly under-regularized. The sweep's main knob is therefore C, plus L1-vs-L2 and class_weight (the project prioritizes unsafe recall).

## Grid

- penalty/solver: l2 (lbfgs) over C = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]; l1 (saga) over C = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
- class_weight: none / balanced / {0:1, 1:2}
- total 51 configs, 0 failed, max_iter=5000, seed=42
- selection: max validation F1 at threshold 0.5 (same protocol as attempt 1; ties -> higher recall -> smaller C)
- **test.jsonl untouched** (asserted uncaptured at sweep start).

## Top 10 configs by validation F1 (threshold 0.5)

| # | config | val P | val R | val F1 | val FPR | train F1 | fit s |
|---|--------|------:|------:|-------:|--------:|---------:|------:|
| 1 | l2_lbfgs_C0.03_cw_balanced | 0.947 | 0.900 | **0.9231** | 0.081 | 0.9992 | 3.9 |
| 2 | l2_lbfgs_C0.01_cw_balanced | 0.941 | 0.900 | **0.9201** | 0.091 | 0.9950 | 2.9 |
| 3 | l2_lbfgs_C0.1_cw_none | 0.941 | 0.900 | **0.9201** | 0.091 | 1.0000 | 5.1 |
| 4 | l2_lbfgs_C0.3_cw_harmful_x2 | 0.935 | 0.900 | **0.9172** | 0.101 | 1.0000 | 4.2 |
| 5 | l1_saga_C1_cw_none | 0.941 | 0.894 | **0.9167** | 0.091 | 1.0000 | 420.45 |
| 6 | l2_lbfgs_C0.001_cw_harmful_x2 | 0.879 | 0.956 | **0.9162** | 0.212 | 0.9364 | 2.79 |
| 7 | l2_lbfgs_C0.1_cw_balanced | 0.947 | 0.887 | **0.9161** | 0.081 | 1.0000 | 4.1 |
| 8 | l2_lbfgs_C0.01_cw_none | 0.929 | 0.900 | **0.9143** | 0.111 | 0.9937 | 4.79 |
| 9 | l2_lbfgs_C0.03_cw_none | 0.929 | 0.900 | **0.9143** | 0.111 | 0.9996 | 4.6 |
| 10 | l2_lbfgs_C0.003_cw_harmful_x2 | 0.898 | 0.931 | **0.9141** | 0.172 | 0.9664 | 2.6 |

## Selected config

`l2_lbfgs_C0.03_cw_balanced` — Pipeline(StandardScaler -> LogisticRegression(penalty=l2, solver=lbfgs, C=0.03, class_weight=balanced, max_iter=5000)).

- Validation @0.5: P 0.947 / R 0.900 / F1 0.9231 / FPR 0.081 (confusion {'tp': 144, 'fp': 8, 'fn': 16, 'tn': 91})
- Train @0.5: F1 0.9992 (attempt 1 had train F1 1.000; the selected C no longer fully separates train)
- Nonzero coefficients: 4096/4096

## Decision-threshold tuning (validation)

Scanning thresholds 0.01..0.99 for the selected config: best threshold **0.540** -> P 0.960 / R 0.894 / F1 0.9256 / FPR 0.061. (At the default 0.5: F1 0.9231.) This is the blocking-threshold selection step from the final system design; predictions/metrics above are reported at 0.5 for comparability with attempt 1.

## Comparison on the same validation split (259 rows)

| System | P | R | F1 |
|--------|---:|---:|---:|
| Probe attempt 1 (guessed C=1.0 balanced) | 0.940 | 0.881 | 0.9097 |
| **Probe attempt 2 (swept, @0.5)** | 0.947 | 0.900 | **0.9231** |
| Probe attempt 2 (swept, tuned thr 0.540) | 0.960 | 0.894 | 0.9256 |
| Gemma zero-shot (v2) | 0.776 | 0.910 | 0.838 |
| Qwen3Guard native (v2) | 0.960 | 0.894 | 0.9256 |
| Locked Gemma LoRA (v2) | 0.981 | 0.956 | 0.9684 |

## Notes / limits

- Validation-selected on 259 rows; expect some selection noise. The LoRA ablation already showed a validation winner can drop on test — treat the sweep ranking as noisy beyond the top few configs.
- Only layer 27 / last-token features exist on disk. A layer sweep or token-window pooling needs NEW captures (GPU) — separate experiment.
- Threshold tuning also happens on the same 259 validation rows, so the tuned-threshold F1 is mildly optimistic; the @0.5 number is the honest like-for-like comparison with attempt 1.
- test.jsonl remains uncaptured and unevaluated; the one-shot test eval is a separate future step once the probe recipe is fully locked.

## Recall-oriented threshold analysis (added post-hoc, same winner probabilities)

The tuned threshold above (0.540) is F1-optimal, but the project prioritizes **unsafe recall**, and for the routing use case the probe's FNs are the costly error (they bypass the guard entirely; FPs only cost a guard call). A fine scan (0.01–0.99, step 0.005, recomputed from `predictions_validation.csv`, overwriting the earlier coarse scan) gives the full trade-off — `threshold_scan.csv` now includes `routing_rate` (fraction of traffic with prob ≥ t, i.e. sent to the guard):

| threshold | P | R | F1 | FPR | FN | routing rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0.540 (F1-optimal) | 0.960 | 0.894 | **0.9256** | 0.061 | 17 | 0.575 |
| 0.500 (default) | 0.947 | 0.900 | 0.9231 | 0.081 | 16 | 0.587 |
| 0.300 | 0.906 | 0.906 | 0.906 | 0.152 | 15 | 0.618 |
| 0.150 | 0.874 | **0.950** | 0.910 | 0.222 | 8 | 0.672 |
| 0.100 | 0.842 | 0.9625 | 0.898 | 0.293 | 6 | 0.707 |
| 0.040 | 0.804 | **0.975** | 0.881 | 0.384 | 4 | 0.749 |
| 0.010 | 0.738 | **0.9875** | 0.845 | 0.566 | 2 | 0.826 |

Read: the knee for high recall is **t≈0.10–0.15** (R 0.95–0.9625, FPR 0.22–0.29, ~2/3 of traffic routed). Below that, recall keeps climbing — the earlier coarse scan (min t=0.05) wrongly suggested a hard ceiling at R 0.9625; the fine scan reaches **R 0.9875 (2 FN) at t=0.01** — but the cost is steep (FPR 0.57, 83% routed). Practical max-recall operating points: t=0.15 for R≥0.95, t=0.04 for R≥0.975.

Also confirmed empirically by the sweep grid: **class-weighted loss lands on the same trade-off curve as threshold shifting** (e.g. `l2 C=0.001 cw_harmful_x2` @0.5 → P 0.879 / R 0.956 ≈ winner @ t≈0.15 → P 0.874 / R 0.950; `l1 saga C=0.01 cw_harmful_x2` @0.5 → P 0.780 / R 0.975 ≈ winner @ t=0.04). No reason to prefer weighted training over post-hoc threshold tuning for recall; tune the threshold on validation instead.

Caveats: all operating points are validation-selected on 259 rows (mildly optimistic); the final system design has validation select the probe *routing* threshold separately from the ensemble blocking threshold, so treat this table as the menu for that future decision, not a locked choice.

## Artifacts

- `sweep_results.csv` — all configs, fit stats, val/train metrics
- `probe.joblib` — selected probe (StandardScaler + LogReg pipeline)
- `probe_config.json` — full locked + selected config, thresholds
- `metrics_validation.json`, `predictions_validation.csv`
- `threshold_scan.csv` — fine threshold scan (0.01–0.99 step 0.005) of the winner's validation probabilities, incl. routing_rate
- `validation_inspection.json` — pre-sweep data/feature inspection + attempt-1 error breakdown
- `run.log`
