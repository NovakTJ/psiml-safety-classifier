# Ensemble (probe + finetuned classifier) evaluation -- v2 data

- generated: 2026-08-14 23:09:22 UTC by `eval_ensemble_v2.py` v1.0.0
- probe scores: `results/linear_probe_v3_locked/scores_combined.csv`
- classifier scores: `results/gemma_lora_v2_logit_scores/scores_combined.csv`
- heuristic: probe gate `p_probe > tau_route` -> else ALLOW; on routed rows `p_final = alpha*p_probe + (1-alpha)*p_clf`, BLOCK iff `p_final > tau_block`. Both scores Platt-calibrated to P(harmful) on validation.

## Config selection (validation only)

- grid: 141 tau_route x 10 alpha x 19 tau_block = 26790 configs (full table: `sweep_results.csv`)
- objective requested: `constrained_cost`; used: `constrained_cost`
  - constrained_cost keeps recall >= 0.95 and FPR <= 0.15, then minimizes r(pi=0.05)
| parameter | value |
|---|---|
| tau_route (probe routing threshold) | 0.4358 |
| alpha (probe weight in blend) | 0.50 |
| tau_block (blocking threshold) | 0.5000 |

| metric (validation) | value | Wilson 95% CI |
|---|---|---|
| unsafe recall R = r_H*b_H | 0.9500 | [0.904, 0.974] |
| benign FPR = r_B*b_B | 0.0202 | [0.006, 0.071] |
| precision (val pi=0.62!) | 0.9870 |  |
| F1 | 0.9682 |  |
| routing rate r (val) | 0.6332 |  |
| r_H = P(fire|harmful) | 0.9500 | [0.904, 0.974] |
| r_B = P(fire|benign) | 0.1212 | [0.071, 0.200] |
| b_H = P(block|harmful,fired) | 1.0000 |  |
| b_B = P(block|benign,fired) | 0.1667 |  |
| confusion (tp/fp/fn/tn) | 152/2/8/97 |  |

## Cost model -- the plug-in formula

Per exchange, in units of one classifier call (kappa = c_probe / c_clf, realistically ~1e-4..1e-3):

```
C_always_on = 1
C_ensemble  = kappa + r(pi)
r(pi)       = pi*r_H + (1-pi)*r_B          # routing rate for deployment
                                                prevalence pi
recall      = r_H * b_H                    # exact (blocked => fired)
FPR         = r_B * b_B                    # probe FPR multiplied DOWN by clf
precision(pi) = pi*R / (pi*R + (1-pi)*FPR)
savings     = 1 - kappa - r(pi)            # vs always-on classifier
break-even  : ensemble cheaper iff r(pi) < 1 - kappa
```

Measured stage rates on validation: r_H=0.9500 [0.904, 0.974], r_B=0.1212 [0.071, 0.200], b_H=1.0000, b_B=0.1667.
These four numbers are prevalence-independent -- when a representative traffic sample exists (WildBench-like), plug its harmful prevalence pi into the table below (or measure r directly on that trace). The v2 val split is ~62% harmful, so the raw val routing rate overstates deployment cost.

| pi (harmful prev.) | routing rate r(pi) | classifier-free frac | precision(pi) | rel. cost k=0.0001 | rel. cost k=0.001 | rel. cost k=0.01 |
|---|---|---|---|---|---|---|
| 0.001 | 0.1220 | 0.8780 | 0.0450 | 0.1221 | 0.1230 | 0.1320 |
| 0.01 | 0.1295 | 0.8705 | 0.3220 | 0.1296 | 0.1305 | 0.1395 |
| 0.05 | 0.1627 | 0.8373 | 0.7122 | 0.1628 | 0.1637 | 0.1727 |
| 0.2 | 0.2870 | 0.7130 | 0.9216 | 0.2871 | 0.2880 | 0.2970 |

(`deployment_cost_table.csv`, `cost_analysis.json`)

## How to read / caveats

- **recall ceiling**: overall recall = r_H*b_H <= r_H. tau_route must be set for routing recall *above* the recall target, leaving room for the classifier's conditional miss rate (b_H).
- **the classifier's job is precision**: FPR = r_B*b_B multiplies the probe's benign fire rate down by the classifier's conditional benign block rate.
- **deployment precision** collapses at small pi unless FPR is tiny -- that multiplicative FPR reduction is the cascade's main benefit, not recall.
- per-exchange view: at deployment the probe scores a ~16-token sliding window mid-stream; per-response fire prob = 1 - prod(1 - p_window) over windows (grows with length). Convert when window-level data exists.
- calibration and selection both happen on validation (259 rows) -> mild selection optimism; Wilson CIs above quantify the binomial noise.
- v2 dataset property: `final_label == prompt_harm_label` on every row, so response-side detection (the "$x example") stays untested here -- the ensemble inherits that gap from its components' training/eval data.

## Test

Not run. `test.jsonl` untouched. When real score files exist and the config above is reviewed, run once with `--run-test`.

## Next steps to make this real

1. probe scores: dump winner-probe logits per row (exists for validation in `linear_probe_v2_layer27_lasttoken_sweep/predictions_validation.csv` as `prob_harmful`; test activations not captured yet -- deliberate).
2. classifier scores: extend `eval_final_lora_test.py` to record the first-token log-odds logit("harm") - logit("un") per row (locked Gemma LoRA adapter).
3. re-run this script with both files; review selected config; then and only then `--run-test`.
4. plug real pi (traffic prevalence) and kappa (timing ratio) into `--prevalences` / `--kappas` / `--deploy-prevalence` when a representative trace exists.
