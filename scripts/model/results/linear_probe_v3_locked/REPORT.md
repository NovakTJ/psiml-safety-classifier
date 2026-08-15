# Linear probe v3 -- LOCKED

**Winner: `layer15_mean_response`, C=0.003, class_weight={0: 1.0, 1: 2.0}`** -- supersedes attempt-2's layer27_last as the project's probe baseline.

## Why this and not the higher-F1 layer15_mean_prompt

sweep_probe_v3_layer_pooling.py's F1-only sweep picked layer15_mean_prompt (val F1 0.9779) -- but that pooling ignores the response entirely, and gemma_v2_no_refusal's final_label collapses to prompt_harm_label on every row, so val F1 alone can't tell a true exchange classifier from a prompt-only one in disguise. Confirmed on data/switched_prompt_dataset.jsonl (51 benign-prompt + harmful-response rows): layer15_mean_prompt recall = **0.0196** (1/51). layer15_mean_response, selected with switched-prompt recall as a second axis, scores **1.0** (51/51) on the same stress test.

## Validation (threshold 0.5, 259 rows)
P=0.9107 R=0.9563 F1=0.9329 FPR=0.1515

## Switched-prompt stress test (51 rows, all harmful by construction)
recall = 1.0000 (51/51)

## Test (threshold 0.5, 227 rows, one-shot, first time test.jsonl was touched in the probe experiment)

| System | precision | recall | F1 | FPR |
|---|---:|---:|---:|---:|
| **layer15_mean_response (this, locked)** | 0.8446 | 0.9843 | 0.9091 | 0.2300 |
| layer27_last (attempt-2, old baseline) | 0.8188 | 0.8898 | 0.8528 | 0.2500 |

F1 delta vs old baseline: +0.0563

## Journey

1. `capture_probe_multilayer.py` -- one GPU pass, 8 layers x 4 poolings instead of attempt-1/2's single layer27/last-token (results/linear_probe_v3_multilayer_pooling/).
2. `sweep_probe_v3_layer_pooling.py` -- F1-only phase1+2 sweep, won by layer15_mean_prompt (val F1 0.9779), rejected after the switched-prompt check (results/linear_probe_v3_multilayer_pooling_sweep/).
3. `sweep_probe_v3_response_aware.py` -- mean_response-only grid, combined_score selection, won by this config (results/linear_probe_v3_multilayer_pooling_sweep_response_aware/).
4. `eval_test_v3_candidates.py` -- one-shot test.jsonl evaluation, confirmed the win on held-out data (results/linear_probe_v3_multilayer_pooling_sweep_response_aware/test_evaluation/).
5. This file: refit on the full train split, saved as the locked artifact.

## Files
- `probe.joblib` -- sklearn Pipeline(StandardScaler, LogisticRegression), fit on data/gemma_v2_no_refusal/train.jsonl
- `probe_config.json` -- full config + all metrics above, machine-readable
