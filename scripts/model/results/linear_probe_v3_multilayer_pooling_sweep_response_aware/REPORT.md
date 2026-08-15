# Linear probe v3 -- response-aware sweep (mean_response pooling only)

Motivation: sweep_probe_v3_layer_pooling.py's winner (layer15_mean_prompt) scored val F1 0.9779 but recall 0.0196 (1/51) on the switched-prompt stress test (benign prompt + harmful response). This sweep restricts to `mean_response` pooling (looks at the response) across all 8 layers, and selects by combined_score = 0.5*val_f1 + 0.5*switched_recall instead of val_f1 alone.

144 fits total.

## Top 10 by combined score

| Rank | config_id | val_precision | val_recall | val_f1 | switched_recall | combined |
|---|---|---:|---:|---:|---:|---:|
| 1 | layer15_mean_response__C0.003_cw_harmful_x2 | 0.9107 | 0.9563 | 0.9329 | 1.0000 | 0.9665 |
| 2 | layer27_mean_response__C0.003_cw_harmful_x2 | 0.8869 | 0.9313 | 0.9085 | 1.0000 | 0.9543 |
| 3 | layer19_mean_response__C0.003_cw_harmful_x2 | 0.9207 | 0.9437 | 0.9321 | 0.9608 | 0.9464 |
| 4 | layer23_mean_response__C0.003_cw_harmful_x2 | 0.8929 | 0.9375 | 0.9146 | 0.9608 | 0.9377 |
| 5 | layer15_mean_response__C0.01_cw_harmful_x2 | 0.9259 | 0.9375 | 0.9317 | 0.9412 | 0.9364 |
| 6 | layer31_mean_response__C0.003_cw_harmful_x2 | 0.8639 | 0.9125 | 0.8875 | 0.9804 | 0.9340 |
| 7 | layer31_mean_response__C0.01_cw_harmful_x2 | 0.8987 | 0.8875 | 0.8931 | 0.9608 | 0.9269 |
| 8 | layer19_mean_response__C0.01_cw_harmful_x2 | 0.9259 | 0.9375 | 0.9317 | 0.9216 | 0.9266 |
| 9 | layer15_mean_response__C0.003_cw_none | 0.9423 | 0.9187 | 0.9304 | 0.9216 | 0.9260 |
| 10 | layer27_mean_response__C0.01_cw_harmful_x2 | 0.8970 | 0.9250 | 0.9108 | 0.9412 | 0.9260 |

## Winner: `layer15_mean_response__C0.003_cw_harmful_x2`
- val: P=0.9107 R=0.9563 F1=0.9329
- switched-prompt recall: 1.0000

For reference -- best by val_f1 alone: `layer11_mean_response__C0.003_cw_none` (val_f1=0.9490, switched_recall=0.8824); best by switched_recall alone: `layer15_mean_response__C0.003_cw_harmful_x2` (val_f1=0.9329, switched_recall=1.0000).

test.jsonl was NOT captured or evaluated in this run.
