# Linear probe v3 -- layer x pooling screen

Multi-layer capture: /home/mls01/scripts/model/results/linear_probe_v3_multilayer_pooling
8 full-attention layers ([3, 7, 11, 15, 19, 23, 27, 31]) x 4 poolings (['last', 'mean_prompt', 'mean_response', 'mean_last16']) = 32 feature keys, all from the SAME forward pass as attempt 1/2's layer27/last-token capture.

## Phase 1 -- fixed config (C=0.03, class_weight=balanced) across all 32 combos

Reference: attempt-2 sweep's locked layer27_last, same hyperparameters, scored val F1 **0.9231**.

| Rank | feature_key | precision | recall | f1 | fpr |
|---|---|---:|---:|---:|---:|
| 1 | layer15_mean_prompt | 0.9748 | 0.9688 | 0.9718 | 0.0404 |
| 2 | layer19_mean_prompt | 0.9551 | 0.9313 | 0.9430 | 0.0707 |
| 3 | layer31_mean_prompt | 0.9325 | 0.9500 | 0.9412 | 0.1111 |
| 4 | layer23_mean_prompt | 0.9608 | 0.9187 | 0.9393 | 0.0606 |
| 5 | layer11_last | 0.9605 | 0.9125 | 0.9359 | 0.0606 |
| 6 | layer11_mean_prompt | 0.9317 | 0.9375 | 0.9346 | 0.1111 |
| 7 | layer11_mean_response | 0.9664 | 0.9000 | 0.9320 | 0.0505 |
| 8 | layer15_mean_last16 | 0.9363 | 0.9187 | 0.9274 | 0.1010 |
| 9 | layer15_mean_response | 0.9660 | 0.8875 | 0.9251 | 0.0505 |
| 10 | layer19_mean_response | 0.9660 | 0.8875 | 0.9251 | 0.0505 |

## Phase 2 -- L2/lbfgs grid on top 5: ['layer15_mean_prompt', 'layer19_mean_prompt', 'layer31_mean_prompt', 'layer23_mean_prompt', 'layer11_last']

6 C values x 3 class weights x 5 feature keys = 90 fits.

| Rank | config_id | precision | recall | f1 |
|---|---|---:|---:|---:|
| 1 | layer15_mean_prompt__C0.003_cw_balanced | 0.9873 | 0.9688 | 0.9779 |
| 2 | layer15_mean_prompt__C0.01_cw_balanced | 0.9810 | 0.9688 | 0.9748 |
| 3 | layer15_mean_prompt__C1_cw_harmful_x2 | 0.9872 | 0.9625 | 0.9747 |
| 4 | layer15_mean_prompt__C0.003_cw_none | 0.9748 | 0.9688 | 0.9718 |
| 5 | layer15_mean_prompt__C0.01_cw_none | 0.9748 | 0.9688 | 0.9718 |
| 6 | layer15_mean_prompt__C0.03_cw_balanced | 0.9748 | 0.9688 | 0.9718 |
| 7 | layer15_mean_prompt__C0.03_cw_none | 0.9688 | 0.9688 | 0.9688 |
| 8 | layer15_mean_prompt__C0.1_cw_balanced | 0.9747 | 0.9625 | 0.9686 |
| 9 | layer15_mean_prompt__C0.1_cw_none | 0.9627 | 0.9688 | 0.9657 |
| 10 | layer15_mean_prompt__C0.3_cw_none | 0.9627 | 0.9688 | 0.9657 |

## Winner: `layer15_mean_prompt__C0.003_cw_balanced`

- At threshold 0.5: P=0.9873 R=0.9688 F1=0.9779 FPR=0.0202
- Tuned threshold 0.510: P=0.9873 R=0.9688 F1=0.9779
- vs old attempt-2 winner (layer27_last, F1 0.9231): delta = +0.0548

test.jsonl was NOT captured or evaluated in this run.
