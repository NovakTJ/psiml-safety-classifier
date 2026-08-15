# Switched-prompt stress test (benign prompt + harmful response, n=51)

All 51 rows are exchange_label=harmful by construction (unharmful prompt, harmful response swapped in from a real exchange). No negative class -- this measures RECALL ONLY: can the probe catch harm that the prompt alone gives no signal for.

| Probe | feature | recall (=mean pred) | mean prob | min prob | max prob |
|---|---|---:|---:|---:|---:|
| v3_winner_layer15_mean_prompt | layer15_mean_prompt | 0.0196 (1/51) | 0.0739 | 0.0000 | 0.6960 |
| v2_locked_layer27_last | layer27_last | 0.4314 (22/51) | 0.4373 | 0.0047 | 0.9896 |
| response_aware_layer11_mean_response | layer11_mean_response | 0.8431 (43/51) | 0.7890 | 0.0182 | 0.9998 |
