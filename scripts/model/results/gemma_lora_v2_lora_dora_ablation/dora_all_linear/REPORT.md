# dora_all_linear — Gemma 3 1B IT DoRA ablation run

## Konfiguracija (sve zajedničko preuzeto iz Experiment 1)

- method=DoRA, use_dora=True, target_scope=all_linear
- target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
- r=8, alpha=16, dropout=0.05, lr=0.0002, seed=42, max_epochs=8, patience=2
- trainable_params=6,982,144 / 1,006,868,096 (0.6935%)
- peak_vram_gb=21.93

## Rezultati po epohi (validation, generative, greedy, max_new_tokens=10)

```
 epoch  train_loss  val_loss  precision  recall       f1  invalid_count  epoch_duration_seconds
     1      0.1012  0.088612   0.818182 0.95625 0.881844              0              395.975147
     2      0.0420  0.104133   0.894118 0.95000 0.921212              0              389.515815
     3      0.0172  0.105283   0.967105 0.91875 0.942308              0              387.482184
     4      0.0002  0.154937   0.955975 0.95000 0.952978              0              388.028573
     5      0.0006  0.163993   0.965753 0.88125 0.921569              0              387.331945
     6      0.0000  0.164107   0.950311 0.95625 0.953271              0              393.327208
     7      0.0000  0.168864   0.950000 0.95000 0.950000              0              390.296004
     8      0.0000  0.167229   0.950311 0.95625 0.953271              0              390.517639
```

## Najbolja epoha: 6

precision=0.9503, recall=0.9563, F1=0.9533, invalid_rate=0.0000

## Razlog završetka

early stopping: 2 uzastopne epohe bez strogog poboljšanja F1 (posle epohe 8, najbolji F1=0.9533 na epohi 6)

## Napomena

Test skup (`data/gemma_v2_no_refusal/test.jsonl`) NIJE korišćen ni za trening ni za evaluaciju ovog run-a. Ovo je validation-only rezultat.
