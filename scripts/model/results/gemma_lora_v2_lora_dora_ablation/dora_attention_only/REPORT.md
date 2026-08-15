# dora_attention_only — Gemma 3 1B IT DoRA ablation run

## Konfiguracija (sve zajedničko preuzeto iz Experiment 1)

- method=DoRA, use_dora=True, target_scope=attention_only
- target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
- r=8, alpha=16, dropout=0.05, lr=0.0002, seed=42, max_epochs=8, patience=2
- trainable_params=1,560,832 / 1,001,446,784 (0.1559%)
- peak_vram_gb=21.86

## Rezultati po epohi (validation, generative, greedy, max_new_tokens=10)

```
 epoch  train_loss  val_loss  precision  recall       f1  invalid_count  epoch_duration_seconds
     1      0.1154  0.093746   0.857143 0.93750 0.895522              0              236.979722
     2      0.0798  0.082231   0.928571 0.89375 0.910828              0              236.599547
     3      0.0303  0.093436   0.888889 0.95000 0.918429              0              236.789793
     4      0.0135  0.131435   0.960784 0.91875 0.939297              0              237.352940
     5      0.0060  0.138828   0.942308 0.91875 0.930380              0              234.871365
     6      0.0003  0.129559   0.973684 0.92500 0.948718              0              235.915249
     7      0.0002  0.138775   0.967105 0.91875 0.942308              0              236.446016
     8      0.0000  0.137972   0.949367 0.93750 0.943396              0              235.485980
```

## Najbolja epoha: 6

precision=0.9737, recall=0.9250, F1=0.9487, invalid_rate=0.0000

## Razlog završetka

early stopping: 2 uzastopne epohe bez strogog poboljšanja F1 (posle epohe 8, najbolji F1=0.9487 na epohi 6)

## Napomena

Test skup (`data/gemma_v2_no_refusal/test.jsonl`) NIJE korišćen ni za trening ni za evaluaciju ovog run-a. Ovo je validation-only rezultat.
