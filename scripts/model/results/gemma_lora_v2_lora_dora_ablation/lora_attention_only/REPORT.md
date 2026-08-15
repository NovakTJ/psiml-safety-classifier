# lora_attention_only — Gemma 3 1B IT LoRA ablation run

## Konfiguracija (sve zajedničko preuzeto iz Experiment 1)

- method=LoRA, use_dora=False, target_scope=attention_only
- target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
- r=8, alpha=16, dropout=0.05, lr=0.0002, seed=42, max_epochs=8, patience=2
- trainable_params=1,490,944 / 1,001,376,896 (0.1489%)
- peak_vram_gb=21.86

## Rezultati po epohi (validation, generative, greedy, max_new_tokens=10)

```
 epoch  train_loss  val_loss  precision  recall       f1  invalid_count  epoch_duration_seconds
     1      0.1176  0.109931   0.808511 0.95000 0.873563              0              176.353109
     2      0.0745  0.087622   0.922078 0.88750 0.904459              0              175.100737
     3      0.0449  0.080780   0.924051 0.91250 0.918239              0              174.300459
     4      0.0059  0.104943   0.949367 0.93750 0.943396              0              174.885918
     5      0.0041  0.177756   0.978873 0.86875 0.920530              1              174.465428
     6      0.0008  0.171158   0.925926 0.93750 0.931677              0              175.444901
```

## Najbolja epoha: 4

precision=0.9494, recall=0.9375, F1=0.9434, invalid_rate=0.0000

## Razlog završetka

early stopping: 2 uzastopne epohe bez strogog poboljšanja F1 (posle epohe 6, najbolji F1=0.9434 na epohi 4)

## Napomena

Test skup (`data/gemma_v2_no_refusal/test.jsonl`) NIJE korišćen ni za trening ni za evaluaciju ovog run-a. Ovo je validation-only rezultat.
