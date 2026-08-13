# Phase 2 dropout sweep — Gemma 3 1B IT LoRA, v2 dataset (bez refusal-a)

## Cilj

Phase 1 (`scripts/model/sweep_lora_v2.py`) je pretražio `learning_rate x rank` na fiksnom `dropout=0.05` i pronašao tri najbolje (learning_rate, rank, alpha) kombinacije. Phase 2 uzima tačno te tri kombinacije i proverava da li podešavanje LoRA dropout-a (`dropout in {0.0, 0.1}`, naspram Phase 1 fiksnog 0.05) dalje poboljšava rezultat. Šest novih treninga; tri postojeća `dropout=0.05` rezultata iz Phase 1 se ne ponavljaju, samo se čitaju kao reference iz `scripts/model/results/gemma_lora_v2_sweep/` (Phase 1 folder ostaje netaknut).

## Svih devet rezultata (šest novih + tri Phase 1 reference)

Sortirano po F1 opadajuće, zatim recall opadajuće, zatim invalid rate rastuće. `status=pending` znači da taj run još nije izvršen.

```
                            config_id  dropout           source    status  best_epoch  precision  recall       f1  invalid_rate  epochs_completed  duration_seconds
  lr3e-4_r8_alpha16_dropout0.0_seed42     0.00     sweep_phase2 completed           6   0.980769 0.95625 0.968354           0.0                 8       2151.609654
 lr1e-4_r16_alpha32_dropout0.0_seed42     0.00     sweep_phase2 completed           7   0.974522 0.95625 0.965300           0.0                 8       2165.539808
lr1e-4_r16_alpha32_dropout0.05_seed42     0.05 phase1_reference completed           5   0.956790 0.96875 0.962733           0.0                 7       1969.417279
 lr1e-4_r16_alpha32_dropout0.1_seed42     0.10     sweep_phase2 completed           3   0.956790 0.96875 0.962733           0.0                 5       1419.618329
  lr2e-4_r4_alpha8_dropout0.05_seed42     0.05 phase1_reference completed           4   0.968354 0.95625 0.962264           0.0                 6       1701.617597
 lr3e-4_r8_alpha16_dropout0.05_seed42     0.05 phase1_reference completed           2   0.956522 0.96250 0.959502           0.0                 4       1126.123405
  lr3e-4_r8_alpha16_dropout0.1_seed42     0.10     sweep_phase2 completed           5   0.968153 0.95000 0.958991           0.0                 7       1977.642479
   lr2e-4_r4_alpha8_dropout0.0_seed42     0.00     sweep_phase2 completed           4   0.980392 0.93750 0.958466           0.0                 6       1630.550040
   lr2e-4_r4_alpha8_dropout0.1_seed42     0.10     sweep_phase2 completed           8   0.956250 0.95625 0.956250           0.0                 8       2263.301076
```

## Najbolja konfiguracija

**lr3e-4_r8_alpha16_dropout0.0_seed42** (sweep_phase2) — precision=0.9808, recall=0.9563, F1=0.9684, invalid_rate=0.0000, best_epoch=6.

## Dve najbolje konfiguracije predložene za proveru na dodatnim seedovima

- **lr3e-4_r8_alpha16_dropout0.0_seed42** — F1=0.9684, recall=0.9563
- **lr1e-4_r16_alpha32_dropout0.0_seed42** — F1=0.9653, recall=0.9563

## Tok najboljeg runa (po epohama)

Iz `/home/mls01/scripts/model/results/gemma_lora_v2_sweep_phase2/lr3e-4_r8_alpha16_dropout0.0_seed42/epoch_history.csv`:

```
 epoch  train_loss  val_loss  precision  recall       f1  invalid_count  epoch_duration_seconds
     1      0.0921  0.136921   0.739336 0.97500 0.840970              0              212.075617
     2      0.0348  0.101742   0.905882 0.96250 0.933333              0              211.507136
     3      0.0082  0.085648   0.955414 0.93750 0.946372              0              211.356721
     4      0.0028  0.099081   0.950311 0.95625 0.953271              0              211.766898
     5      0.0047  0.114998   0.938650 0.95625 0.947368              0              210.646168
     6      0.0000  0.095503   0.980769 0.95625 0.968354              0              211.817598
     7      0.0000  0.097997   0.980769 0.95625 0.968354              0              211.415331
     8      0.0000  0.099250   0.980769 0.95625 0.968354              0              211.759254
```

## Napomene

- Test skup (`data/gemma_v2_no_refusal/test.jsonl`) NIJE korišćen ni za trening ni za evaluaciju bilo kog runa u ovoj fazi.
- Dodatni seed-ovi za dve najbolje konfiguracije (predložene iznad) NISU automatski pokrenuti — to je namerno ostavljeno kao sledeći, ručno pokrenut korak.
