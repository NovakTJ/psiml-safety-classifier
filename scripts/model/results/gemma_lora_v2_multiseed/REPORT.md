# Multi-seed stability check — Gemma 3 1B IT all-linear LoRA, v2 dataset (bez refusal-a)

## Cilj

Phase 2 (`scripts/model/sweep_lora_v2_phase2.py`) je pronašao dve bliske najbolje all-linear LoRA konfiguracije, obe na `dropout=0.0`, ali obe trenirane samo na `seed=42`. Ovaj eksperiment proverava da li ta razlika (F1 0.9684 vs 0.9653) preživljava kroz više seedova, ili je šum jednog seed-a. Svaka konfiguracija je ponovo istrenirana na `seed in {23, 41}` (4 nova treninga); postojeći `seed=42` rezultat se ne ponavlja, samo se čita iz `scripts/model/results/gemma_lora_v2_sweep_phase2/` (Phase 1 i Phase 2 folderi ostaju netaknuti).

## Konfiguracije

- **Config A**: learning_rate=3e-4, rank=8, alpha=16, dropout=0.0 (Phase 2 seed=42 pobednik, F1=0.9684)
- **Config B**: learning_rate=1e-4, rank=16, alpha=32, dropout=0.0 (Phase 2 seed=42 drugoplasirani, F1=0.9653)

## Svih šest pojedinačnih rezultata

```
config_name                                     config_id  seed           source    status  best_epoch  precision  recall       f1  invalid_rate  epochs_completed  duration_seconds
   config_a  config_a_lr3e-4_r8_alpha16_dropout0.0_seed23    23        multiseed completed           4   0.962733 0.96875 0.965732           0.0                 6       1615.743752
   config_a  config_a_lr3e-4_r8_alpha16_dropout0.0_seed41    41        multiseed completed           6   0.974359 0.95000 0.962025           0.0                 8       2447.185086
   config_a  config_a_lr3e-4_r8_alpha16_dropout0.0_seed42    42 phase2_reference completed           6   0.980769 0.95625 0.968354           0.0                 8       2151.609654
   config_b config_b_lr1e-4_r16_alpha32_dropout0.0_seed23    23        multiseed completed           7   0.962025 0.95000 0.955975           0.0                 8       3546.700349
   config_b config_b_lr1e-4_r16_alpha32_dropout0.0_seed41    41        multiseed completed           8   0.967949 0.94375 0.955696           0.0                 8       2145.749041
   config_b config_b_lr1e-4_r16_alpha32_dropout0.0_seed42    42 phase2_reference completed           7   0.974522 0.95625 0.965300           0.0                 8       2165.539808
```

## Agregirani rezultati (mean ± std preko seed-ova)

- **config_a** (n_seeds=3, seeds=23,41,42): F1 = 0.9654 ± 0.0032 (min=0.9620, max=0.9684), precision = 0.9726 ± 0.0091, recall = 0.9583 ± 0.0095, mean invalid_rate = 0.0000, 3/3 runova bez invalid outputa.

- **config_b** (n_seeds=3, seeds=23,41,42): F1 = 0.9590 ± 0.0055 (min=0.9557, max=0.9653), precision = 0.9682 ± 0.0063, recall = 0.9500 ± 0.0063, mean invalid_rate = 0.0000, 3/3 runova bez invalid outputa.

## Izabrani stabilni pobednik

**config_a** — mean F1 = 0.9654 ± 0.0032 preko 3 seed-a (23,41,42).

**Kriterijum izbora (u ovom redosledu, pobednik nikad nije biran po najboljem pojedinačnom seedu)**: 1) najveći mean F1, 2) zatim najveći mean recall, 3) zatim manji F1 std, 4) zatim manji mean invalid rate.

Razlika u mean F1 naspram `config_b`: +0.0064 (0.9654 vs 0.9590). 

**Poređenje sa seed=42 zaključkom**: na seed=42 samom, Config A (F1=0.9684) je vodio Config B (F1=0.9653) za +0.0031. Multi-seed rezultat se slaže sa tim zaključkom (isti pobednik).

## Napomene

- Test skup (`data/gemma_v2_no_refusal/test.jsonl`) NIJE korišćen ni za trening ni za evaluaciju bilo kog runa.
- Attention-only i DoRA eksperimenti NISU pokrenuti u ovoj fazi.
- **Predlog za sledeći eksperiment**: attention-only ablation (LoRA samo na q_proj/k_proj/v_proj/o_proj, bez gate/up/down_proj) sa zaključanim pobedničkim configom iznad, da se proveri koliko MLP projekcije doprinose F1-u naspram broja trenable parametara.
