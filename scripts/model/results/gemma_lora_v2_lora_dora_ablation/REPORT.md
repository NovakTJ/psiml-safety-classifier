# LoRA vs DoRA / attention-only vs all-linear ablation — Gemma 3 1B IT, v2 dataset

## Cilj kontrolisanog 2x2 eksperimenta

Kontrolisano poređenje da li (a) MLP projekcije (gate/up/down_proj) doprinose rezultatu u odnosu na attention-only LoRA, i (b) da li DoRA (`use_dora=True`) poboljšava rezultat u odnosu na LoRA, kada su svi ostali uslovi identični. Menja se ISKLJUČIVO `use_dora` i `target_modules` (attention-only vs all-linear); learning_rate, rank, alpha, dropout, seed, max_epochs, patience, dataset, prompt, batch/optimizer/scheduler, dtype i gradient checkpointing su nepromenjeni u sva četiri reda ovog poređenja — pročitani direktno iz Experiment 1 (vidi `reference_config.json`), nisu nagađani.

## Postojeći Experiment 1 (referenca, NIJE ponovo treniran)

`/home/mls01/scripts/model/results/gemma_lora_v2_exp1_r8_lr2e4_seed42_max8_es2` — LoRA, all-linear (attention + MLP), r=8, alpha=16, dropout=0.05, lr=0.0002, seed=42. Njegov adapter i rezultati nisu ni pokretani ni menjani od strane ovog skripta -- samo pročitani.

## Tri nova run-a (redosled izvršavanja je zaključan)

1. **lora_attention_only** — method=LoRA, use_dora=False, target_scope=attention_only, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'] [smoke test: PASSED (peak_vram=16.070017337799072, eval_smoke_n_samples=5)]
2. **dora_all_linear** — method=DoRA, use_dora=True, target_scope=all_linear, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'] [smoke test: PASSED (peak_vram=16.098408222198486, eval_smoke_n_samples=5)]
3. **dora_attention_only** — method=DoRA, use_dora=True, target_scope=attention_only, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'] [smoke test: PASSED (peak_vram=16.07821226119995, eval_smoke_n_samples=5)]

## Rezultati (validation, sortirano po F1 opadajuće)

```
              run_id method   target_scope  use_dora    status             source  best_epoch  precision  recall       f1  invalid_rate  trainable_parameters  trainable_percentage  peak_vram_gb  duration_seconds
     dora_all_linear   DoRA     all_linear      True completed        ablation_v2           6   0.950311 0.95625 0.953271           0.0               6982144              0.693452     21.927773       3873.362285
 dora_attention_only   DoRA attention_only      True completed        ablation_v2           6   0.973684 0.92500 0.948718           0.0               1560832              0.155858     21.857813       2440.242915
 lora_attention_only   LoRA attention_only     False completed        ablation_v2           4   0.949367 0.93750 0.943396           0.0               1490944              0.148889     21.857032       1339.772567
lora_all_linear_exp1   LoRA     all_linear     False completed existing_reference           2   0.927273 0.95625 0.941538           0.0               6522880              0.648134           NaN       1115.636143
```

## Poređenje

- **Efekat uklanjanja MLP target modula (LoRA)**: all-linear F1=0.9415 vs attention-only F1=0.9434 (delta=+0.0019); trainable params 6,522,880 -> 1,490,944.
- **Efekat LoRA->DoRA (all-linear)**: F1=0.9415 -> 0.9533 (delta=+0.0117); peak VRAM nan -> 21.93 GB; duration 1116s -> 3873s.
- **Efekat LoRA->DoRA (attention-only)**: F1=0.9434 -> 0.9487 (delta=+0.0053); peak VRAM 21.86 -> 21.86 GB; duration 1340s -> 2440s.

**Najbolja validation konfiguracija (zaključani kriterijum: F1 -> recall -> invalid_rate)**: dora_all_linear — precision=0.9503, recall=0.9563, F1=0.9533. Ovo je validation zaključak; NE predstavlja test performanse.

## Napomene

- Test skup (`data/gemma_v2_no_refusal/test.jsonl`) NIJE učitan niti korišćen ni u jednom koraku ove ablacije -- ni za trening, ni za izbor konfiguracije, ni za metrike.
- `lora_all_linear_exp1` (Experiment 1) je isključivo referenca pročitana sa diska -- adapter i rezultati nisu ponovo generisani ni menjani.
- Rezultati u ovom izveštaju su validation-only i ne treba ih predstavljati kao test rezultate.


---

# Konačna test evaluacija

Test evaluacija sva tri nova adaptera na `data/gemma_v2_no_refusal/test.jsonl` (227 redova/100 grupa, 127 harmful/100 unharmful), evaluirano TAČNO JEDNOM po adapteru, koristeći identičan pipeline (tokenizer/prompt/truncation/parser/generation) kao tokom treninga i validacije. Puni detalji, FP/FN/invalid analiza, i sve sačuvane datoteke su u `test_evaluation/` pod-folderu.

**Metodološka napomena**: `dora_all_linear` je izabran kao validation pobednik PRE ovog testa (na osnovu validation F1). Test rezultati za `lora_attention_only` i `dora_attention_only` predstavljaju samo završno ablation poređenje i NE smeju se koristiti za post-hoc promenu izabranog modela ili hiperparametara.

## Zaključana konfiguracija svakog adaptera

- **lora_attention_only**: method=LoRA, use_dora=False, target_scope=attention_only, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], r=8, alpha=16, dropout=0.05, lr=0.0002, seed=42
- **dora_all_linear**: method=DoRA, use_dora=True, target_scope=all_linear, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], r=8, alpha=16, dropout=0.05, lr=0.0002, seed=42
- **dora_attention_only**: method=DoRA, use_dora=True, target_scope=attention_only, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], r=8, alpha=16, dropout=0.05, lr=0.0002, seed=42

## Validation vs test rezultati

- **lora_attention_only**: validation F1=0.9434 (P=0.9494, R=0.9375) -> test F1=0.9375 (P=0.9302, R=0.9449, FPR=0.0900, FNR=0.0551)
- **dora_all_linear**: validation F1=0.9533 (P=0.9503, R=0.9563) -> test F1=0.9070 (P=0.8931, R=0.9213, FPR=0.1400, FNR=0.0787)
- **dora_attention_only**: validation F1=0.9487 (P=0.9737, R=0.9250) -> test F1=0.9297 (P=0.9225, R=0.9370, FPR=0.1000, FNR=0.0630)

## Confusion matrice (test, nad validnim predikcijama)

**lora_attention_only**:
```
                pred_harmful  pred_unharmful
true_harmful             120               7
true_unharmful             9              91
```

**dora_all_linear**:
```
                pred_harmful  pred_unharmful
true_harmful             117              10
true_unharmful            14              86
```

**dora_attention_only**:
```
                pred_harmful  pred_unharmful
true_harmful             119               8
true_unharmful            10              90
```

## FP/FN/invalid analiza

- **lora_attention_only**: 9 false positives, 7 false negatives, 0 invalid (regex baseline/other systems' invalid handled identically -- invalid never becomes a valid label; on harmful rows it counts as FN, on unharmful rows as FP, end-to-end).
- **dora_all_linear**: 14 false positives, 10 false negatives, 0 invalid (regex baseline/other systems' invalid handled identically -- invalid never becomes a valid label; on harmful rows it counts as FN, on unharmful rows as FP, end-to-end).
- **dora_attention_only**: 10 false positives, 8 false negatives, 0 invalid (regex baseline/other systems' invalid handled identically -- invalid never becomes a valid label; on harmful rows it counts as FN, on unharmful rows as FP, end-to-end).

## Objedinjena tabela svih sistema (test, end-to-end)

Sortirano po F1 opadajuće (najbolji -> najgori):

```
rank               System                Method   Target scope  precision   recall       f1  fpr      fnr  accuracy  invalid_rate
   1           Qwen3Guard      Native zero-shot            N/A   0.945736 0.960630 0.953125 0.07 0.039370  0.947137      0.000000
   2  LoRA attention-only                  LoRA attention_only   0.930233 0.944882 0.937500 0.09 0.055118  0.929515      0.000000
   3  DoRA attention-only                  DoRA attention_only   0.922481 0.937008 0.929688 0.10 0.062992  0.920705      0.000000
   4     Gemma sweep LoRA                  LoRA     all_linear   0.927419 0.905512 0.916335 0.09 0.094488  0.907489      0.000000
   5  Gemma initial LoRA                  LoRA     all_linear   0.888889 0.944882 0.916031 0.15 0.055118  0.903084      0.000000
   6      DoRA all-linear                  DoRA     all_linear   0.893130 0.921260 0.906977 0.14 0.078740  0.894273      0.000000
   7      Gemma zero-shot             Zero-shot            N/A   0.725000 0.913386 0.808362 0.44 0.086614  0.757709      0.022026
   8       Regex baseline Regex (train-derived)            N/A   0.333333 0.031496 0.057554 0.08 0.968504  0.422907      0.000000
```

## Kratki zaključci

- Efekat uklanjanja MLP target modula (LoRA, test): all-linear (Exp1) F1=0.9160 -> attention-only F1=0.9375 (delta=+0.0215).
- Efekat LoRA->DoRA (all-linear, test): F1=0.9160 -> 0.9070 (delta=-0.0091).
- Efekat LoRA->DoRA (attention-only, test): F1=0.9375 -> 0.9297 (delta=-0.0078).
- Najveći test F1: **Qwen3Guard** (F1=0.9531).
- Najveći harmful recall: **Qwen3Guard** (recall=0.9606).

**Ponovljena metodološka napomena**: `dora_all_linear` je zaključan kao izabrani model PRE test evaluacije, na osnovu validation F1. Test rezultati za `lora_attention_only` i `dora_attention_only` NISU osnova za bilo kakvu naknadnu promenu te selekcije.
