# Experiment 1 — Gemma 3 1B IT LoRA, v2 dataset

## Cilj

Prvi pravi v2 LoRA eksperiment (do 8 epoha, F1-based early stopping, patience=2).
Prethodni trening od 3 epohe (`gemma_lora_pilot_v2_no_refusal_r8_lr2e4_seed42/`) bio je
isključivo tehnička provera pipeline-a, memorije i konfiguracije — NIJE puni eksperiment
i ovde se ne koristi kao glavni rezultat.

## Dataset i target pravilo

`data/gemma_v2_no_refusal/train.jsonl` (1985 redova/800 grupa),
`validation.jsonl` (259 redova/100 grupa). `test.jsonl` NIJE korišćen.

`final_label` = harmful ako je `prompt_harm_label == harmful` ILI `response_harm_label == harmful`.
`response_refusal_label` ne utiče na target (programski potvrđeno pre treninga).

## Konfiguracija

- LoRA: r=8, alpha=16, dropout=0.05, bias=none, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
- lr=2e-4, batch=4×grad_accum=8 (eff. 32), warmup_ratio=0.05, weight_decay=0, max_grad_norm=1.0
- AdamW, linear scheduler (računat za max 8 epoha), BF16, gradient checkpointing UKLJUČEN
- seed=data_seed=42, max_seq_length=1024
- Nov base model + nov, netreniran LoRA adapter (bez resume_from_checkpoint)
- Prompt (zaključan, isti kao v2 zero-shot): `prompt_1`

## Rezultati po završenoj epohi (validation, harmful = pozitivna klasa)

 epoch  train_loss  val_loss  precision  recall     f1  invalid_count  invalid_rate  tp  fp  fn  tn  epoch_duration_seconds  best_so_far  epochs_without_improvement
     1      0.0942    0.1074     0.8235  0.9625 0.8876              0           0.0 154  33   6  66                223.3207         True                           0
     2      0.0586    0.0729     0.9273  0.9562 0.9415              0           0.0 153  12   7  87                222.0903         True                           0
     3      0.0130    0.1165     0.8889  0.9500 0.9184              0           0.0 152  19   8  80                222.0422        False                           1
     4      0.0010    0.1101     0.9379  0.9438 0.9408              0           0.0 151  10   9  89                222.5426        False                           2

## Razlog završetka treninga

early stopping: 2 uzastopne epohe bez strogog poboljšanja F1 (posle epohe 4, najbolji F1=0.9415 na epohi 2)

Poslednja završena epoha: **4** / max 8.
Early stopping aktiviran: **True**.

## Najbolja epoha

Epoha **2** (kriterijum: najveći F1 → veći recall → manji invalid rate, preko svih
završenih epoha): precision=0.9273, recall=0.9563,
F1=0.9415, invalid_count=0, invalid_rate=0.00%.

Adapter: `best_adapter/` (iz `checkpoint-126`, NIJE merge-ovan u bazni model).

## Poređenje sa zaključanim v2 zero-shot rezultatom

| metrika | zero-shot v2 | LoRA Exp1 (best) | delta |
|---|---:|---:|---:|
| precision | 0.7760 | 0.9273 | +0.1513 |
| recall | 0.9103 | 0.9563 | +0.0460 |
| f1 | 0.8378 | 0.9415 | +0.1037 |
| invalid_rate | 0.0154 | 0.0000 | -0.0154 |

## Napomene

- Test skup NIJE korišćen ni za trening ni za evaluaciju.
- Nema plotova u ovom run-u — `step_history.csv`/`epoch_history.csv` sačuvani za kasniju analizu.
- Prethodni 3-epoha run je tehnička provera pipeline-a, ostaje netaknut na disku.


---

# Final test evaluation — locked epoch 2 adapter

## Zašto epoha 2

Epoha 2 ima najveći validation harmful F1 (0.9415) među sve 4 završene epohe
Eksperimenta 1 (epohe 3 i 4 nisu strogo poboljšale F1, što je pokrenulo early
stopping posle epohe 4). Adapter je izabran **isključivo na validation skupu**
— `test.jsonl` nije bio ni učitan pre ovog koraka.

Adapter: `/home/mls01/scripts/model/results/gemma_lora_v2_exp1_r8_lr2e4_seed42_max8_es2/best_adapter`
SHA256 (adapter_model.safetensors): `6a2683f45098dd50d89d7ff9fc31fb9dda31b1926c7a0bbcc5f95b630421fdac`

## Test skup

`/home/mls01/data/gemma_v2_no_refusal/test.jsonl` — 227 redova, 100 original_idx grupa.

Distribucija final_label:
final_label
harmful      127
unharmful    100

## A. Test metrike — nad validnim predikcijama

 precision   recall       f1  tp  fp  fn  tn  valid_count  invalid_count  invalid_rate
  0.888889 0.944882 0.916031 120  15   7  85          227              0           0.0

## B. Test metrike — end-to-end (invalid uvek pogrešan)

 precision   recall       f1  tp  fp  fn  invalid_count  invalid_rate  accuracy
  0.888889 0.944882 0.916031 120  15   7              0           0.0  0.903084

## Confusion matrix (test, nad validnim predikcijama)

                pred_harmful  pred_unharmful
true_harmful             120               7
true_unharmful            15              85

## Greške

False positives: 15 | False negatives: 7 | Invalid: 0

## Validation vs. test (zaključani adapter)

                    split  precision  recall     f1  invalid_count  invalid_rate
     validation — epoch 2     0.9273  0.9562 0.9415              0           0.0
test — zaključani adapter     0.8889  0.9449 0.9160              0           0.0

## Zero-shot v2 vs. LoRA (test, oba na osnovu validnih predikcija)

                                         model  precision  recall     f1  invalid_count  invalid_rate
                       zero-shot v2 (prompt_1)     0.7296  0.9431 0.8227              5         0.022
LoRA Exp1 (best_adapter, epoch 2) — valid-only     0.8889  0.9449 0.9160              0         0.000

## Zaključak o generalizaciji

Validation F1 (epoha 2) = 0.9415, test F1 (valid-only) = 0.9160
(delta -0.0255). Rezultat na testu je blizu validaciji, bez znakova preteranog prilagođavanja validation skupu tokom izbora checkpointa.
U odnosu na zaključani v2 zero-shot rezultat na test skupu (F1=0.8227),
LoRA (valid-only) F1 je +0.0933.

**NAPOMENA: test rezultati NISU korišćeni za bilo kakvo dodatno podešavanje modela,
prompta, parsera ili konfiguracije — ovo je jednokratna, konačna evaluacija.**
