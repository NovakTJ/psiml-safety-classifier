# Konačna test evaluacija — LoRA/DoRA ablacija

Test evaluacija tri adaptera iz LoRA/DoRA ablacije (`lora_attention_only`, `dora_all_linear`, `dora_attention_only`) na `data/gemma_v2_no_refusal/test.jsonl` (227 redova/100 grupa, 127 harmful/100 unharmful). Isti bazni model, prompt, tokenizacija/truncation, parser i generation parametri (`do_sample=False, max_new_tokens=10`) kao tokom treninga/validacije — uvezeno iz `run_lora_dora_ablation_v2.py`, nije reimplementirano.

`21/227` (9.25%) test primera je skraćeno (head-tail truncation na max_seq_length).

**Metodološka napomena**: `dora_all_linear` je izabran kao validation pobednik PRE ovog testa. Test rezultati za `lora_attention_only` i `dora_attention_only` služe isključivo za kompletiranje ablation poređenja i ne smeju se koristiti za post-hoc promenu izabranog modela.

## Rezultati po adapteru (validation -> test)

```
             run_id  val_best_epoch  val_precision  val_recall   val_f1  test_precision_e2e  test_recall_e2e  test_f1_e2e  test_fpr_e2e  test_fnr_e2e  invalid_rate
lora_attention_only               4       0.949367     0.93750 0.943396            0.930233         0.944882     0.937500          0.09      0.055118           0.0
    dora_all_linear               6       0.950311     0.95625 0.953271            0.893130         0.921260     0.906977          0.14      0.078740           0.0
dora_attention_only               6       0.973684     0.92500 0.948718            0.922481         0.937008     0.929688          0.10      0.062992           0.0
```

### lora_attention_only

Confusion matrix (nad validnim predikcijama):
```
                pred_harmful  pred_unharmful
true_harmful             120               7
true_unharmful             9              91
```

### dora_all_linear

Confusion matrix (nad validnim predikcijama):
```
                pred_harmful  pred_unharmful
true_harmful             117              10
true_unharmful            14              86
```

### dora_attention_only

Confusion matrix (nad validnim predikcijama):
```
                pred_harmful  pred_unharmful
true_harmful             119               8
true_unharmful            10              90
```

## Objedinjena tabela (test, end-to-end)

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

## Napomena

Test skup korišćen tačno jednom po adapteru, nakon zaključavanja konfiguracije. Adapter SHA256 hash-ovi potvrđeno nepromenjeni pre/posle (vidi `adapter_hashes.json` u svakom pod-folderu).
