# Gemma 3 1B IT — LoRA v2 (no-refusal) sanity pilot

## Konfiguracija

- **seed**: 42
- **data_seed**: 42
- **epochs**: 3
- **learning_rate**: 0.0002
- **per_device_train_batch_size**: 4
- **gradient_accumulation_steps**: 8
- **effective_batch_size**: 32
- **per_device_eval_batch_size**: 8
- **warmup_ratio**: 0.05
- **weight_decay**: 0.0
- **max_grad_norm**: 1.0
- **precision**: BF16
- **optimizer**: AdamW
- **lr_scheduler**: linear
- **max_seq_length**: 1024
- **lora_r**: 8
- **lora_alpha**: 16
- **lora_dropout**: 0.05
- **lora_bias**: none
- **lora_target_modules**: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
- **gradient_checkpointing**: True
- **pytorch_cuda_alloc_conf**: expandable_segments:True
- **quantization**: none
- **dataset**: gemma_v2_no_refusal (train.jsonl + validation.jsonl, test NIJE korišćen)

Biblioteke: torch 2.11.0+cu128, transformers 4.57.6,
huggingface_hub 0.36.0, peft 0.20.0, accelerate 1.14.0.
GPU: NVIDIA A100-SXM4-40GB.

## Dataset

`data/gemma_v2_no_refusal/train.jsonl` (1985 redova/800 grupa) i
`validation.jsonl` (259 redova/100 grupa). `test.jsonl` NIJE korišćen.
`final_label` = prompt harmful OR response harmful; `response_refusal_label`
je samo metadata (programski potvrđeno pre treninga).

## Parametri

Ukupno: 1,006,408,832 | Trainable: 6,522,880 (0.6481%) — samo LoRA.

## Truncation

Train skraćeno: 181/1985
(9.12%). Validation skraćeno:
24/259 (9.27%).
Target + `<end_of_turn>` sačuvani u 100% primera.

## Rezultati po epohama (validation, harmful = pozitivna klasa)

                  epoch train_loss  val_loss  precision  recall     f1  invalid_count  invalid_rate
                      1   0.232933  0.074421     0.8800  0.9625 0.9194              0        0.0000
                      2   0.057117  0.090372     0.8947  0.9562 0.9245              0        0.0000
                      3   0.009583  0.096456     0.9490  0.9312 0.9401              0        0.0000
zero-shot v2 (prompt_1)       None      None     0.7760  0.9103 0.8378              4        0.0154

## Izabrani checkpoint

Epoha **3** (checkpoint-189) — najveći F1 = 0.9401
(tie-break: recall 0.9313, invalid_rate 0.00%).
Adapter: `best_adapter/` (NIJE merge-ovan u bazni model).

## Poređenje sa v2 zero-shot baselineom

| metrika | zero-shot v2 | LoRA (best) | delta |
|---|---:|---:|---:|
| precision | 0.7760 | 0.9490 | +0.1730 |
| recall | 0.9103 | 0.9313 | +0.0210 |
| f1 | 0.8378 | 0.9401 | +0.1023 |
| invalid_rate | 0.0154 | 0.0000 | -0.0154 |

## Greške najboljeg checkpointa (validation)

False positives: 8 | False negatives: 11 | Invalid: 0

## Napomene

- Test skup NIJE korišćen ni za trening ni za evaluaciju.
- Sweep NIJE pokrenut — jedan zaključan config (sanity pilot).
- Stari v1 LoRA rezultati (`gemma_lora_pilot_r8_lr2e4_seed42/`) ostaju netaknuti.
