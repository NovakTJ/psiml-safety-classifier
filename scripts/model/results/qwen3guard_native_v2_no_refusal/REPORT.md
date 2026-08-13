# Qwen3Guard-Gen — native zero-shot evaluacija (v2, bez refusal-a)

## Cilj

Zero-shot evaluacija lokalnog Qwen3Guard-Gen-8B modela na v2 (bez refusal-a) datasetu, koristeći njegov **originalni native chat template i strukturirani output** (`Safety:` / `Categories:` / `Refusal:`), BEZ Gemma `prompt_1` instrukcije, bez dodatnog system prompta, i bez fine-tuninga. Cilj je dobiti uporediv, ali metodološki različit (native-format) baseline naspram tri Gemma sistema.

## Checkpoint

- Lokalna putanja: `/data/models/Qwen3Guard-Gen-8B`
- README naslov: Qwen3Guard-Gen-8B
- architectures: ['Qwen3ForCausalLM'], model_type: `qwen3`
- Broj parametara: 8,190,735,360
- dtype: `torch.bfloat16`, lokalno učitan (`local_files_only=True`), bez preuzimanja.
- Context limit (`max_position_embeddings`): 32768

## Zašto native chat template

Qwen3Guard-Gen tokenizer sam ubacuje kompletan policy/task tekst (safety kategorije, instrukcije za format odgovora) u zavisnosti od toga da li je poslednja poruka `user` (prompt moderation) ili `assistant` (response moderation) — vidi `chat_template` u `tokenizer_config.json`. Pisanje bilo kakve dodatne instrukcije (kao Gemma `prompt_1`) bi odstupilo od zvaničnog načina korišćenja modela i falsifikovalo poređenje 'native baseline' vs. fine-tuned Gemma klasifikator.

## Two-stage prompt/response evaluacija

1. **Prompt moderation** se pokreće za SVAKI red. Ako je native Safety `Controversial` ili `Unsafe`, OR uslov je već ispunjen -> `harmful`, i response moderation se NE pokreće (`response_skip_reason='prompt_unsafe'`).
2. **Response moderation** se pokreće SAMO ako je prompt `Safe` I response nije prazan. Rezultat `Controversial`/`Unsafe` -> `harmful`; `Safe` -> `unharmful`.
3. Ako je prompt `Safe` i response prazan -> `unharmful` bez pokretanja response moderacije (`response_skip_reason='empty_response'`).
4. Nevalidan `Safety` (nedostaje ili konflikt) u bilo kom koraku -> `prediction='invalid'`, sa `invalid_stage` koji beleži gde.

## Binarno mapiranje

Native severity -> binary: `{'Safe': 'safe', 'Controversial': 'unsafe', 'Unsafe': 'unsafe'}`

Binary -> dataset label: `{'safe': 'unharmful', 'unsafe': 'harmful'}`

`Controversial` se tretira kao `unsafe` (zaključano unapred, sekcija 4 zadatka).

## Refusal pravilo

`Refusal` (samo prisutan u response moderation izlazu) se čuva isključivo radi analize grešaka i NE utiče na `prediction`. Nedostajući `Refusal` nije invalid dokle god je `Safety` ispravno parsiran — isto važi i za `Categories`.

## Analiza dužine inputa

Tokenizovano SVIH mogućih prompt-only i prompt+response render-a (validation + test, 719 ukupno) posle native chat template-a:

```
min: 300
median: 506.0
p90: 1428.4000000000003
p95: 2103.8
p99: 4291.260000000029
max: 7560
context_limit: 32768
```

Nijedan input ne premašuje context limit -> truncation NIJE korišćen.

## Validation rezultati

259 redova / 100 grupa. Validation ovde NIJE korišćen za izbor prompta (nema prompt selection) — služi za proveru pipeline-a i zaključavanje konfiguracije pre testa.

**Metrics on valid predictions:**
```json
{
  "precision": 0.959731543624161,
  "recall": 0.89375,
  "f1": 0.9255663430420711,
  "tp": 143,
  "fp": 6,
  "fn": 17,
  "tn": 93,
  "accuracy": 0.9111969111969112,
  "specificity": 0.9393939393939394,
  "fpr": 0.06060606060606061,
  "fnr": 0.10625,
  "balanced_accuracy": 0.9165719696969697,
  "mcc": 0.8190457314461952,
  "valid_count": 259,
  "invalid_count": 0,
  "invalid_rate": 0.0,
  "total": 259
}
```

**End-to-end metrics:**
```json
{
  "precision": 0.959731543624161,
  "recall": 0.89375,
  "f1": 0.9255663430420711,
  "tp": 143,
  "fp": 6,
  "fn": 17,
  "tn": 93,
  "accuracy": 0.9111969111969112,
  "specificity": 0.9393939393939394,
  "fpr": 0.06060606060606061,
  "fnr": 0.10625,
  "balanced_accuracy": 0.9165719696969697,
  "mcc": 0.8190457314461952,
  "invalid_count": 0,
  "invalid_rate": 0.0,
  "total": 259
}
```

## Konačni test rezultati

227 redova / 100 grupa, evaluirano TAČNO JEDNOM istim zaključanim pipeline-om posle validation evaluacije.

**Metrics on valid predictions:**
```json
{
  "precision": 0.9457364341085271,
  "recall": 0.9606299212598425,
  "f1": 0.9531249999999999,
  "tp": 122,
  "fp": 7,
  "fn": 5,
  "tn": 93,
  "accuracy": 0.947136563876652,
  "specificity": 0.93,
  "fpr": 0.07,
  "fnr": 0.03937007874015748,
  "balanced_accuracy": 0.9453149606299213,
  "mcc": 0.8926706356420311,
  "valid_count": 227,
  "invalid_count": 0,
  "invalid_rate": 0.0,
  "total": 227
}
```

**End-to-end metrics (glavno poređenje):**
```json
{
  "precision": 0.9457364341085271,
  "recall": 0.9606299212598425,
  "f1": 0.9531249999999999,
  "tp": 122,
  "fp": 7,
  "fn": 5,
  "tn": 93,
  "accuracy": 0.947136563876652,
  "specificity": 0.93,
  "fpr": 0.07,
  "fnr": 0.03937007874015748,
  "balanced_accuracy": 0.9453149606299213,
  "mcc": 0.8926706356420311,
  "invalid_count": 0,
  "invalid_rate": 0.0,
  "total": 227
}
```

## Invalid outputi

Validation: 0/259 invalid (0.00%). Test: 0/227 invalid (0.00%). Uzrok invalid-a je isključivo nedostajuća ili višestruka `Safety:` linija u raw outputu (pogledati `*_invalid_examples.csv` za tačne redove i sirov tekst); `Categories`/`Refusal` nikada ne uzrokuju invalid.

## Poređenje sa tri Gemma sistema (test skup, end-to-end metrike)

Pre poređenja programski potvrđeno: sva 4 sistema koriste identičnih 227 `row_id` vrednosti i istu `final_label` kolonu iz istog v2 test skupa.

```
                                         system  precision   recall       f1  fpr      fnr  accuracy  invalid_rate
                             Gemma zero-shot v2   0.725000 0.913386 0.808362 0.44 0.086614  0.757709      0.022026
     Gemma initial LoRA (Experiment 1, seed=42)   0.888889 0.944882 0.916031 0.15 0.055118  0.903084      0.000000
Gemma sweep LoRA (validation-selected, seed=42)   0.927419 0.905512 0.916335 0.09 0.094488  0.907489      0.000000
                    Qwen3Guard native zero-shot   0.945736 0.960630 0.953125 0.07 0.039370  0.947137      0.000000
```

Qwen3Guard koristi svoj originalni native moderation format (two-stage OR, Safety/Categories/Refusal); sva tri Gemma sistema koriste ranije zaključani generativni klasifikacioni format (`prompt_1`, target 'harmful'/'unharmful'). Ovo NIJE poređenje istog formata — to je namerno, jer je cilj uporediti gotov specijalizovani safety-moderation model u svom prirodnom režimu rada naspram fine-tuned opšte-namenskog modela.

## Zaključak

Qwen3Guard native zero-shot postiže end-to-end F1=0.9531 na test skupu (0.9457 precision / 0.9606 recall / FPR=0.0700), naspram najboljeg Gemma sistema (Gemma sweep LoRA (validation-selected, seed=42), F1=0.9163). Ovo je jedan zero-shot run bez ikakvog podešavanja specifičnog za dataset (nema prompt selection, nema fine-tuninga) — rezultat treba čitati kao 'gotov model iz kutije', ne kao gornju granicu Qwen3Guard performansi na ovom podatku. Razlike u F1 između sistema su realne, ali dataset je relativno mali (227 test redova) pa pojedinačne procentne poene ne treba preuveličavati.

## Napomena o testu

Test skup je korišćen TAČNO JEDNOM, posle validation evaluacije, sa već zaključanim pipeline-om (native template, parser, mapiranje, generation parametri, OR logika, bez truncation-a). Test rezultati nisu korišćeni ni za kakvu naknadnu promenu sistema.
