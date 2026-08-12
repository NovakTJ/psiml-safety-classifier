# Gemma 3 1B IT — LoRA pilot (r=8, lr=2e-4, seed=42)

**Datum:** 2026-08-12 · **Notebook:** `scripts/model/gemma_lora.ipynb` · **Kernel:** `ccpp`

Jedan pilot run sa unapred zaključanom konfiguracijom — **nije** hiperparametarski sweep.
Cilj je bio dokazati da LoRA pipeline radi end-to-end i dobiti referentnu tačku
naspram Gemma zero-shot baseline-a.

---

## Rezultat ukratko

**Najbolja epoha: 3** (`checkpoint-186`), izabrana po najvećem validation harmful F1.

| metrika | zero-shot (prompt_1) | LoRA (epoha 3) | delta |
|---|---:|---:|---:|
| precision | 0.8410 | **0.9586** | +0.1176 |
| recall | 0.9220 | **0.9419** | +0.0199 |
| F1 | 0.8790 | **0.9501** | **+0.0711** |
| invalid rate | 3.17% | **0.00%** | −3.17 pp |

Najzanimljiviji nalaz nije F1 nego **invalid rate: 3.17% → 0.00% u sve tri epohe**.
Zero-shot Gemma je povremeno brbljala umesto da vrati labelu; SFT na dvotokenskom
targetu potpuno uklanja problem poštovanja formata. Metrike „nad validnim
predikcijama" i end-to-end metrike se ovde poklapaju, jer nema nevalidnih izlaza.

---

## Rezultati po epohama

Validation = ceo `val_df` (252 reda: 172 harmful, 80 unharmful), `harmful` = pozitivna klasa,
greedy decoding (`do_sample=False`, `max_new_tokens=10`), isti striktni parser kao baseline.

| epoch | train_loss | val_loss | precision | recall | F1 | invalid |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2923 | 0.0906 | 0.9866 | 0.8547 | 0.9159 | 0 (0.00%) |
| 2 | 0.0598 | 0.0818 | 0.9032 | 0.9767 | 0.9385 | 0 (0.00%) |
| **3** | 0.0227 | 0.0640 | 0.9586 | 0.9419 | **0.9501** | 0 (0.00%) |

`val_loss` monotono pada (0.0906 → 0.0818 → 0.0640), pa na 3 epohe **nema znakova
overfittinga** — vredi probati duži trening u sweep-u.

Epohe se ponašaju karakteristično: epoha 1 je vrlo konzervativna (precision 0.987,
recall 0.855 — propušta harmful), epoha 2 preteruje u suprotnom smeru (recall 0.977,
precision 0.903), epoha 3 balansira. Ako se kasnije prioritizuje **unsafe recall**
(što je za ovaj projekat verovatan cilj), epoha 2 je zanimljivija od epohe 3 uprkos
nižem F1 — pravilo izbora je ovde bilo zaključano na F1.

Konfuziona matrica, epoha 3: TP 162 · FP 7 · FN 10 · TN 73.

---

## Postavka

**Format (generativni SFT).** User turn = zaključani zero-shot `prompt_1` input
(identičan `gemma_demo.ipynb`), assistant turn = samo `harmful` / `unharmful`.
Prefiks se gradi sa `add_generation_prompt=True`, pa je **bit-identičan** onome
što model vidi na inferenciji.

**Loss masking.** `labels = [-100] * len(prefix) + target_ids`. Loss ide isključivo
na target labelu i terminator; nijedan input token ne učestvuje. Dokazano u notebooku
za obe klase pre treninga (npr. unharmful primer: 1017 input tokena `-100`, loss samo
na `un|harm|ful|<end_of_turn>`).

**LoRA.** `r=8, alpha=16, dropout=0.05, bias=none, task_type=CAUSAL_LM`, na svih 7
projekcija (`q/k/v/o/gate/up/down_proj`). Trainable **6.522.880 / 1.006.408.832 = 0.648%**;
verifikovano da su gradijenti isključivo na LoRA tenzorima (364 tenzora, 0 ne-LoRA)
i da je bazni model zamrznut. Standardna LoRA — ne DoRA/PiSSA/QLoRA, bez kvantizacije,
adapter **nije** merge-ovan.

**Hiperparametri.** 3 epohe, lr 2e-4, batch 4 × grad-accum 8 (efektivno 32),
eval batch 8, warmup 0.05, weight decay 0.0, max grad norm 1.0, BF16, AdamW,
linearni scheduler, `seed = data_seed = 42`. 62 optimizer koraka po epohi (186 ukupno).

---

## Truncation (`max_seq_length = 1024`)

Duge primere **ne izbacujemo** — skraćuje se samo sadržaj, head-tail na nivou tokena
(čuva se i početak i kraj, sredina se zamenjuje sa `...`), da signal na kraju teksta
ne bi automatski nestao. Target + `<end_of_turn>` se rezervišu iz budžeta **pre** svega
ostalog, a instrukcija i chat-template struktura se uvek čuvaju. Kad postoji response,
budžet se deli na pola uz preraspodelu neiskorišćenog dela kraće komponente.

| split | skraćeno | max dužina |
|---|---:|---:|
| train | 182 / 1960 (9.29%) | 1022 tokena |
| validation | 19 / 252 (7.54%) | 1020 tokena |

**Target i EOS sačuvani u 100% primera** (2212/2212), verifikovano asertima.

---

## Pitfalls otkriveni u ovom runu

**1. Terminator je `<end_of_turn>` (106), ne `<eos>` (1).** Gemma chat template zatvara
assistant turn sa `<end_of_turn>`, i to je token koji stvarno zaustavlja `generate()`
(`config.eos_token_id = [1, 106]`). Trening na pogrešnom terminatoru bi značio da model
nikad ne nauči da stane tamo gde inferencija očekuje.

**2. Gemma 3 1B OOM-uje na 40 GB A100 — krivac je vocab, ne težine.** Prvi run je pao
na CUDA OOM u epohi 2. Vocab ima 262k tokena, pa je logit tenzor za `batch 4 × 1024`
tačno ~4 GiB u float32 (CE loss radi upcast), plus toliko za gradijent, povrh aktivacija
26 slojeva; uz to je 10 GiB od 34 GiB bilo „reserved but unallocated" (fragmentacija).

Rešenje, **bez menjanja ijednog hiperparametra**:
- `gradient_checkpointing=True` uz `gradient_checkpointing_kwargs={"use_reentrant": False}`
  **i** `base_model.enable_input_require_grads()` — bez ovog drugog PEFT ne dobija
  gradijente kroz checkpointovane blokove;
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
- `prediction_loss_only=True` da Trainer-ov eval ne gomila 262k-široke logite.

Peak memorija 35.8 → ~20 GB. Epoha 1 se posle promene reprodukovala **bit-identično**
(`eval_loss` 0.09056287258863449), što potvrđuje da checkpointing ne menja numeriku.
Ovo je jedino odstupanje od početne konfiguracije i posledica je OOM-a, ne tuning-a.

---

## Analiza grešaka (epoha 3)

**7 false positives, 10 false negatives, 0 invalid.**

Deo FP-ova je sporan zbog samog OR pravila. Primer `orig-70825`: benigan prompt
(„What is the official email of the Tokyo National Museum?") i odgovor koji je zapravo
izbegavanje („I'm not able to browse the web…"), označen kao `compliance` → ground truth
`unharmful`, model kaže `harmful`. Model hvata refusal-ton koji je labeler klasifikovao
kao compliance — granica refusal/deflection je nejasna u samim labelama.

FP-ovi se koncentrišu na primere **bez responsea** i na ne-engleske jezike (sw, ml),
gde je odluka moguća samo na osnovu prompta.

FN-ovi su uglavnom suptilni jailbreak/role-play prompti bez eksplicitno štetnog
rečnika (npr. `orig-17237` — „As a renowned dermatologist…", `trans-51015-de` — role-play
sa promenom identiteta). To je očekivano slabo mesto i direktno relevantno za red-teaming.

---

## Artefakti

```
scripts/model/results/gemma_lora_pilot_r8_lr2e4_seed42/
├── best_adapter/            # = checkpoint-186 (bajt-identičan), NIJE merge-ovan
├── checkpoint-62/           # epoha 1
├── checkpoint-124/          # epoha 2
├── checkpoint-186/          # epoha 3 (najbolji)
├── training_history.csv     # metrike po epohama
├── pilot_config.json        # konfiguracija + param count + razlog za checkpointing
└── REPORT.md
```

Namerno **nisu** sačuvane kopije dataseta ni train/validation splitova.
Ceo `scripts/model/results/` je gitignorisan (~251 MB optimizer state-a).

**`test_df` (259 redova) nije ni dodirnut** — ni za evaluaciju ni za izbor checkpointa.
Koristi se samo u assertu koji proverava veličinu splita.

---

## Sledeći koraci

1. **Hiperparametarski sweep** — ova konfiguracija je dokazana polazna tačka, ne optimum.
   Vredi varirati: broj epoha (val_loss je i dalje padao), `r`/`alpha`, lr.
2. **Razmotriti kriterijum izbora** — ako je cilj unsafe recall, F1 nije prava metrika
   (epoha 2 ima recall 0.977).
3. **Tek na kraju** jedan run sa zaključanom konfiguracijom na netaknutom `test_df`.
4. Ista evaluacija za **Qwen3Guard zero-shot** na istom splitu, za pošteno poređenje.
