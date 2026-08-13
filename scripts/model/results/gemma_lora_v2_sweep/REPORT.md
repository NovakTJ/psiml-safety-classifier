# Phase 1 hyperparameter sweep — Gemma 3 1B IT LoRA, v2 dataset (bez refusal-a)

## Cilj

Prvi sistematski sweep LoRA hiperparametara na `data/gemma_v2_no_refusal/`, nakon
Experiment 1 pilot runa (`gemma_lora_v2_exp1_r8_lr2e4_seed42_max8_es2/`, r=8/lr=2e-4)
koji je dokazao pipeline ali nije bio tuniran. Cilj: naći bolju kombinaciju
`learning_rate` × `rank` pre finalne, jednokratne evaluacije na `test.jsonl`.

## Pretraženi prostor

- **learning_rate** ∈ {5e-5, 1e-4, 2e-4, 3e-4}
- **rank (r)** ∈ {4, 8, 16}, **alpha = 2×r** (fiksni odnos — alpha nije nezavisna varijabla)
- dropout=0.05, seed=data_seed=42 — fiksno, isto kao Experiment 1
- Sve ostalo identično Experiment 1 konfiguraciji (batch=4×grad_accum=8, warmup_ratio=0.05,
  AdamW + linear scheduler, BF16, gradient checkpointing, max_seq_length=1024,
  prompt `prompt_1`, target_modules q/k/v/o/gate/up/down)

Grid = 4×3 = 12 ćelija. Ćelija (lr=2e-4, r=8) je **ponovo iskorišćena** iz već
završenog Experiment 1 runa (`source=existing_reference`) umesto ponovnog treniranja
— dakle **11 novih runova** stvarno istrenirano u ovom sweep-u.

Early stopping: 2 uzastopne epohe bez strogog poboljšanja validation F1 (harmful = pozitivna klasa),
maksimalno 8 epoha (kao Experiment 1).

## Izvršenje

- Skripta: `scripts/model/sweep_lora_v2.py`, pokrenuta preko `scripts/model/launch_sweep_lora_v2.sh`
  (nohup+setsid, detached, PID-lockfile protiv duplog pokretanja).
- Start: **2026-08-13 01:26 UTC**, kraj: **06:26 UTC** — ukupno ~5h, unutar budžeta od 340 min.
- Zbir trajanja 11 treniranih runova: **~300 min** (prosek ~27 min/run) — poklapa se sa
  ukupnim vremenom, bez praznog hoda ili zaglavljivanja između runova.
- `sweep_state.json`: `"incomplete": false`, svih 12 konfiguracija `"completed"`.
- `sweep.err`: samo bezopasna upozorenja o verziji kernela (4.18 < preporučenih 5.5), nema pravih grešaka.
- `invalid_count` = **0 na svih 12 konfiguracija**, na svim epohama — model nikad nije
  izbacio format koji se ne parsira kao harmful/unharmful.
- Skripta je bezbedna za ponovno pokretanje: završeni configi se preskaču, neuspešni
  se ponavljaju, a config zatečen kao "running" (proces ubijen usred treninga) kreće
  iznova od baznog modela — Trainer stanje se nikad ne nastavlja usred treninga.

## Rezultati (validation, 259 redova, harmful = pozitivna klasa)

Sortirano po F1, opadajuće:

```
config                    lr       r   |  P       R       F1      | best_ep  epochs  trajanje
lr1e-4_r16_alpha32       0.0001  16    | 0.9568  0.9688  0.9627   |    5       7      32.8min  ← NAJBOLJI
lr2e-4_r4_alpha8         0.0002   4    | 0.9684  0.9563  0.9623   |    4       6      28.4min  (praktično izjednačen sa #1)
lr3e-4_r8_alpha16        0.0003   8    | 0.9565  0.9625  0.9595   |    2       4      18.8min
lr3e-4_r16_alpha32       0.0003  16    | 0.9563  0.9563  0.9563   |    5       7      32.7min
lr3e-4_r4_alpha8         0.0003   4    | 0.9620  0.9500  0.9560   |    4       6      28.0min
lr2e-4_r16_alpha32       0.0002  16    | 0.9503  0.9563  0.9533   |    6       8      37.4min
lr1e-4_r8_alpha16        0.0001   8    | 0.9735  0.9187  0.9453   |    5       7      32.5min  (najviša preciznost, ali slab recall)
lr2e-4_r8_alpha16        0.0002   8    | 0.9273  0.9563  0.9415   |    2       4      18.6min  (= stari Experiment 1 rezultat, ponovo iskorišćen)
lr5e-5_r8_alpha16        5e-05    8    | 0.9325  0.9500  0.9412   |    3       5      23.4min
lr5e-5_r16_alpha32       5e-05   16    | 0.9321  0.9437  0.9379   |    2       4      18.7min  (najslabiji od r=16)
lr5e-5_r4_alpha8         5e-05    4    | 0.9313  0.9313  0.9313   |    3       5      23.6min
lr1e-4_r4_alpha8         0.0001   4    | 0.9367  0.9250  0.9308   |    3       5      23.5min  ← NAJSLABIJI
```

Puna tabela sa svim kolonama (invalid_count, train_loss, val_loss, stop_reason,
run_path): `sweep_summary.csv` u ovom folderu. Pun trening log: `sweep.out`/`sweep.err`.
Adapteri i checkpointi po konfiguraciji: `lr..._r..._alpha..._dropout0.05_seed42/` podfolderi.

## Prosečan F1 po hiperparametru

```
po rank-u:           r=4 → 0.9451   r=8 → 0.9469   r=16 → 0.9525
po learning_rate-u:  5e-5 → 0.9368  1e-4 → 0.9463  2e-4 → 0.9524  3e-4 → 0.9572
```

## Analiza / zaključci

**1. Pobednik je čist — najbolji je i po F1 i po recall-u istovremeno.**
`lr=1e-4, r=16` ima i najveći F1 (0.9627) i najveći recall (0.9688) od svih 12
konfiguracija — nema kompromisa preciznost-za-recall koji bi zakomplikovao izbor.
Ovo je posebno relevantno jer projekat verovatno prioritetizuje unsafe recall
iznad F1 (videti CLAUDE.md napomenu iz Experiment 1) — ovde se srećom poklapaju.

**2. Razlika između #1 i #2 je verovatno šum, ne signal.** Top dva runa
(F1 0.9627 vs 0.9623) razlikuju se za 0.0004 — na validaciji od 259 primera to je
red veličine jednog jedinog primera. Praktično su izjednačeni; ne treba čitati
previše u to što je `r=16` "pobedio" `r=4` sa tolikom razlikom.

**3. Veći rank generalno pomaže, ali ne linearno i ne sam po sebi.** Prosek raste
sa rank-om (0.945 → 0.947 → 0.953), ali najslabiji r=16 run (lr=5e-5, F1=0.9379)
je gori od najboljeg r=4 runa. Rank pomaže samo u kombinaciji sa dovoljno visokim
learning rate-om da se ta dodatna kapacitet stvarno iskoristi.

**4. Viši learning rate ubrzava konvergenciju, ali ne poboljšava krajnji rezultat
proporcionalno.** Runovi sa lr=3e-4 dosledno staju najranije (4 epohe, best_epoch=2)
— model brzo "upamti" trening skup i early stopping ga zaustavi. Runovi sa nižim
lr (5e-5, 1e-4) troše više epoha (5-7) da stignu do često slabijeg platoa.
Prosek po lr ipak blago raste sa lr (do 3e-4) — ali pobednički run je na lr=1e-4,
ne na najvišem lr iz grida, pa "viši lr = bolje" nije čist zaključak na nivou
pojedinačnog runa.

**5. Najbolji model prilično overfit-uje trening skup po loss-u, ali to mu ne
škodi na validaciji.** Pobednik ima train_loss=0.0002 (praktično nula — model je
"upamtio" trening primere) dok je val_loss=0.124 — jedan od najvećih gap-ova u
sweep-u (drugi najveći: `lr=2e-4, r=16`, gap=0.139, F1=0.9533 — slabiji).
Konfiguracije sa najmanjim train/val gap-om (npr. `lr=5e-5, r=16`, gap=0.005) su
"najzdravije" po ovoj metrici, ali imaju niži F1 (0.938). Train/val loss gap dakle
NIJE dobar kriterijum za biranje konfiguracije na ovom zadatku — SFT na 2-3-token
target ("harmful"/"unharmful") ide skoro do nule na trening loss-u vrlo rano bez
da to ugrožava klasifikacionu generalizaciju.

**6. Jedan run je loš praktični izbor uprkos pristojnom F1:** `lr=1e-4, r=8` ima
najvišu preciznost od svih (0.9735) ali i najniži recall od "razumnih" konfiguracija
(0.9187) — F1=0.9453 ga gura na sredinu tabele, ali ako je prioritet da se ne
propusti harmful sadržaj (visok recall), ovo bi bio najgori izbor uprkos
pristojnom F1-u.

## Napomene / šta nedostaje

- Svi brojevi ovde su **validacioni** (259 redova). `test.jsonl` (227 redova) je i
  dalje netaknut, namerno, dok se finalna konfiguracija ne zaključa.
- S obzirom da su #1 (`lr=1e-4, r=16`) i #2 (`lr=2e-4, r=4`) statistički
  izjednačeni, vredelo bi razmisliti o ponavljanju na drugom seed-u pre
  konačnog zaključavanja — ili prosto zaključati `lr=1e-4, r=16` jer ima i
  bolji recall (relevantniji za projekat).
- Sledeći korak: zaključati izabranu konfiguraciju i uraditi jednokratnu,
  finalnu evaluaciju na `test.jsonl` (kao što je urađeno za Experiment 1 —
  videti `gemma_lora_v2_exp1_r8_lr2e4_seed42_max8_es2/REPORT.md` za format).
