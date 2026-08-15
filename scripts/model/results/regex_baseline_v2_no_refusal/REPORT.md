# Train-derived keyword/regex baseline

Ovo je **train-derived keyword/regex baseline** — regex obrasci nisu ručno napisani niti preuzeti iz spoljne liste harmful reči, i ovo nije neuronski/ML model. Svi izrazi su automatski izvedeni isključivo iz `train.jsonl` statistikom (group-aware document frequency + smoothed log-odds), po unapred zaključanim pravilima navedenim ispod.

## Cilj

Jednostavan, reproduktibilan baseline za v2 (bez refusal-a) harmful/unharmful klasifikaciju, kao donja granica za poređenje sa Gemma zero-shot/LoRA i Qwen3Guard sistemima. Regex provera i `prompt` i `response`; pozitivna klasa je `harmful`.

## Zašto izraze učimo samo iz train skupa

Validation i test moraju ostati nezagađeni informacijom korišćenom za konstrukciju modela — izvlačenje kandidata isključivo iz train skupa je analogno tome da se model trenira samo na train podacima. Validation se koristi SAMO za izbor top_k/min_distinct_matches konfiguracije (ne za generisanje izraza), a test se učitava tek nakon što je konfiguracija potpuno zaključana.

## Normalizacija (zaključana)

1. Unicode NFKC
2. lowercase
3. zamena interpunkcije/ostalih non-word separatora jednim razmakom (regex [^\w]+ -> ' ')
4. spajanje višestrukih razmaka
5. uklanjanje početnih/završnih razmaka

Bez prevođenja, stemming/lemmatizacije, LLM-a, semantic embeddings-a, ručnog dekodiranja obfuskacija ili spoljnih rečnika.

## Group-aware izvlačenje kandidata

Iz normalizovanog prompt+response svakog train reda izvučeni su word n-grami dužine [1, 2, 3]. Unija n-grama po redu (prompt + response, svaki izraz doprinosi najviše jednom po redu), zatim unija po original_idx grupi (svaki izraz doprinosi najviše jednom po grupi, bez obzira na broj redova/augmentacija u grupi ili broj ponavljanja unutar reda). Train: 400 harmful grupa, 400 unharmful grupa. Filteri kandidata: harmful_group_df >= 3, harmful_group_prevalence > unharmful_group_prevalence, izraz nije potpuno numerički. Rezultat: **16139 kandidata** iz 497428 ukupno distinct n-grama u train grupama.

## Formula rangiranja

Fiksno Jeffreys-stil (add-0.5) smoothing na group-presence brojevima:

```
smoothed_log_odds = ln((a+0.5)/(H-a+0.5)) - ln((b+0.5)/(U-b+0.5))
a = harmful_group_df, b = unharmful_group_df
H = 400 (n_harmful_train_groups), U = 400 (n_unharmful_train_groups)
```

Kandidati su rangirani opadajuće po smoothed_log_odds (tie-break: veći harmful_group_df, zatim alfabetski po izrazu, radi potpune determinističnosti). Rangiranje se NE menja ručno nakon sortiranja.

## Regex konstrukcija

Svaki izraz je re.escape-ovan i uokviren (?<!\w)...(?!\w) granicama, tako da kratak izraz ne može pogoditi deo duže reči (npr. izraz 'cat' ne pogađa 'category').

## Validation grid i rezultati

Mreža: top_k_patterns=[25, 50, 100, 200, 400] x min_distinct_matches=[1, 2] = 10 konfiguracija (nijedna preskočena — 16139 kandidata je više od svake testirane top_k vrednosti).

```
 top_k_patterns  min_distinct_matches  precision  recall       f1      fpr  n_patterns_used
             25                     1   0.909091 0.12500 0.219780 0.020202               25
             25                     2   1.000000 0.12500 0.222222 0.000000               25
             50                     1   0.833333 0.12500 0.217391 0.040404               50
             50                     2   1.000000 0.12500 0.222222 0.000000               50
            100                     1   0.800000 0.12500 0.216216 0.050505              100
            100                     2   1.000000 0.12500 0.222222 0.000000              100
            200                     1   0.807692 0.13125 0.225806 0.050505              200
            200                     2   0.909091 0.12500 0.219780 0.020202              200
            400                     1   0.812500 0.16250 0.270833 0.060606              400
            400                     2   0.909091 0.12500 0.219780 0.020202              400
```

## Zaključana konfiguracija

Izabrano prema redosledu (max F1 -> max recall -> min FPR -> min broj obrazaca -> veći min_distinct_matches na potpunom tie-u): **top_k_patterns=400, min_distinct_matches=1** (validation F1=0.2708).

SHA256 zaključane liste od 400 obrazaca: `add95ede20d89015825c619821df840a12390456bd62896b0bfe80739fc16993` — potvrđeno nepromenjen pre i posle test evaluacije.

## Validation i test metrike (zaključana konfiguracija)

Regex baseline nema invalid izlaz (uvek vraća harmful ili unharmful), pa su **valid-only i end-to-end metrike identične** (invalid_count=0, invalid_rate=0).

**Validation** (259 redova): precision=0.8125, recall=0.1625, F1=0.2708, FPR=0.0606, FNR=0.8375, accuracy=0.4595, MCC=0.1505, TP=26 FP=6 FN=134 TN=93.

**Test** (227 redova, evaluiran TAČNO JEDNOM posle zaključavanja): precision=0.3333, recall=0.0315, F1=0.0576, FPR=0.0800, FNR=0.9685, accuracy=0.4229, MCC=-0.1076, TP=4 FP=8 FN=123 TN=92.

Test confusion matrix:
```
                pred_harmful  pred_unharmful
true_harmful               4             123
true_unharmful             8              92
```

**Značajan pad validation->test F1 (0.2708 -> 0.0576)** — objašnjenje ispod u sekciji Ograničenja.

## Broj obrazaca i najčešće vrste pogodaka (agregirano)

Zaključana lista ima 400 obrazaca, raspodela po ngram_size: {2: 217, 3: 179, 1: 4}.

**Ključan nalaz**: 221/400 (55%) zaključanih obrazaca su fragmenti dužine <=2 karaktera (bez razmaka) — npr. dvoslovni parovi kao 'a t', 'a n', 'i n'. **0 od 400** zaključanih obrazaca su 'realne reči' od >=4 karaktera. Prvi kandidat sa >=4 karaktera (bez razmaka) u celokupnoj rangiranoj listi je na rangu **558** — dakle NIJE dostignut ni jednom testiranom top_k vrednošću (maks. testiran top_k=400). Primeri takvih 'pravih' kandidata (rang>=558, ilustrativno, NISU u zaključanoj listi): ['targeting', 'request as it', 'strongly', 'derogatory', 'against ethical', 'for educational', 'let us', 'classified'].

## Uzrok: obfuskacija je u ovom datasetu isključivo harmful

Programski potvrđeno: svi redovi sa augmentation_type koji sadrži 'obfuscation' imaju final_label='harmful' u sva tri splita (train/validation/test) — obfuskacija je u v2 datasetu isključivo primenjena na harmful promptove (verovatno kao adversarial augmentacija za testiranje jailbreak-bypass-a). Neki obfuskacioni stilovi (npr. wide_spacing, vaporwave) ubacuju bukvalne razmake između svakog slova originalnog teksta. Pod propisanom jednostavnom whitespace-tokenizacijom, to razbija reči na pojedinačna slova, pa se karakter-bigrami/trigrami tih slova (npr. 'a t') pojavljuju u desetinama harmful train grupa i skoro nikad u unharmful grupama — ne zato što nose harmful semantiku, već zato što se ta specifična obfuskacija dešava samo na harmful sadržaju. Ovo NIJE bug u kodu — ovo je stvarno svojstvo dataset-a u kombinaciji sa propisanom jednostavnom normalizacijom (koja ne sme ručno dekodirati obfuskacije).

## FP/FN analiza (agregirano)

Recall na obfuskovanim harmful redovima naspram ne-obfuskovanih (razlika potvrđuje uzrok iznad):

- validation: 38.3% (obfuskacija) vs 3.0% (bez obfuskacije)
- test: 13.8% (obfuskacija) vs 0.0% (bez obfuskacije) — baseline praktično ne detektuje harmful sadržaj koji nije karakter-spacing-obfuskovan na test skupu.

FN (validation, 134 redova) po augmentation_type (top 5): {'original': 49, 'translation': 48, 'obfuscation_en': 17, 'obfuscation_tr': 9, 'obfuscation_promptonly_tr': 6}

FN (test, 123 redova) po augmentation_type (top 5): {'original': 50, 'translation': 48, 'obfuscation_promptonly_tr': 8, 'obfuscation_promptonly_en': 6, 'obfuscation_tr': 6}

FP (validation, 6 redova) po augmentation_type: {'translation': 3, 'original': 3}, po jeziku: {'en': 3, 'hi': 1, 'te': 1, 'it': 1}

FP (test, 8 redova) po augmentation_type: {'translation': 4, 'original': 4}, po jeziku: {'en': 4, 'id': 2, 'cs': 1, 'pt': 1}

Test greške nisu korišćene za dodavanje/uklanjanje regex obrazaca — konfiguracija je ostala zaključana pre i posle testa (potvrđeno SHA256 hash-om).

## Poređenje sa ostalim sistemima (test, end-to-end)

Pre poređenja programski potvrđeno: svih 5 sistema koristi identičnih 227 row_id vrednosti i istu final_label kolonu iz istog v2 test skupa. Metrike za sva 4 postojeća sistema su PRERAČUNATE iz njihovih sačuvanih test_results_full.csv fajlova istom compute_metrics_end_to_end funkcijom (nisu hardkodovane), bez ponovnog pokretanja bilo kog modela.

```
            System  precision  harmful_recall       f1  fpr      fnr  accuracy  invalid_rate
    Regex baseline   0.333333        0.031496 0.057554 0.08 0.968504  0.422907      0.000000
   Gemma zero-shot   0.725000        0.913386 0.808362 0.44 0.086614  0.757709      0.022026
Gemma initial LoRA   0.888889        0.944882 0.916031 0.15 0.055118  0.903084      0.000000
  Gemma sweep LoRA   0.927419        0.905512 0.916335 0.09 0.094488  0.907489      0.000000
        Qwen3Guard   0.945736        0.960630 0.953125 0.07 0.039370  0.947137      0.000000
```

## Ograničenja keyword/regex pristupa

- **Nema semantičkog razumevanja** — baseline ne razlikuje harmful i benign upotrebu istih reči/fraza, i ne generalizuje na parafraze.
- **Group-level statistika na ovom datasetu je dominirana artefaktom formatiranja, ne sadržajem**: pošto je obfuskacija u v2 datasetu isključivo harmful, a propisana jednostavna tokenizacija ne razlikuje stvarni razmak između reči od razmaka ubačenog unutar obfuskovane reči, ceo zaključani top-400 sastoji se od kratkih karakter-fragmenata, a ne od semantičkih harmful termina (prvi takav termin je na rangu 558, van dostignutog opsega).
- **Slaba generalizacija train->validation->test**: F1 pada sa 0.2708 (validation) na 0.0576 (test) jer se koja konkretna obfuskaciona pod-vrsta (npr. wide_spacing/vaporwave sa bukvalnim razmacima) nalazi u kom splitu razlikuje po slučaju uzorkovanja grupa, a ne po stabilnom semantičkom signalu.
- **Praktično nikakva detekcija ne-obfuskovanog harmful sadržaja na testu** (0.0% recall) — baseline je u praksi gotovo isključivo detektor 'da li tekst sadrži karakter-po-karakter razmaknutu obfuskaciju', ne detektor harmful sadržaja.
- **Nema obrade konteksta** (npr. 'the $x example' iz CLAUDE.md) — regex gleda prompt i response nezavisno, red po red, bez ikakvog multi-turn razumevanja.
- **Osetljivo na normalizaciju** — drugačija (agresivnija ili obfuskacija-svesna) normalizacija bi promenila kandidate i rezultate, ali bi izašla iz okvira 'jednostavnog' baseline-a specificiranog za ovaj eksperiment.

## Potvrda metodologije

Test skup NIJE korišćen za generisanje izraza (izrazi su izvedeni isključivo iz train.jsonl) niti za izbor top_k/min_distinct_matches (izabrano isključivo na validation skupu). Test je učitan i evaluiran TAČNO JEDNOM, nakon zaključavanja locked_regex_config.json/locked_patterns.csv, i SHA256 zaključane liste ostaje identičan pre i posle test evaluacije. Nijedan GPU model nije pokretan; ovaj notebook je čisto CPU (pandas/re/math).
