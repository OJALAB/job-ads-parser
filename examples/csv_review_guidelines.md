# CSV Review Guidelines

Ten plik opisuje, jak ręcznie walidować kandydatów skilli w CSV wygenerowanym przez:

```bash
PYTHONPATH=src python3 -m esco_skill_batch export-review-csv \
  --input .demo/review-queue.jsonl \
  --output .demo/review-queue.csv
```

## Co oznacza jeden wiersz

Jeden wiersz to **jedna unikalna fraza skilla**, a nie jedna oferta pracy.

Najważniejsze kolumny:

- `mention_raw`: fraza tak, jak pojawiła się w ofertach
- `mention_normalized`: znormalizowana forma do grupowania
- `canonical_mention`: docelowa forma, którą chcemy utrwalić po review
- `occurrence_count`: ile razy ta fraza pojawiła się w ofertach
- `example_context_1..5`: przykładowe konteksty z prawdziwych ofert
- `candidate_1_*`, `candidate_2_*`, `candidate_3_*`: najlepsze kandydaty ESCO
- `decision`: pole do uzupełnienia przez człowieka
- `selected_esco_uri`: URI wybranego konceptu ESCO
- `notes`: opcjonalny komentarz

## Dozwolone decyzje

W kolumnie `decision` wpisujemy tylko:

- `accept_esco`
- `no_match`
- `reject`

## Jak wypełniać

### 1. `accept_esco`
Użyj, jeśli fraza jest skillą i dobrze mapuje się do ESCO.

Wtedy:
- wpisz `accept_esco` w `decision`
- wklej URI do `selected_esco_uri`
- w `canonical_mention` zostaw najlepszą formę kanoniczną

Przykład:

- `mention_raw`: `zapytania SQL`
- `decision`: `accept_esco`
- `selected_esco_uri`: `http://data.europa.eu/esco/skill/sql`
- `canonical_mention`: `zapytania sql`

### 2. `no_match`
Użyj, jeśli fraza wygląda jak skill, ale nie ma sensownego dopasowania do ESCO.

Wtedy:
- wpisz `no_match` w `decision`
- zostaw `selected_esco_uri` puste
- opcjonalnie dopisz komentarz w `notes`

### 3. `reject`
Użyj, jeśli to nie jest skill do naszego zadania.

Typowe przykłady:
- benefity
- lokalizacja
- forma pracy
- ogólne formalności
- stanowisko zamiast umiejętności

Wtedy:
- wpisz `reject` w `decision`
- zostaw `selected_esco_uri` puste

## Zasady praktyczne

- Oceniaj frazę na podstawie `example_context_*`, nie tylko samego `mention_raw`.
- Jeśli `candidate_1_uri` jest wyraźnie poprawny, zwykle wybierz go.
- Jeśli kandydat ESCO jest błędny, ale fraza nadal jest skillą, wybierz `no_match`, nie `reject`.
- `canonical_mention` powinno być krótką, stabilną formą tej samej umiejętności.
- Nie wpisuj własnych nowych kolumn i nie zmieniaj nagłówków.

## Minimalny workflow

1. Wygeneruj kolejkę review JSONL.
2. Wyeksportuj CSV.
3. Uzupełnij tylko:
   - `decision`
   - `selected_esco_uri`
   - `canonical_mention`
   - `notes`
4. Zaimportuj CSV z powrotem:

```bash
PYTHONPATH=src python3 -m esco_skill_batch import-review-csv \
  --queue .demo/review-queue.jsonl \
  --input .demo/review-queue.csv \
  --output .demo/reviewed-queue.jsonl
```

5. Zbuduj korpus do fine-tuningu:

```bash
PYTHONPATH=src python3 -m esco_skill_batch build-finetune-corpus \
  --input examples/kprm_warszawa_live_10.jsonl \
  --reviewed-queue .demo/reviewed-queue.jsonl \
  --output-dir .demo/review-corpus \
  --text-field skills_text
```
