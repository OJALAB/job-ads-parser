# ESCO Skill Batch

Batchowe narzędzie do wyciągania skilli z ofert pracy i mapowania ich do lokalnego słownika ESCO.

Aktualny domyślny zakres:
- `Skills`
- `Transversal skills`

Architektura od początku zostawia miejsce na późniejsze rozszerzenie o `Knowledge`.

## Co robi

Narzędzie działa w dwóch krokach:

1. `build-index`
   Buduje lokalny indeks ESCO z oficjalnego eksportu CSV.
2. `extract-batch`
   Czyta oferty pracy partiami, wyciąga tylko wzmianki o skillach i przypisuje do kandydatów ESCO.

## Obsługiwane tryby ekstrakcji

- `ollama`
  Najprostszy start na Twoim serwerze. Wykorzystuje lokalne `http://127.0.0.1:11434`.
- `gliner`
  Docelowy extractor HF dla większej kontroli nad typem encji.
- `passthrough`
  Do testów, rerunów i sytuacji, gdy wzmianki o skillach są już wcześniej wyciągnięte.

## Obsługiwane tryby mapowania

- `lexical`
  Bez dodatkowych zależności. Działa od razu po instalacji pakietu.
- `embedding`
  Wymaga `sentence-transformers` i najlepiej modelu `BAAI/bge-m3`.
- `hybrid`
  Łączy embeddingi i scoring leksykalny.

## Instalacja

Minimalna:

```bash
python -m pip install -e .
```

Z embeddingami:

```bash
python -m pip install -e ".[embedding]"
```

Z GLiNER:

```bash
python -m pip install -e ".[gliner,embedding]"
```

## Przygotowanie danych ESCO

1. Pobierz oficjalny eksport ESCO CSV z:
   [https://esco.ec.europa.eu/en/use-esco/download](https://esco.ec.europa.eu/en/use-esco/download)
2. Wskaż plik CSV ze skillami przy budowie indeksu.

Narzędzie stara się automatycznie rozpoznać typowe kolumny takie jak:
- `conceptUri`
- `preferredLabel`
- `altLabels`
- `hiddenLabels`
- `description`
- `definition`
- `scopeNote`
- `skillType`
- `reuseLevel`
- `inScheme`

## Szybki start

Budowa indeksu bez embeddingów:

```bash
esco-skill-batch build-index \
  --esco-csv /path/to/esco_skills.csv \
  --output-dir /path/to/index
```

Budowa indeksu z embeddingami:

```bash
esco-skill-batch build-index \
  --esco-csv /path/to/esco_skills.csv \
  --output-dir /path/to/index \
  --embedding-model BAAI/bge-m3
```

Batch przez Ollamę:

```bash
esco-skill-batch extract-batch \
  --input /path/to/job_ads.jsonl \
  --output /path/to/results.jsonl \
  --index-dir /path/to/index \
  --text-field description \
  --id-field id \
  --extractor ollama \
  --ollama-model qwen3:14b \
  --mapping-backend hybrid \
  --top-k 5
```

Batch przez GLiNER:

```bash
esco-skill-batch extract-batch \
  --input /path/to/job_ads.jsonl \
  --output /path/to/results.jsonl \
  --index-dir /path/to/index \
  --text-field description \
  --id-field id \
  --extractor gliner \
  --gliner-model urchade/gliner_multi-v2.1 \
  --mapping-backend hybrid \
  --top-k 5
```

Tryb testowy `passthrough`:

```bash
esco-skill-batch extract-batch \
  --input /path/to/preextracted.jsonl \
  --output /path/to/results.jsonl \
  --index-dir /path/to/index \
  --extractor passthrough \
  --mentions-field skills_raw \
  --mapping-backend lexical
```

## Format wejścia

### JSONL

Każda linia to obiekt JSON. Domyślnie tekst brany jest z pola podanego w `--text-field`.

Przykład:

```json
{"id": "job-1", "description": "We need Python, SQL and communication skills."}
```

### CSV

Podaj `--text-field`, np. `description`.

### TXT

Każda linia jest traktowana jako osobna oferta.

## Format wyjścia

Wyjście to JSONL. Każdy rekord zawiera:
- identyfikator wejściowy
- listę wyciągniętych wzmiankowanych skilli
- listę dopasowań ESCO z `esco_uri`, `preferred_label`, `category`, `score`

## Rozszerzenie o Knowledge

Na etapie budowy indeksu wystarczy dodać:

```bash
--include-knowledge
```

Nie trzeba zmieniać architektury pipeline'u.

## Przykładowe pliki

W katalogu `examples/` są małe dane do szybkiego smoke testu.

## Gotowe przykłady do uruchomienia na serwerze

Minimalny smoke test bez modeli:

```bash
cd /path/to/job-offers-parser
bash examples/run_server_example.sh
```

Realny batch z Ollamą:

```bash
cd /path/to/job-offers-parser
export PYTHON_BIN=/opt/miniconda3/bin/python
export ESCO_CSV=/path/to/esco/skills.csv
export INPUT_FILE=/path/to/job_ads.jsonl
bash examples/run_server_ollama_example.sh
```

Jeśli chcesz użyć środowiska `llm`, ustaw po prostu właściwy interpreter w `PYTHON_BIN`.

## Testy

Testy nie wymagają `pytest` ani dostępu do sieci. Uruchomienie:

```bash
cd /path/to/job-offers-parser
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Albo przez gotowy skrypt:

```bash
cd /path/to/job-offers-parser
bash examples/run_tests.sh
```
