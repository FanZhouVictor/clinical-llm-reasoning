# Step 1: Load and inspect the TREC Clinical Trials data

**Date:** 2026-11-13
**Problem:** Every later step needs clean, verifiable patient, label and trial tables,
and a human-readable view of what the data actually looks like.
**Inputs / outputs:** official TREC CT 2021/2022 files → `patients.jsonl`,
`labels.jsonl`, `trials.jsonl`, `stats.json` in `<data_dir>/processed/`, plus the
exploration notebook [`notebooks/01_explore_trec.ipynb`](../../notebooks/01_explore_trec.ipynb).

**Acceptance criteria** (fixed before running):
1. Patient and label counts exactly match the official files.
2. ≥99% of judged trials found in the snapshot.
3. ≥95% of judged trials split into labelled inclusion/exclusion sections; failures
   counted and shown.
4. Unit tests on hand-checked real trials; lint + tests pass locally and in CI.
5. Package installs and runs inside WSL on the GPU desktop.

## Data card

| Item | Value |
|---|---|
| Source | [trec-cds.org](https://www.trec-cds.org/) (topics, ClinicalTrials.gov snapshot 2021-04-27, 5 zips, 1.8 GB) and [trec.nist.gov](https://trec.nist.gov/data/trials/) (qrels) |
| Integrity | SHA-256 of all 9 files in [`manifests/trec_ct.sha256`](../../manifests/trec_ct.sha256) |
| Patients | 2021: 75 · 2022: 50 (synthetic admission-note style descriptions) |
| Patient note length | median 117 words (51–218) |
| Labelled pairs | 2021: 35,832 · 2022: 35,394 |
| Labels 2021 | eligible 5,570 · excluded 6,019 · not relevant 24,243 |
| Labels 2022 | eligible 3,939 · excluded 3,036 · not relevant 28,419 |
| Judged trials per patient | median 388 not relevant, 52 excluded, 60 eligible |
| Unique judged trials | 48,714 — all found in the snapshot (0 missing) |
| Criteria length | median 140 words, mean 238; 2.7% of trials exceed 1,000 words (max 2,539) |
| Structured fields | minimum age 89.8% · maximum age 53.0% · sex restricted 15.8% · healthy-volunteer flag 98.0% |

**Criteria section split** (judged trials):

| Status | Trials | Share |
|---|---|---|
| both sections | 45,561 | 93.5% |
| inclusion only | 1,156 | 2.4% |
| exclusion only | 12 | 0.0% |
| unsectioned (free prose, no headers) | 1,974 | 4.1% |
| no criteria text | 11 | 0.0% |
| **sectioned total** | | **95.9%** |

## Candidates and decisions

| Decision | Candidates | Chosen | Why |
|---|---|---|---|
| Trial text source | live ClinicalTrials.gov API · official 2021-04-27 snapshot | snapshot | Labels were judged against this version; trials have been edited since |
| Patient ids | topic number · `year-topic` | `year-topic` | Topic numbers restart each year; the 2022 file's header also says "2021" |
| Header detection | exact "Inclusion/Exclusion Criteria:" · broader pattern | broader pattern | Exact match gave 95.0% sectioned; adding observed variants ("Inclusion:", bulleted/numbered headers, qualifiers such as "Major/Infant/Recipient", "Criteria for exclusion", "Non-inclusion" = exclusion, inline "Exclusion criteria included") raised it to 95.9% |
| False-positive guard | — | connecting words ("for", "of", …) cannot be qualifiers; a "."-ended header must end the line | A wrapped prose line ("…eligible for inclusion. Exclusion criteria include…") was misread as a header |
| Long header lines | drop header text · keep it | keep when > 8 words and no colon | Such lines carry criterion text ("…for controls males and females of 18 years…") |
| Unsectioned trials | force a guess · keep whole text | keep whole `criteria_text`, both sections empty | Avoids silently mislabelling exclusions as inclusions; later steps read the full text |

## Results against acceptance criteria

1. Counts match: 75/50 patients; 35,832/35,394 pairs — **PASS**
2. 48,714 / 48,714 judged trials found — **PASS**
3. 95.9% sectioned; failures counted (table above) and shown in the notebook — **PASS**
4. 80 tests on 7 real trials + synthetic header cases; CI green — **PASS**
5. Installed and run in WSL2 Ubuntu 24.04 on the desktop — **PASS**

## Observations for later steps

- Most judged pairs are *not relevant*; *eligible vs excluded* is the hard distinction.
- Prompt length is driven by trial criteria, not patient notes.
- Age and sex are structured and can be checked by code before any LLM call.
- Some criteria are points-based scores or nested logic (e.g. NCT02006511: "at least
  8 points" from weighted risk factors). A flat AND/OR list cannot represent these;
  Step 3's graph format must.
- Another unsectioned style: oncology (PDQ) records use "DISEASE CHARACTERISTICS:" /
  "PATIENT CHARACTERISTICS:" blocks.

## Reproduce

```bash
python scripts/download_trec_ct.py
python scripts/build_trec_tables.py
pip install -e ".[notebook]"
python notebooks/build_01_explore_trec.py
jupyter nbconvert --to notebook --execute --inplace notebooks/01_explore_trec.ipynb
```
