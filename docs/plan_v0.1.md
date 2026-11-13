# Implementation plan: clinical-llm-reasoning v0.1

**Date:** 2026-11-13 · **Status:** Steps 0–2 approved; Steps 3–8 proposed

## Goal

Screen patients for clinical trials with an LLM that (1) turns free-text eligibility
criteria into an explicit reasoning graph, (2) judges each criterion with calibrated
uncertainty, (3) identifies which missing patient information actually changes the
decision, and (4) shows the reasoning path behind every decision.

## Data

- **TREC Clinical Trials 2021** (75 patient notes, 35,832 judged patient–trial pairs)
  and **2022** (50 notes, 35,394 pairs). Labels: 2 = eligible, 1 = excluded,
  0 = not relevant.
- Trials: the official ClinicalTrials.gov snapshot of 2021-04-27 used by both tracks
  (5 zip parts, ~1.8 GB compressed). The live registry is *not* used, because trials
  have been edited since judging.
- Split: develop and calibrate on 2021, report on 2022 (different patients, so this
  also tests generalisation).
- A full 2022 run (~35k pairs) is too slow for one GPU; evaluation uses a fixed,
  stratified sample (e.g. up to 30 judged trials per patient). Sample size is reported.

## Compute

One NVIDIA RTX 4000 Ada (20 GB) desktop; models served with vLLM in WSL2 Ubuntu 24.04
through an OpenAI-compatible endpoint, so the backend (vLLM, Ollama, hosted API) is a
config change. Model choice: see
[decisions/0001-open-model-selection.md](decisions/0001-open-model-selection.md).

## Steps

| Step | What changes | Why | Acceptance criteria |
|---|---|---|---|
| **0. Scaffold** | `src/` package, `pyproject.toml`, config with env overrides, pytest, ruff, GitHub Actions, research log | Reproducibility on two machines | Fresh install works on macOS and WSL; lint + tests pass locally and in CI |
| **1. Data** | Download script with SHA-256 manifest; parsers producing `patients.jsonl`, `labels.jsonl`, `trials.jsonl` (inclusion/exclusion split); exploration notebook; data card | Everything depends on it; the data must be inspectable by a human | Counts match the official files (75/50 notes; 35,832/35,394 pairs); ≥99% of judged trials found in the snapshot; ≥95% split into non-empty criteria, failures counted and shown; unit tests on 5 hand-checked trials |
| **2. Baselines + model pilot** | B1: ask the LLM directly "eligible?". B2: judge each criterion separately, combine with a flat rule. Pilot three models on ~200 2021 pairs | Something to beat; choose the main model with data | Both baselines run end to end; 3-class macro-F1 and NDCG@10 / P@10 over judged trials; model chosen by macro-F1 per GPU-hour and logged |
| **3. Criteria → graph** | LLM builds a conditional reasoning graph (GPS-style JSON, schema-constrained decoding, validate + repair); deterministic evaluator with three-valued logic (met / not met / unknown) | Core method (adapted from GPS, pp. 4–5, prompt in App. C.1 p. 17) | ≥95% schema-valid graphs; 30 hand-checked gold graphs in `benchmarks/gold_graphs/`; evaluator unit tests cover the logic tables |
| **4. Graph screening** | Judge each condition node, evaluate the graph, output decision + path | Main result and explanation | Results on 2022; target: macro-F1 ≥ B2, with a decision path for every pair |
| **5. Conformal abstention** | Split conformal on the eligibility label; calibrate on 2021, test on 2022 | Uncertainty quantification | Coverage within ±3 points of target at 80/90/95%; abstention-vs-accuracy curves; shift reported honestly |
| **6. Decisive-gap questioning** | Hide sentences from the note; simulated patient answers; GPS ordering (Eq. 1–2, p. 5) vs random, ask-all, LLM-chooses | Information seeking | Accuracy vs number of questions; target: reach full-note accuracy with fewer questions |
| **7. Faithfulness** | Change the evidence a decision path cites; does the decision change? | Measured explainability | Flip rate vs a random-evidence control |
| **8. GRPO (optional)** | LoRA/QLoRA GRPO on a small model with an accuracy × efficiency reward (GPS Eq. 13, p. 7) | Reinforcement learning | Beats the same model untrained on Step 6 metrics |

## Step 3 candidates (to be decided with data)

| ID | Graph-building method | Notes |
|---|---|---|
| A | Flat rule-based: inclusion = AND of all bullets, exclusion = NOT any | No LLM; same structure as baseline B2 |
| B | Prompted LLM with an adaptation of the GPS DAG-extraction prompt + JSON-schema-constrained decoding + validation/repair | Starting point; handles nesting, "unless", "only if" |
| C | Large model builds graphs once, hand-checked, then distilled into the 7–8B model | Only if B's graph quality is too low |
| D | Train a GPS-style Reasoner with GRPO | Step 8, small model only. The GPS repository releases training code, not a trained checkpoint, and the original training used 8×A800 GPUs (GPS p. 22) |

Graph storage: generated graphs are cached per trial at
`<data_dir>/graphs/<model>/<prompt-version>/<NCT-ID>.json` with model, prompt version
and timestamp (not in git; regenerable). Hand-checked gold graphs are committed under
`benchmarks/gold_graphs/`.

## Working rules

- Every step gets a research-log entry: problem, candidates, evaluation, winner, why
  ([research_log.md](research_log.md)).
- The README only reports numbers produced by committed code and configs.
- Data never enters git; the SHA-256 manifest does.
- Small commits, one step at a time.

## Risks

- TREC labels are per (patient, trial) pair, not per criterion; criterion-level
  quality relies on hand review in Step 3.
- The simulated patient in Step 6 is an approximation; stated as such.
- Quantized models and inference settings affect results; versions and checkpoint
  revisions are pinned and reported.
