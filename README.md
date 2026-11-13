# clinical-llm-reasoning

Graph-guided LLM reasoning over clinical text: structured eligibility reasoning,
calibrated uncertainty, and explainable decision paths.

> **Status: early development.** No results are reported yet. This README will be
> updated with reproducible numbers as each component is evaluated.

## Goal

Clinical decisions often depend on conditional criteria applied to incomplete patient
information. This project studies LLM systems that

1. compile free-text criteria into explicit conditional reasoning graphs,
2. judge each criterion against patient text with calibrated (conformal) uncertainty,
3. identify which missing information actually changes the decision, and
4. expose the reasoning path behind every decision.

## Planned components

| Component | Description |
|---|---|
| Criteria → graph | Parse trial eligibility criteria into a conditional reasoning graph |
| Criterion judging | Met / not met / unknown per criterion from patient text |
| Conformal abstention | Prediction sets per criterion; abstain or query when uncertain |
| Decisive-gap selection | Rank missing criteria by their effect on the eligibility decision |
| Explanations | Graph path as explanation, with counterfactual faithfulness checks |
| RL fine-tuning | GRPO on small open models (optional) |

## Data

Experiments use publicly available data only, starting with the
[TREC Clinical Trials Track](https://www.trec-cds.org/) (2021/2022) patient
descriptions and relevance judgments over ClinicalTrials.gov records.
Datasets are downloaded by scripts and are not stored in this repository.

## Related work

The method draws on ideas from graph-guided proactive information seeking
(GPS, ICLR 2026) and Backjump-on-Graph (ICML 2026).

## License

MIT
