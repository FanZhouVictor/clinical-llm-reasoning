# Decision 0001: Open-weight model selection

**Date:** 2026-11-13 · **Status:** shortlist decided; main model chosen by the Step 2 pilot

## Problem

Choose open-weight LLMs for criterion judging, criteria-to-graph construction and
clarification, plus a small model for later GRPO fine-tuning, under these constraints:

- **Hardware:** one RTX 4000 Ada, 20 GB VRAM (~18.6 GB free), FP8-capable; vLLM in WSL2.
- **Workload:** prompts of ~2k–8k tokens (patient note + trial criteria); thousands of
  pairs, so throughput matters; JSON-schema-constrained output.
- **Licence:** must allow research use and publishing results from a public MIT repo.
- **Comparability:** the GPS paper used Qwen2.5-7B-Instruct.

## Candidates

Memory = weights + BF16 KV cache for 4 sequences × 8k tokens, **calculated, not
measured**; add ~2–3 GB runtime overhead. Repository existence, licence and creation
date were checked against the Hugging Face API on 2026-11-13.

| Model (HF id) | Created on HF | Params | Quantized checkpoint | Est. memory | Licence | Verdict |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3-8B` | 2025-04-27 | 8.2B | `Qwen/Qwen3-8B-AWQ` (official) | ~10 GB | Apache-2.0 | **Pilot: main candidate** |
| `Qwen/Qwen3.5-9B` | 2026-02-27 | ~9B | `RedHatAI/Qwen3.5-9B-FP8-dynamic` | ~14 GB | Apache-2.0 | **Pilot: newer challenger** |
| `Qwen/Qwen2.5-7B-Instruct` | 2024-09 | 7.6B | `Qwen/Qwen2.5-7B-Instruct-AWQ` (official) | ~7 GB | Apache-2.0 | **Pilot + always reported (GPS comparability)** |
| `Qwen/Qwen3-14B` | 2025-04 | 14.8B | `Qwen/Qwen3-14B-AWQ` (official) | ~14 GB | Apache-2.0 | Scale check later |
| `google/medgemma-1.5-4b-it` | 2026-01-07 | 4B | — (BF16 preferred) | ~12 GB | HAI-DEF, gated | Later: does medical pretraining help? |
| `Qwen/Qwen3-1.7B` | 2025-04-27 | 1.7B | BF16 for training | ~7 GB | Apache-2.0 | **GRPO model (Step 8)** |
| `openai/gpt-oss-20b` | 2025-08-04 | 21B (3.6B active) | native MXFP4 | ~14 GB | Apache-2.0 | Not now: reasoning-first, long outputs |
| `ibm-granite/granite-4.2-8b` | 2026-08-07 | 8B | official FP8 | ~14 GB | Apache-2.0 | Not now |
| `mistralai/Ministral-3-14B-Instruct-2512` | 2025-10-31 | ~14B | community AWQ only | ~15 GB | Apache-2.0 | Not now: community quant, little headroom |
| DeepSeek-R1-0528-Qwen3-8B, Phi-4-mini, Llama-3.1-8B | — | — | — | — | various | Not now: reasoning-heavy, weaker, or no advantage over Qwen |

## Evidence (verified vs unverified)

- **Verified** (model card, checked 2026-11-13): Qwen3.5-9B FP8-dynamic keeps
  88.9% IFEval prompt-strict vs 90.3% for BF16, and 82.4% vs 82.5% MMLU-Pro
  ([RedHatAI card](https://huggingface.co/RedHatAI/Qwen3.5-9B-FP8-dynamic)).
- **Unverified** (gated card not checked): MedGemma 1.5 4B reports 69.1% MedQA.
- No public benchmark establishes which model is best at TREC eligibility screening,
  so the choice is made empirically.

## Decision

1. **Step 2 pilot** on ~200 stratified 2021 pairs with both baselines and three models:
   Qwen3-8B-AWQ (thinking off), Qwen3.5-9B-FP8, Qwen2.5-7B-Instruct-AWQ.
2. **Main model** = best 3-class macro-F1 per GPU-hour in the pilot; the result and
   reasoning are added to this record.
3. **Qwen2.5-7B-Instruct** is reported in final results regardless (GPS comparability).
4. **Qwen3-1.7B** is the GRPO model for Step 8.
5. MedGemma 1.5 and Qwen3-14B are optional follow-up checks.

## Risks

- Thinking modes produce long outputs that cut throughput; thinking mode and output
  budgets are fixed across compared runs.
- Quantization can change accuracy; quantized runs are labelled as such.
- vLLM, Transformers and checkpoint revisions are pinned per run.
- Licences: MIT applies to this code, not to model weights. MedGemma (HAI-DEF) is gated
  and prohibits automated clinical decisions; offline research use on TREC only.

## Pilot results

_To be filled in by Step 2._
