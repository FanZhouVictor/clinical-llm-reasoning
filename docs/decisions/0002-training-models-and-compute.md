# Decision 0002: Which Qwen models to fine-tune (SFT) and train with RL, and the cost

**Date:** 2026-11-13 · **Status:** proposed; time estimates to be calibrated by a
1.7B smoke test

## Compute available

| Pool | GPUs | Memory | Role |
|---|---|---|---|
| Desktop | 1 × RTX 4000 Ada | 20 GB | Development, smoke tests |
| Delta A | 4 × A100 | 40 GB each | Main training (≤ 8B), full evaluation |
| Pool B | 2–3 × A100 | 80 GB each | 32B teacher inference, optional 14B training |

Allocation: not a constraint. Wall-clock time (including queue) is the real limit.

## Workload assumptions

- **SFT data:** ~12k rejection-sampled teacher traces from 2021 pairs (all eligible and
  excluded pairs plus a sample of not-relevant), ~1.2k tokens each → ~15M tokens per
  epoch, 2 epochs → ~30M tokens.
- **GRPO:** 32 prompts × 8 samples per step, ~400 output tokens per sample, ~400 steps
  (≈ one pass over the training prompts). Single-turn eligibility decisions.
- **Throughput:** ~100 TFLOP/s effective per A100 for training (~1/3 of BF16 peak);
  LoRA training ≈ 5·N·T FLOPs with gradient checkpointing, plus 2·N·T for the GRPO
  reference forward pass; generation with vLLM. A 2× inefficiency allowance is included.
- These are **arithmetic estimates, ±2×**, not measurements.

## Estimates

| Model | Fits where (LoRA) | SFT (2 epochs) | GRPO (400 steps) | Role |
|---|---|---|---|---|
| Qwen3-1.7B | Desktop (QLoRA smoke test), 1–2 × A100-40 | ~1–2 GPU-h · **< 1 h** wall | ~10–15 GPU-h · **3–5 h** wall | Fast ablations; smoke-test every pipeline change |
| Qwen3-4B | 4 × A100-40 | ~3–4 GPU-h · ~1 h | ~20–30 GPU-h · 5–8 h | Optional middle point of the scale curve |
| **Qwen3-8B** | 4 × A100-40 (1 GPU vLLM + 3 GPUs training) | ~7 GPU-h · **~2 h** | ~40–60 GPU-h · **10–15 h** | **Main trained model** |
| Qwen2.5-7B-Instruct | 4 × A100-40 | ~6 GPU-h · ~2 h | ~40–55 GPU-h · 10–14 h | One run for GPS comparability (GPS trained this model) |
| Qwen3-14B | 3 × A100-80 (1 vLLM + 2 training) | ~12 GPU-h · ~4–6 h | ~60–90 GPU-h · **~1 day** | Optional scale check |
| Qwen3-32B | 2 × A100-80 (or 4 × A100-40), tensor parallel | not planned | not planned | **Teacher and upper bound, inference only** |

**Other costs:**
- Teacher traces (Qwen3-32B, ~18k prompts × 2 samples): ~3–6 h wall on 2 × A100-80.
- Full 2022 evaluation per model: ~1 h for direct judging (B1-style); ~5–10 h for
  per-criterion judging (B2 / graph methods, ~10 calls per pair).
- Multi-turn clarifier RL (Step 6 setting, with a simulated patient) costs roughly
  3–5× single-turn GRPO: ~1.5–3 days for 8B.

**Whole Step 8 programme** (1.7B ladder with ablations, 8B base / SFT / GRPO-only /
SFT+GRPO, one Qwen2.5-7B run, teacher data, evaluations): roughly **300–450 A100
GPU-hours**, about **1–2 weeks of wall-clock** including queue waits.

For scale: Backjump-on-Graph's multi-turn RL stage used 562.4 GPU-hours on 8 × H100
(BoG p. 16). Our main runs are single-turn with short outputs, hence much cheaper.

## Not chosen for training

| Model | Reason |
|---|---|
| Qwen3-32B | Training cost and memory high, small expected value; used as teacher instead |
| Qwen3.5-9B | Newer hybrid architecture; training-tool support less mature (unverified) — keep for inference comparisons only |
| Qwen3-30B-A3B (MoE) | LoRA/RL on MoE adds engineering risk without a clear benefit here |
| Full-parameter fine-tuning of 8B | Needs ~128 GB for weights + optimizer state; LoRA is sufficient and is what GPS used (LoRA rank 64, GPS p. 22) |

## Decision

1. Train **Qwen3-1.7B** and **Qwen3-8B** with SFT and GRPO (base / SFT / GRPO-only /
   SFT+GRPO); one **Qwen2.5-7B-Instruct** SFT+GRPO run for GPS comparability.
2. **Qwen3-32B** is the teacher and prompted upper bound (inference only).
3. **Qwen3-14B** training only if the 1.7B → 8B trend suggests scale matters.
4. Before any cluster run: a 1.7B smoke test (20–50 GRPO steps) on the desktop; its
   measured step time replaces the estimates above in this record.

## Measured costs

_To be filled in from logged runs._
