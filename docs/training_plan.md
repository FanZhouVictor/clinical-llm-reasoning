# Training plan: SFT and RL (Step 8)

**Date:** 2026-11-13 · **Status:** proposed · Model and compute choices:
[decisions/0002-training-models-and-compute.md](decisions/0002-training-models-and-compute.md)

## Question

Can supervised fine-tuning (SFT) and reinforcement learning (GRPO) make a small open model
a better clinical-trial eligibility judge than a larger prompted model, while spending
fewer tokens per decision?

## What is trained

The **eligibility judge**: given a patient note and a trial's criteria, output short
per-criterion verdicts (met / not met / unknown, with the evidence sentence) and a final
label (eligible / excluded / not relevant) in a fixed JSON format. The same judge is
then reused inside the graph pipeline (Step 4) and conformal abstention (Step 5).

| Model | Role | Runs |
|---|---|---|
| Qwen3-1.7B | Small model; fast ablations | base · SFT · GRPO-only · SFT+GRPO + ablations |
| **Qwen3-8B** | Main trained model | base · SFT · GRPO-only · SFT+GRPO |
| Qwen2.5-7B-Instruct | GPS comparability (the model GPS trained) | base · SFT+GRPO |
| Qwen3-14B | Optional, only if 1.7B → 8B shows a clear scale trend | SFT+GRPO |
| Qwen3-32B | Teacher and prompted upper bound; **not trained** | inference only |

## Data splits (no leakage into the test set)

| Split | Patients | Used for |
|---|---|---|
| Train | 60 of the 75 patients in 2021 (fixed random split by patient) | SFT traces and GRPO prompts |
| Validation / calibration | the other 15 patients in 2021 | Checkpoint selection, reward tuning, conformal calibration of trained models |
| Test | all 50 patients in 2022 | Reported results only, never used for tuning |

Splits are by **patient**, so no patient appears in two splits. Trials judged in both
2021 and 2022 are counted; test results are reported with and without them.

## Stage 8a: SFT

1. **Teacher traces.** Qwen3-32B (BF16, tensor parallel on 2 × A100-80) judges a set of
   train-split pairs balanced across labels: all eligible and excluded pairs plus an
   equal-sized sample of not-relevant pairs. It draws 2 samples per pair at
   temperature 0.7.
2. **Rejection filter.** Keep a trace only if its final label matches the TREC label and
   its JSON is valid (the same idea as BoG's SFT on reference-guided trajectories, p. 4).
   Report the acceptance rate per label. Hand-check 50 kept traces for "right label,
   wrong reasoning".
3. **Training.** LoRA (rank 64, as GPS used, p. 22), loss on the completion only, 2 epochs,
   max length 4,096 tokens, cosine learning-rate schedule. Checkpoint chosen by
   validation macro-F1.

Expected size: ~12k accepted traces (~30M tokens over 2 epochs). This is an estimate;
the acceptance rate is measured.

## Stage 8b: GRPO

- **Start point:** the SFT checkpoint (main) or the base model (GRPO-only ablation).
- **Prompts:** train-split pairs, balanced by label, so the policy cannot win by always
  answering "not relevant".
- **Reward** (verifiable from TREC labels). Multiplicative accuracy gating follows GPS
  (Eq. 13, p. 7); the length term follows BoG's hybrid reward (p. 5):

  `r = r_acc · (1 + λ · r_eff) + r_fmt`

  | Term | Definition |
  |---|---|
  | `r_acc` | 1 if the final label equals the TREC label, else 0 |
  | `r_eff` | `max(0, 1 − tokens / budget)`: shorter correct decisions score higher |
  | `r_fmt` | +0.1 if the JSON is valid and every verdict cites evidence; −0.5 if invalid |
  | `λ` | Efficiency weight, tuned on validation (0, 0.2, 0.5) |

- **Settings (starting values):** 32 prompts × 8 samples per step, ~400 steps, max
  completion 512 tokens, KL coefficient 0.02, LoRA rank 64. Uses Hugging Face TRL's GRPO
  trainer with a vLLM generation server (1 GPU generates, the rest train).
- **Logged every step:** mean reward and each term, accuracy, completion length, KL.

## Stage 8c: Evaluation and ablations

All on the full 2022 test set.

| Comparison | Answers |
|---|---|
| base vs SFT vs GRPO-only vs SFT+GRPO (1.7B and 8B) | What each training stage adds |
| trained 1.7B / 8B vs prompted Qwen3-8B and Qwen3-32B | Can a small trained model match a large prompted one? |
| λ = 0 vs tuned λ | Does the efficiency reward cut tokens without losing accuracy? |
| SFT data size (25 / 50 / 100%) at 1.7B | How much teacher data is needed |
| Qwen2.5-7B SFT+GRPO vs base | Comparison with the model GPS trained |

**Metrics:** 3-class macro-F1; eligible-vs-excluded F1; NDCG@10 and P@10; tokens and
seconds per decision; conformal coverage when the trained judge is used in Step 5.

## Acceptance criteria

1. Every run is reproducible from a committed config and seed, and logs GPU-hours.
2. SFT 8B beats base 8B on 2022 macro-F1.
3. SFT+GRPO beats SFT-only on 2022 macro-F1, or matches it with fewer tokens per decision.
4. Results are reported whether or not "small trained ≥ large prompted" holds. It is a
   hypothesis, not a target to tune toward on the test set.

## Order of work and estimated time

Estimates are ±2× and are replaced by measured numbers in decision 0002.

| # | Task | Where | Est. wall-clock |
|---|---|---|---|
| 1 | Judge prompt + output parser + reward functions with unit tests (reuses Step 2) | Mac / CI | — |
| 2 | Smoke test: 1.7B SFT on 200 traces + 20–50 GRPO steps | Desktop | 1–2 h |
| 3 | Teacher traces with Qwen3-32B | 2 × A100-80 | 3–6 h |
| 4 | 1.7B: SFT, GRPO-only, SFT+GRPO, λ and data-size ablations | Delta 4 × A100-40 | ~1–2 days |
| 5 | 8B: SFT, GRPO-only, SFT+GRPO | Delta 4 × A100-40 | ~2–3 days |
| 6 | Qwen2.5-7B SFT+GRPO | Delta 4 × A100-40 | ~1 day |
| 7 | Full 2022 evaluation of all checkpoints | Delta | ~1 day |
| 8 | Optional Qwen3-14B | 3 × A100-80 | ~1–2 days |

Stage 8 needs only the Step 2 judging prompt and parser, so it can run in parallel with
Steps 3–7 once Step 2 is done.

## Engineering

- `scripts/train_sft.py`, `scripts/train_grpo.py` (TRL + PEFT), `scripts/make_teacher_traces.py`
- `configs/train/*.yaml` (one file per run), `slurm/*.sbatch` (cluster jobs)
- Logging to local TensorBoard/CSV; no external tracking service unless approved.
- Checkpoints stay on cluster storage; only metrics and small result files are copied back.
  LoRA adapters may later be published on Hugging Face (decided separately).

## Risks

| Risk | Mitigation |
|---|---|
| Reward hacking (e.g. always "not relevant") | Label-balanced prompts; monitor per-label accuracy |
| Teacher gives the right label with wrong reasoning | Hand-check 50 traces; evidence citation required by `r_fmt` |
| TREC label noise | Report it; no tuning on the test set |
| Overfitting to 60 training patients | Validation on held-out 2021 patients; early stopping |
| Leakage through trials shared across years | Report with and without shared trials |
| Long thinking-mode outputs | Thinking off; 512-token completion cap |
