# Research log

One entry per step. Each entry answers the same questions so decisions stay
traceable: what problem we are solving, which candidate solutions were considered,
how they were compared, which one won, and why. Numbers here must come from runs
that are committed and reproducible; link the script or config that produced them.

**Plan:** [plan_v0.1.md](plan_v0.1.md) · [training_plan.md](training_plan.md) (SFT + RL) ·
**Decisions:** [0001 open-model selection](decisions/0001-open-model-selection.md) · [0002 training models and compute](decisions/0002-training-models-and-compute.md)

| Step | Problem | Status | Entry |
|---|---|---|---|
| 0 | Reproducible project scaffold | done | [steps/step0_scaffold.md](steps/step0_scaffold.md) |
| 1 | Load and inspect TREC Clinical Trials data | done | [steps/step1_data.md](steps/step1_data.md) |
| 2 | Baselines for patient–trial eligibility | planned | — |
| 3 | Turn eligibility criteria into a reasoning graph | planned | — |

## Entry template

```markdown
# Step N: <title>

**Date:** YYYY-MM-DD
**Problem:** what we are solving and why it matters for the end goal.
**Inputs / outputs:** what goes in, what must come out.
**Acceptance criteria:** testable conditions decided before running.

## Candidates
| ID | Method | Source / rationale | Cost |
|---|---|---|---|

## Evaluation setup
Data split, sample size, metrics, seeds, config/script used.

## Results
| Candidate | Metric 1 | Metric 2 | Notes |
|---|---|---|---|

## Decision
Which candidate wins and why; what was rejected and why.

## Open issues
Known limitations, things to revisit.
```
