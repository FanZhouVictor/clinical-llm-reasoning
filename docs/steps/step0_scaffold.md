# Step 0: Project scaffold

**Date:** 2026-11-13
**Problem:** Experiments must be reproducible on two machines (macOS laptop for
development, Linux/WSL2 GPU desktop for model runs) and checkable by others.
**Inputs / outputs:** none → an installable package, config system, tests, and CI.
**Acceptance criteria:**
1. `pip install -e ".[dev]"` succeeds in a fresh environment.
2. `ruff check .` and `pytest` pass locally.
3. GitHub Actions CI passes on Python 3.10 and 3.12.

## Candidates
| ID | Choice | Rationale |
|---|---|---|
| A | `src/` layout + `pyproject.toml` (hatchling) | Standard packaging; prevents importing uninstalled code |
| B | Flat scripts folder | Faster to start, but not installable or testable |

## Decision
A. Config is a YAML file (`configs/default.yaml`) with `CLINREASON_<KEY>`
environment overrides, so the same code runs on both machines by changing only
`CLINREASON_DATA_DIR` and the LLM endpoint. Data never lives inside the repository.

## Results
See the CI badge / workflow runs for the commit that added this entry.
