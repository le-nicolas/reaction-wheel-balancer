# Sim-to-Real Discrepancy Log

Use this file as a living log during hardware bring-up.

## Session Metadata

| Field | Value |
|---|---|
| Date | |
| Operator | |
| Robot ID | |
| Firmware commit | |
| Runtime commit | |
| Controller mode/profile | |
| Export command | |
| Generated header hash | |

## Replay Inputs

| Field | Value |
|---|---|
| Hardware trace CSV | |
| Benchmark CSV | |
| Sensitivity summary CSV | |
| Replay command | |

## Quantitative Results

| Metric | Target / Threshold | Observed | Pass |
|---|---:|---:|---|
| `sim_real_consistency_mean` | `>= 0.60` | | |
| `sim_real_traj_nrmse_mean` | `<= 0.75` | | |
| `survival_rate` | `1.00` | | |
| `crash_rate` | `0.00` | | |
| `wheel_over_hard_mean` | `0.00` | | |
| `mean_sat_rate_abs` | `<= 0.90` | | |

## Symptoms and Root-Cause Hypothesis

| Symptom | First Seen In | Likely Driver | Evidence | Action |
|---|---|---|---|---|
| | | | | |

## Parameter Changes Applied

| Parameter | Old | New | Why | Expected Effect |
|---|---:|---:|---|---|
| | | | | |

## Verification After Change

| Check | Result | Notes |
|---|---|---|
| Re-export parity (`final/test_export_parity.py`) | | |
| Sim benchmark rerun | | |
| Hardware replay rerun | | |
| Safety latch behavior | | |

## Open Items

- [ ] 
