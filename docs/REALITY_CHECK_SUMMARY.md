# Reality Check Summary (Metrics-First)

This file tracks measurable evidence for robustness and sim-to-real readiness using saved artifacts.

## Artifact sources

- Fast benchmark snapshot (`2026-02-21`): `final/results/benchmark_20260221_100424.csv`
- Nightly-long baseline sweep (`2026-02-21`): `final/results/benchmark_20260221_140044.csv`
- MPC fast-pr retest (`2026-02-21`): `final/results/benchmark_20260221_141746.csv`
- MPC longer-run retest (`2026-02-21`): `final/results/benchmark_20260221_142539.csv`
- Current payload A/B run (`2026-02-21`): `final/results/benchmark_20260221_144534.csv` and `final/results/benchmark_20260221_144433.csv`
- Targeted `R_du_by`/payload sweep (`2026-02-21`): `final/results/open_boundaries_20260221_142334.csv`
- Targeted long-horizon drift sweeps (`2026-02-21`): `final/results/open_boundaries_long_20260221_142430.csv`, `final/results/open_boundaries_drift_20260221_142520.csv`
- Historical MPC snapshot (`2026-02-16`): `final/results/benchmark_20260216_011044.csv`
- Sensitivity reference run (`2026-02-15`): `final/results/benchmark_20260215_215203.csv`
- Hardware-readiness report (`2026-02-15`): `final/results/readiness_20260215_220928.json`

## 1) Controller robustness snapshot

Scope: baseline rows, `mode_default / nominal / default`, benchmark artifact `benchmark_20260221_100424.csv`.

| Controller family | Survival rate | Crash rate | Composite score |
|---|---:|---:|---:|
| `baseline_mpc` | 1.000 | 0.000 | 86.793 |
| `current` | 1.000 | 0.000 | 81.862 |
| `hybrid_modern` | 1.000 | 0.000 | 80.727 |
| `baseline_robust_hinf_like` | 1.000 | 0.000 | 80.232 |
| `paper_split_baseline` | 1.000 | 0.000 | 69.169 |

## 2) MPC status (historical underperformance vs current tuning)

| Artifact date | Artifact | `baseline_mpc` survival | `baseline_mpc` crash | `baseline_mpc` score |
|---|---|---:|---:|---:|
| 2026-02-16 | `benchmark_20260216_011044.csv` | 0.625 | 0.375 | 37.837 |
| 2026-02-21 | `benchmark_20260221_100424.csv` | 1.000 | 0.000 | 86.793 |

Interpretation:
- The earlier 0.375 crash-rate snapshot is real and preserved for traceability.
- Current defaults no longer show that failure on the fast benchmark profile.
- `baseline_mpc` remains a comparator baseline, not the recommended deployment family.

## 3) Structural root-cause postmortem (model integrity)

Confirmed structural faults (fixed):
- Unrooted base: gravity coupling into pitch/roll dynamics was effectively missing until the world weld was restored (`final/final.xml`, `lock_base_to_world`).
- Miswired actuator: `base_y_force` command path was not driving the intended slider axis; corrected to `joint="base_y_slide"` in `final/final.xml`.

Elimination record (ruled out with runs/data before model fix):
- Gain magnitude
- Delta-u limits
- Kalman filter bias
- Linearization point
- Q-matrix zeros
- Timestep mismatch
- COM offset
- Clipping asymmetry

Known residual watchpoints:
- Smooth profile keeps a very cheap base-y slew penalty (`R_du` term dropped from `0.8` to `0.01` in `final/runtime_config.py`), so base-y actuation should be watched under payload/asymmetric disturbance testing.
- The structural unit eigenvalue on base-y position is expected (integrator mode), so long-horizon position drift remains a behavior to monitor rather than a bug.

## 4) Nightly-long regression envelope (post-fix)

Scope: baseline-only sweep (`trials=0`) with `benchmark_profile=nightly_long`, 40 episodes x 6000 steps, `current` family, model variants `nominal,inertia_plus,friction_low,com_shift`, domain profiles `default,rand_light,rand_medium`.

Source:
- `benchmark_20260221_140044.csv`
- `benchmark_20260221_140044_summary.txt`
- `benchmark_20260221_140044_protocol.json`

| Mode | Variant | Crash rate (`default / rand_light / rand_medium`) |
|---|---|---|
| `mode_default` | `nominal` | `0.375 / 0.375 / 0.375` |
| `mode_default` | `com_shift` | `0.150 / 0.150 / 0.125` |
| `mode_default` | `friction_low` | `0.000 / 0.000 / 0.000` |
| `mode_default` | `inertia_plus` | `1.000 / 1.000 / 1.000` |
| `mode_low_spin_robust` | `nominal` | `0.375 / 0.375 / 0.375` |
| `mode_low_spin_robust` | `com_shift` | `0.050 / 0.050 / 0.050` |
| `mode_low_spin_robust` | `friction_low` | `0.000 / 0.000 / 0.000` |
| `mode_low_spin_robust` | `inertia_plus` | `1.000 / 1.000 / 1.000` |

Interpretation:
- The fast profile `0%` crash result does not transfer to nightly-long stress conditions.
- `inertia_plus` remains a hard failure mode in both stability profiles.
- `low-spin-robust` improves `com_shift` crash rate (0.15 -> 0.05), but nominal nightly-long crash remains unchanged (0.375), and wheel budget usage is materially higher in robust mode.

## 5) Open-scope boundary checks (`2026-02-21`)

### 5.1 MPC retest after structural model fixes

Fast profile (`benchmark_20260221_141746.csv`, 8 episodes x 2000 steps, nominal/default):
- `mode_default / baseline_mpc`: crash rate `0.000`

Longer run (`benchmark_20260221_142539.csv`, 20 episodes x 6000 steps, nominal/default):
- `mode_default / baseline_mpc`: crash rate `0.000`

Result: the previously recorded historical `0.375` MPC crash snapshot is not present in these current-code retests.

### 5.2 Payload effect in benchmark (current family)

No payload (`benchmark_20260221_144534.csv`, 20 episodes x 6000 steps, nominal/default):
- `mode_default / current`: crash rate `0.000`

Payload `0.8 kg` (`benchmark_20260221_144433.csv`, same protocol):
- `mode_default / current`: crash rate `0.150`

Result: payload sensitivity is now measurable in benchmark artifacts and is non-trivial.

### 5.3 Targeted `R_du_by` sensitivity under payload (headless evaluator sweep)

Source: `open_boundaries_20260221_142334.csv` (robust mode, nominal/default).

Key cases:
- `r_du_by=0.8`, payload `0.8 kg`: crash rate `0.167`
- `r_du_by=0.01`, payload `0.8 kg`: crash rate `0.583`

Result: with this controlled sweep, aggressive `r_du_by=0.01` is substantially less robust under payload than `r_du_by=0.8`.

### 5.4 Long-horizon drift/stability probes

Sources:
- `open_boundaries_long_20260221_142430.csv`
- `open_boundaries_drift_20260221_142520.csv`

Observed in long quiet runs (20k steps):
- Large `worst_base_y_m` (~3.9m) and non-zero crash rates remain possible even with zero external disturbances.

Interpretation:
- The long-horizon behavior boundary is still open and should not be considered closed by fast-profile outcomes.

## 6) Sim-to-real gap proxy table (quantified)

Method:
- Use nominal/default rows as anchor per `(mode, family)`.
- Measure deltas under model/domain stress factors.
- Aggregate using `final/sim2real_sensitivity.py`.

Source:
- `benchmark_20260215_215203.csv`
- Generated from `final/sim2real_sensitivity.py` on 2026-02-21 (command in section 8)

| Factor | Samples | Mean abs Δ score | Mean abs Δ worst tilt (deg) | Mean sensitivity index |
|---|---:|---:|---:|---:|
| `inertia_plus` | 10 | 6.5865 | 0.0660 | 6.6360 |
| `rand_light` | 10 | 2.4938 | 0.0424 | 2.5256 |

Interpretation:
- In this reference run, inertia mismatch drives larger performance drift than light domain randomization.

## 7) Hardware replay/readiness metrics

Source: `readiness_20260215_220928.json` and `benchmark_20260215_220928.csv`.

| Metric | Value | Target / expectation | Status |
|---|---:|---:|---|
| Deterministic rerun Δ score | 0.0 | 0.0 | Pass |
| Deterministic rerun Δ survival | 0.0 | 0.0 | Pass |
| `readiness_sign_sanity_pass` | 0 / 6 rows | 6 / 6 rows | Fail |
| `readiness_replay_pass` | 0 / 6 rows | 6 / 6 rows | Fail |
| `readiness_overall_pass` | 0 / 6 rows | 6 / 6 rows | Fail |
| `sim_real_consistency_mean` | n/a (no hardware trace replay) | >= 0.60 | Not measured |
| `sim_real_traj_nrmse_mean` | n/a (no hardware trace replay) | <= 0.75 | Not measured |

Current gap to close:
- This repo has quantified simulation robustness and sensitivity, but still needs hardware-trace replay metrics to quantify true sim-to-real fidelity.

## 8) Repro commands

```powershell
# Fast benchmark snapshot
python final/benchmark.py --benchmark-profile fast_pr --episodes 8 --steps 2000 --trials 0 --controller-families current,current_dob,hybrid_modern,paper_split_baseline,baseline_mpc,baseline_robust_hinf_like --model-variants nominal --domain-rand-profile default --compare-modes default-vs-low-spin-robust --primary-objective balanced

# Nightly-long baseline sweep used in section 4
python final/benchmark.py --benchmark-profile nightly_long --episodes 40 --steps 6000 --trials 0 --controller-families current --model-variants nominal,inertia_plus,friction_low,com_shift --domain-rand-profile default,rand_light,rand_medium --compare-modes default-vs-low-spin-robust --primary-objective balanced

# MPC retest (fast + longer)
python final/benchmark.py --benchmark-profile fast_pr --episodes 8 --steps 2000 --trials 0 --controller-families baseline_mpc --model-variants nominal --domain-rand-profile default --compare-modes default-vs-low-spin-robust --primary-objective balanced
python final/benchmark.py --benchmark-profile nightly_long --episodes 20 --steps 6000 --trials 0 --controller-families baseline_mpc --model-variants nominal --domain-rand-profile default --compare-modes default-vs-low-spin-robust --primary-objective balanced

# Payload A/B on current family (same protocol, different payload mass)
python final/benchmark.py --benchmark-profile nightly_long --episodes 20 --steps 6000 --trials 0 --controller-families current --model-variants nominal --domain-rand-profile default --compare-modes default-vs-low-spin-robust --primary-objective balanced --payload-mass 0.0
python final/benchmark.py --benchmark-profile nightly_long --episodes 20 --steps 6000 --trials 0 --controller-families current --model-variants nominal --domain-rand-profile default --compare-modes default-vs-low-spin-robust --primary-objective balanced --payload-mass 0.8

# Sensitivity summary from a benchmark CSV that includes variant/domain factors
python final/sim2real_sensitivity.py --csv final/results/benchmark_20260215_215203.csv --out-prefix final/results/sim2real_sensitivity_20260215_215203_ref

# Readiness report run (requires hardware trace input to fully measure replay metrics)
python final/benchmark.py --release-campaign --episodes 1 --trials 0 --steps 120 --controller-families current,hybrid_modern,paper_split_baseline --model-variants nominal --domain-rand-profile default --compare-modes default-vs-low-spin-robust --primary-objective balanced --readiness-report --hardware-trace-path path/to/hardware_trace.csv
```
