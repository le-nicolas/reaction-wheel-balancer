# Sim-to-Real Workflow

This runbook defines what to do after exporting firmware parameters, including how to measure and track simulation-vs-hardware mismatch.

## 1) Export and Parity Check

Generate firmware parameters from the same controller profile you intend to run:

```powershell
python final/export_firmware_params.py --mode robust --real-hardware --out final/firmware/controller_params.h
```

Run parity tests before flashing:

```powershell
python -m pytest final/test_export_parity.py -q
python -m pytest final/test_firmware_scenario_parity.py -q
```

## 2) Firmware Bring-Up (ESP32)

Recommended bring-up order:

1. Flash with conservative limits (`--real-hardware` export).
2. Validate sensor freshness and watchdog behavior with motor outputs disabled.
3. Enable wheel actuation only.
4. Enable base actuation only after wheel-only checks are stable.
5. Enable full controller and safety latches.

The generated header (`final/firmware/controller_params.h`) is the contract between sim/runtime and firmware control constants.

## 3) Capture a Hardware Trace

Create a CSV with these required columns (same sample order and rate):

- `pitch_rate`
- `roll_rate`
- `base_vx`
- `base_vy`

Optional but recommended metadata columns:

- `timestamp_us`, `mode`, `fault_state`, `battery_v`

## 4) Replay Consistency Check

Run benchmark with hardware replay enabled:

```powershell
python final/benchmark.py --benchmark-profile fast_pr --episodes 8 --steps 3000 --trials 0 --controller-families current --model-variants nominal,inertia_plus,inertia_minus,friction_low,friction_high,com_shift --domain-rand-profile default,rand_light,rand_medium,rand_heavy --compare-modes default-vs-low-spin-robust --primary-objective balanced --hardware-trace-path path/to/hardware_trace.csv
```

Focus on these replay outputs in the resulting benchmark CSV:

- `sim_real_consistency_mean` (higher is better)
- `sim_real_traj_nrmse_mean` (lower is better)

## 5) Sensitivity Sweep

Use the benchmark CSV to rank which modeled factors move performance most:

```powershell
python final/sim2real_sensitivity.py --csv final/results/benchmark_<timestamp>.csv
```

The script writes:

- `final/results/sim2real_sensitivity_<timestamp>_detail.csv`
- `final/results/sim2real_sensitivity_<timestamp>_summary.csv`
- `final/results/sim2real_sensitivity_<timestamp>_summary.md`

Interpretation:

- `model_variant:*` factors isolate plant mismatches (inertia, friction, COM shift).
- `domain_profile:*` factors isolate sensing/timing/noise mismatch pressure.
- Higher `sensitivity_index` means larger deviation from nominal baseline.

## 6) Discrepancy Logging Discipline

For every hardware session, fill in:

- `docs/SIM_TO_REAL_DISCREPANCY_LOG.md`

Minimum acceptance gates before promoting tuning changes:

1. Parity tests pass (`final/test_export_parity.py`, `final/test_firmware_scenario_parity.py`).
2. Replay metrics satisfy your thresholds.
3. No new safety latch regressions.
4. Sensitivity summary does not show uncontrolled degradation on high-risk factors.
