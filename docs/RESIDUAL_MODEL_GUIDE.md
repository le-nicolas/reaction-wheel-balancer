# Residual Model Guide

The residual policy is an additive correction:

`u_total = u_nominal + delta_u`

where `delta_u` is produced by a small neural network and safety-clamped before application.

## 1) Runtime Semantics

The loader in `final/residual_model.py` expects:

- `input_dim = 15`
- `output_dim = 3`
- MLP with `hidden_dim` and `hidden_layers` metadata
- checkpoint fields:
  - `state_dict` (or `model_state_dict`)
  - `input_mean`, `input_std`
  - `output_scale`

Feature order (15 total):

1. `x_est[0:9]`:
   - pitch, roll
   - pitch_rate, roll_rate, wheel_rate
   - base_x, base_y, base_vx, base_vy
2. `u_eff_applied[0:3]`: effective applied command from previous step
3. `u_nominal[0:3]`: nominal controller output before residual

Output order (3 total):

- `delta_u = [delta_rw, delta_bx, delta_by]`

## 2) Safety and Gating

Runtime checks in `ResidualPolicy.step(...)`:

1. Disable residual if tilt/rate exceed gate thresholds.
2. Disable residual on non-finite features.
3. Clip output per-channel using `--residual-max-rw`, `--residual-max-bx`, `--residual-max-by`.

Default gate thresholds:

- `--residual-gate-tilt-deg 8.0`
- `--residual-gate-rate 3.0`

## 3) Physical Meaning of `--residual-scale`

Runtime computes:

`delta_u = raw_network_output * output_scale * residual_scale`

then applies channel clipping.

So `--residual-scale 0.20` means:

- allow 20% of the calibrated residual amplitude (`output_scale`), before final max-abs clipping
- keep the residual in a low-authority assist role relative to the nominal controller

## 4) Data Collection

Use paired trace captures from the same seed/disturbance setup:

1. Baseline run (nominal controller):

```powershell
python final/final.py --mode robust --controller-family current --seed 12345 --trace-events-csv final/results/trace_baseline.csv
```

2. Teacher run (stronger reference policy, for example `hybrid_modern`):

```powershell
python final/final.py --mode robust --controller-family hybrid_modern --seed 12345 --trace-events-csv final/results/trace_teacher.csv
```

3. Build training dataset:

```powershell
python final/build_residual_dataset.py --baseline-trace final/results/trace_baseline.csv --teacher-trace final/results/trace_teacher.csv --out final/results/residual_dataset.npz
```

Dataset builder aligns by `step`, estimates base velocities from trace position/time, and sets:

- features from baseline trajectory
- target as `teacher_u - baseline_u`

## 5) Train

```powershell
python final/train_residual_model.py --dataset final/results/residual_dataset.npz --out final/results/residual.pt --epochs 50 --batch-size 1024 --hidden-dim 32 --hidden-layers 2
```

The output checkpoint is directly loadable by `--residual-model`.

## 6) Evaluate and Deploy

Dry-run in sim:

```powershell
python final/final.py --mode robust --residual-model final/results/residual.pt --residual-scale 0.20 --log-control-terms --trace-events-csv final/results/trace_residual_eval.csv
```

Check runtime summary:

- residual applied rate
- residual clipped count
- residual gate-blocked count
- residual max absolute per channel

If clipping is high, either reduce `--residual-scale` or retrain with tighter target bounds.
