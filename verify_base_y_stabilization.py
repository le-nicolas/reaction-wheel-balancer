#!/usr/bin/env python3
"""
Verify that base_y eigenvalue is stabilized after Q tuning.
"""

import numpy as np
import mujoco
from pathlib import Path
from scipy.linalg import solve_discrete_are

# Add final to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent / "final"))

from runtime_model import lookup_model_ids, reset_state, enforce_planar_root_attitude
from runtime_config import build_config, parse_args

def lift_discrete_dynamics(A: np.ndarray, B: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order-hold lifting."""
    n = max(int(steps), 1)
    if n == 1:
        return A.copy(), B.copy()
    A_lift = np.linalg.matrix_power(A, n)
    acc = np.zeros_like(A)
    Ak = np.eye(A.shape[0], dtype=float)
    for _ in range(n):
        acc += Ak
        Ak = A @ Ak
    B_lift = acc @ B
    return A_lift, B_lift

def _solve_discrete_are_robust(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, label: str) -> np.ndarray:
    """Solve DARE with regularization fallback."""
    reg_steps = (0.0, 1e-12, 1e-10, 1e-8, 1e-6)
    eye = np.eye(R.shape[0], dtype=float)
    last_exc: Exception | None = None
    for eps in reg_steps:
        try:
            P = solve_discrete_are(A, B, Q, R + eps * eye)
            if np.all(np.isfinite(P)):
                return 0.5 * (P + P.T)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"{label} DARE failed after regularization fallback: {last_exc}") from last_exc

def _solve_linear_robust(gram: np.ndarray, rhs: np.ndarray, label: str) -> np.ndarray:
    """Solve linear system with fallback to lstsq."""
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
        if not np.all(np.isfinite(sol)):
            raise RuntimeError(f"{label} linear solve returned non-finite result.")
        return sol

def main():
    # Load model
    xml_path = Path(__file__).parent / "final" / "final.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    # Build config with default settings (non-smooth mode = robust)
    args = parse_args(["--mode", "robust"])  # Explicitly use robust mode
    cfg = build_config(args)
    
    ids = lookup_model_ids(model)
    
    # Reset to upright
    reset_state(model, data, ids.q_pitch, ids.q_roll, pitch_eq=0.0, roll_eq=0.0)
    
    # Linearize
    control_steps = 1 if not cfg.hardware_realistic else max(1, int(round(1.0 / (model.opt.timestep * cfg.control_hz))))
    control_dt = control_steps * model.opt.timestep
    
    nx = 2 * model.nv + model.na
    nu = model.nu
    A_full = np.zeros((nx, nx))
    B_full = np.zeros((nx, nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A_full, B_full, None, None)
    
    idx = [
        ids.v_pitch, ids.v_roll, model.nv + ids.v_pitch, model.nv + ids.v_roll,
        model.nv + ids.v_rw, ids.v_base_x, ids.v_base_y, model.nv + ids.v_base_x, model.nv + ids.v_base_y,
    ]
    A_step = A_full[np.ix_(idx, idx)]
    B_step = B_full[np.ix_(idx, [ids.aid_rw, ids.aid_base_x, ids.aid_base_y])]
    A, B = lift_discrete_dynamics(A_step, B_step, control_steps)
    NX = A.shape[0]
    NU = B.shape[1]
    
    # Build augmented system
    A_aug = np.block([[A, B], [np.zeros((NU, NX)), np.eye(NU)]])
    B_aug = np.vstack([B, np.eye(NU)])
    Q_aug = np.block([[cfg.qx, np.zeros((NX, NU))], [np.zeros((NU, NX)), cfg.qu]])
    
    # Solve DARE
    P_aug = _solve_discrete_are_robust(A_aug, B_aug, Q_aug, cfg.r_du, label="Controller")
    K_du = _solve_linear_robust(B_aug.T @ P_aug @ B_aug + cfg.r_du, B_aug.T @ P_aug @ A_aug, label="Controller gain")
    
    # Compute closed-loop eigenvalues
    A_closed = A_aug - B_aug @ K_du
    eigs = np.linalg.eigvals(A_closed)
    eigs_sorted = np.sort(np.abs(eigs))[::-1]
    
    print("\n=== BASE_Y STABILIZATION VERIFICATION ===\n")
    print(f"Open-loop A eigenvalues (magnitude):")
    for i, ev in enumerate(np.sort(np.abs(np.linalg.eigvals(A)))[::-1]):
        mode_name = ["base_y (2)", "base_x (2)", "wheel_rate", "pitch_rate", "roll_rate", "pitch", "roll"][i] if i < 7 else "extra"
        print(f"  {i+1}: {ev:.6f}  ({mode_name})")
    
    print(f"\nClosed-loop eigenvalues (magnitude, top 9/12):")
    for i, ev in enumerate(eigs_sorted[:9]):
        print(f"  {i+1}: {ev:.6f}")
    
    all_stable = np.all(eigs_sorted < 1.0)
    print(f"\n[PASS] All closed-loop poles inside unit circle: {all_stable}")
    max_eig = np.max(eigs_sorted)
    print(f"  Largest magnitude: {max_eig:.6f}")
    
    if max_eig >= 1.0:
        print(f"  [WARN] Unstable eigenvalue(s) detected at magnitude {max_eig:.6f}")
        unstable_eigs = eigs[np.abs(eigs) >= 1.0]
        print(f"    Unstable poles: {sorted(np.abs(unstable_eigs), reverse=True)}")
    else:
        print(f"  [OK] Base-y mode is stable (max eig margin: {1.0 - max_eig:.6f})")
    
    print(f"\nQ_x[base_y, base_y] (state 6): {cfg.qx[6,6]}")
    print(f"Q_x[base_y_rate, base_y_rate] (state 8): {cfg.qx[8,8]}")
    print(f"R_du[base_y_force, base_y_force] (input 2): {cfg.r_du[2,2]}")

if __name__ == "__main__":
    main()
