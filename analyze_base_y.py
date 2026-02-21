#!/usr/bin/env python3
"""
Analyze base_y controllability and system structure.
"""

import numpy as np
import mujoco
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "final"))

from runtime_model import lookup_model_ids, reset_state
from runtime_config import build_config, parse_args

def _controllability_rank(A: np.ndarray, B: np.ndarray) -> int:
    n = A.shape[0]
    ctrb = B.copy()
    Ak = np.eye(n, dtype=float)
    for _ in range(1, n):
        Ak = Ak @ A
        ctrb = np.hstack([ctrb, Ak @ B])
    return int(np.linalg.matrix_rank(ctrb))

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

def main():
    # Load model
    xml_path = Path(__file__).parent / "final" / "final.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    # Build config
    args = parse_args(["--mode", "robust"])
    cfg = build_config(args)
    
    ids = lookup_model_ids(model)
    reset_state(model, data, ids.q_pitch, ids.q_roll, pitch_eq=0.0, roll_eq=0.0)
    
    # Linearize
    control_steps = 1 if not cfg.hardware_realistic else max(1, int(round(1.0 / (model.opt.timestep * cfg.control_hz))))
    
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
    
    state_names = ["pitch", "roll", "pitch_rate", "roll_rate", "wheel_rate", "base_x", "base_y", "base_x_rate", "base_y_rate"]
    control_names = ["wheel_spin", "base_x_force", "base_y_force"]
    
    print("\n=== SYSTEM STRUCTURE ANALYSIS ===\n")
    
    print("Full system controllability:")
    rank = _controllability_rank(A, B)
    print(f"  Combined rank: {rank}/{A.shape[0]} (full rank = all modes controllable)")
    
    print("\nPer-input controllability (one input at a time):")
    for j, u_name in enumerate(control_names):
        B_single = B[:, [j]]
        rank_single = _controllability_rank(A, B_single)
        print(f"  {u_name:20s} (B col {j}): rank {rank_single}/9")
    
    print("\nBase_Y-specific controllability:")
    # base_y is at index 6, base_y_rate is at index 8
    # Extract subsystem of base_y dynamics
    base_y_idx = 6
    base_y_rate_idx = 8
    
    # Look at which inputs affect base_y directly
    print(f"\nB matrix (control influence on states):")
    for j, u_name in enumerate(control_names):
        b_y = B[base_y_idx, j]
        b_vy = B[base_y_rate_idx, j]
        print(f"  {u_name:20s}: base_y_effect={b_y:10.6f}, base_y_rate_effect={b_vy:10.6f}")
    
    print(f"\nA matrix (state dynamics):")
    print(f"  A[base_y, base_y] = {A[base_y_idx, base_y_idx]:.6f}  (self-dynamics)")
    print(f"  A[base_y, base_y_rate] = {A[base_y_idx, base_y_rate_idx]:.6f}  (velocity coupling)")
    print(f"  A[base_y_rate, base_y] = {A[base_y_rate_idx, base_y_idx]:.6f}")
    print(f"  A[base_y_rate, base_y_rate] = {A[base_y_rate_idx, base_y_rate_idx]:.6f}  (rate damping)")
    
    # Analyze base_y as 2D subsystem
    A_by = A[np.ix_([base_y_idx, base_y_rate_idx], [base_y_idx, base_y_rate_idx])]
    B_by = B[np.ix_([base_y_idx, base_y_rate_idx], [2])]  # Only base_y_force
    
    print(f"\nBase_Y subsystem (when only considering base_y inputs):")
    print(f"  A_by =\n{A_by}")
    print(f"  B_by =\n{B_by}")
    
    rank_by = _controllability_rank(A_by, B_by)
    print(f"  Controllability: rank={rank_by}/2")
    
    # Eigenvalues of base_y subsystem
    eigs_by = np.linalg.eigvals(A_by)
    print(f"  Eigenvalues (magnitude): {[f'{abs(e):.6f}' for e in eigs_by]}")

if __name__ == "__main__":
    main()
