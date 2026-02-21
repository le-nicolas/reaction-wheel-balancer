#!/usr/bin/env python3
"""
Diagnose roll dynamics changes after root weld modification.
Focus on A[roll_rate, roll] and B[roll_rate, base_y] which changed with the weld.
"""

import numpy as np
import mujoco
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "final"))

from runtime_model import lookup_model_ids, reset_state
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

def main():
    xml_path = Path(__file__).parent / "final" / "final.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    args = parse_args(["--mode", "robust"])
    cfg = build_config(args)
    
    ids = lookup_model_ids(model)
    reset_state(model, data, ids.q_pitch, ids.q_roll, pitch_eq=0.0, roll_eq=0.0)
    
    control_steps = 1 if not cfg.hardware_realistic else max(1, int(round(1.0 / (model.opt.timestep * cfg.control_hz))))
    
    nx = 2 * model.nv + model.na
    nu = model.nu
    A_full = np.zeros((nx, nx))
    B_full = np.zeros((nx, nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A_full, B_full, None, None)
    
    # Extract state indices from linearization
    idx = [
        ids.v_pitch,              # 0
        ids.v_roll,               # 1
        model.nv + ids.v_pitch,   # 2 (pitch_rate)
        model.nv + ids.v_roll,    # 3 (roll_rate) 
        model.nv + ids.v_rw,      # 4
        ids.v_base_x,             # 5
        ids.v_base_y,             # 6
        model.nv + ids.v_base_x,  # 7 (base_x_rate)
        model.nv + ids.v_base_y,  # 8 (base_y_rate)
    ]
    A_step = A_full[np.ix_(idx, idx)]
    B_step = B_full[np.ix_(idx, [ids.aid_rw, ids.aid_base_x, ids.aid_base_y])]
    A, B = lift_discrete_dynamics(A_step, B_step, control_steps)
    
    state_names = ["pitch", "roll", "pitch_rate", "roll_rate", "wheel_rate", "base_x", "base_y", "base_x_rate", "base_y_rate"]
    control_names = ["wheel_spin", "base_x_force", "base_y_force"]
    
    print("\n=== ROLL DYNAMICS ANALYSIS ===\n")
    
    print("State ordering in A/B matrices:")
    for i, name in enumerate(state_names):
        print(f"  [{i}] {name}")
    print()
    
    # Key indices in the reordered system
    roll_idx = 1
    roll_rate_idx = 3
    base_y_idx = 6
    base_y_rate_idx = 8
    
    print("Roll subsystem structure:")
    print(f"  A[roll_rate={roll_rate_idx}, roll={roll_idx}] = {A[roll_rate_idx, roll_idx]:.8f}")
    print(f"    ^ How roll position drives roll rate (should be ~0 for free joint)")
    print()
    print(f"  A[roll_rate={roll_rate_idx}, roll_rate={roll_rate_idx}] = {A[roll_rate_idx, roll_rate_idx]:.8f}")
    print(f"    ^ Roll rate damping")
    print()
    
    print("Base_Y coupling to roll rate:")
    print(f"  A[roll_rate={roll_rate_idx}, base_y={base_y_idx}] = {A[roll_rate_idx, base_y_idx]:.8f}")
    print(f"    ^ How base_y position affects roll rate (should show passive coupling if base moves to resist roll)")
    print()
    print(f"  A[roll_rate={roll_rate_idx}, base_y_rate={base_y_rate_idx}] = {A[roll_rate_idx, base_y_rate_idx]:.8f}")
    print(f"    ^ How base_y velocity affects roll rate")
    print()
    
    print("Control input effect on roll rate:")
    for j, u_name in enumerate(control_names):
        print(f"  B[roll_rate={roll_rate_idx}, {u_name}] = {B[roll_rate_idx, j]:.8f}")
    print()
    
    print("Roll position dynamics:")
    print(f"  A[roll={roll_idx}, roll_rate={roll_rate_idx}] = {A[roll_idx, roll_rate_idx]:.8f}")
    print(f"    ^ Integration of roll rate (should be ~0.004 for 250Hz control)")
    print()
    
    print("Full roll-related rows of A matrix:")
    print(f"  A[roll={roll_idx}, :] = {A[roll_idx, :]}")
    print(f"  A[roll_rate={roll_rate_idx}, :] = {A[roll_rate_idx, :]}")
    print()
    
    print("Full roll-related rows of B matrix:")
    print(f"  B[roll={roll_idx}, :] = {B[roll_idx, :]}")
    print(f"  B[roll_rate={roll_rate_idx}, :] = {B[roll_rate_idx, :]}")
    print()
    
    # Analyze roll eigenvalue
    print("Open-loop dynamics eigenvalues:")
    eigs = np.linalg.eigvals(A)
    for i, e in enumerate(sorted(np.abs(eigs), reverse=True)):
        print(f"  {i+1}: {e:.6f}")

if __name__ == "__main__":
    main()
