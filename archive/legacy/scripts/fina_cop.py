from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import solve_discrete_are

# ============================================================
# Load model
# ============================================================
model = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[3] / "final" / "final.xml"))
data  = mujoco.MjData(model)
dt    = model.opt.timestep

# ============================================================
# Resolve joints
# ============================================================
def jid(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

jid_stick = jid("stick_hinge")
jid_rw    = jid("wheel_spin")
jid_base  = jid("base_wheel")

q_stick = model.jnt_qposadr[jid_stick]
v_stick = model.jnt_dofadr[jid_stick]
v_rw    = model.jnt_dofadr[jid_rw]
v_base  = model.jnt_dofadr[jid_base]

# ============================================================
# Resolve actuators
# ============================================================
def aid(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

aid_rw    = aid("wheel_spin")
aid_base  = aid("base_wheel")
aid_stick = aid("stick_hinge")

# ============================================================
# Upright equilibrium angle
# ============================================================
THETA_EQ = 0.0  # θ=0 is wheel pointing up (inverted equilibrium)

# ============================================================
# Linearization about upright
# ============================================================
def linearize():
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    data.qpos[q_stick] = THETA_EQ
    mujoco.mj_forward(model, data)

    nx = model.nq + model.nv
    nu = model.nu

    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))

    mujoco.mjd_transitionFD(
        model, data, 1e-6, True, A, B, None, None
    )

    idx = [
        q_stick,
        model.nq + v_stick,
        model.nq + v_rw,
        model.nq + v_base
    ]

    A_r = A[np.ix_(idx, idx)]
    B_r = B[np.ix_(idx, [aid_rw, aid_base])]

    return A_r, B_r

A, B = linearize()

print("\n=== LINEARIZATION ===")
print(f"A = \n{A}")
print(f"B = \n{B}")
print(f"A eigenvalues: {np.linalg.eigvals(A)}")

# ============================================================
# Discrete LQR
# ============================================================
# Q: penalize state deviations, R: penalize control effort
# States: [θ, θ_dot, ω_rw, v_base]
Q = np.diag([80.0, 30.0, 1.0, 800.0])    # Focus on minimizing base velocity
R = np.diag([1.0, 0.001])                # Cheap base force, expensive reaction wheel

P = solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)

print("\n=== LQR GAINS ===")
print(f"K = \n{K}")
print(f"Max reaction wheel gain: {np.max(np.abs(K[:, 0:2]))}")

# ============================================================
# Kalman Filter
# ============================================================
Qn = np.diag([1e-4, 1e-4, 1e-5, 1e-5])
Rn = np.diag([5e-6, 1e-4, 3e-4, 3e-4])
C  = np.diag([1.0, 0.5, 0.3, 0.3])

Pk = solve_discrete_are(A.T, C.T, Qn, Rn)
L  = Pk @ C.T @ np.linalg.inv(C @ Pk @ C.T + Rn)

# ============================================================
# Initial conditions
# ============================================================
# Start with stick perfectly upright
data.qpos[q_stick] = 0.0  # Upright
data.qvel[:] = 0.0
data.ctrl[:] = 0.0
mujoco.mj_forward(model, data)

# Initialize state estimate
x_est = np.array([0.0, 0.0, 0.0, 0.0])

# ============================================================
# Integral action (base wheel only)
# ============================================================
theta_int = 0.0
KI_BASE   = 6.0

# ============================================================
# Torque & Force limits
# ============================================================
MAX_TAU_RW    = 500.0    # Reaction wheel torque limit
MAX_F_BASE    = 100.0    # Base cart force limit

# ============================================================
# Sensor noise
# ============================================================
rng   = np.random.default_rng()
SIGMA = np.array([0.002, 0.01, 0.02, 0.02])

# ============================================================
# Disturbance parameters
# ============================================================
DISTURBANCE_MAGNITUDE = 20.0  # Strong disturbances to force base motion
DISTURBANCE_INTERVAL  = 100

# ============================================================
# Simulation loop
# ============================================================
with mujoco.viewer.launch_passive(model, data) as viewer:

    u_prev = np.zeros(2)
    step_count = 0
    max_angle = 0.0

    while viewer.is_running():
        step_count += 1

        # ----------------------------------------------------
        # True state (ANGLE SHIFTED)
        # ----------------------------------------------------
        x_true = np.array([
            data.qpos[q_stick] - THETA_EQ,
            data.qvel[v_stick],
            data.qvel[v_rw],
            data.qvel[v_base]
        ])

        # ----------------------------------------------------
        # Measurement
        # ----------------------------------------------------
        y = x_true + rng.normal(0.0, SIGMA)

        # ----------------------------------------------------
        # Kalman filter
        # ----------------------------------------------------
        x_pred = A @ x_est + B @ u_prev
        y_pred = C @ x_pred
        x_est  = x_pred + L @ (y - y_pred)

        # ----------------------------------------------------
        # Integral action (anti-windup)
        # ========== SWING-UP vs LQR ==========
        # LQR control both wheels continuously
        u_lqr = -K @ x_est
        u = np.array([
            u_lqr[0],   # Reaction wheel from LQR
            u_lqr[1]    # BASE WHEEL ENABLED - controls platform position
        ])

        # ----------------------------------------------------
        # Saturation
        # ----------------------------------------------------
        u[0] = np.clip(u[0], -MAX_TAU_RW,   MAX_TAU_RW)
        u[1] = np.clip(u[1], -MAX_F_BASE,   MAX_F_BASE)

        # ----------------------------------------------------
        # Apply control
        data.ctrl[:] = 0.0
        data.ctrl[aid_rw]    = u[0]
        data.ctrl[aid_base]  = u[1]
        data.ctrl[aid_stick] = 0.0

        # Apply random disturbance to stick (push it sideways)
        if step_count % DISTURBANCE_INTERVAL == 0:
            stick_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stick")
            # Random force in X-Z plane (horizontal and vertical perturbations)
            force = np.array([
                rng.uniform(-DISTURBANCE_MAGNITUDE, DISTURBANCE_MAGNITUDE),
                0.0,
                rng.uniform(-DISTURBANCE_MAGNITUDE * 0.5, DISTURBANCE_MAGNITUDE * 0.5)
            ])
            data.xfrc_applied[stick_body_id, :3] = force
        else:
            # Clear disturbance
            stick_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stick")
            data.xfrc_applied[stick_body_id, :3] = 0.0
        
        mujoco.mj_step(model, data)
        viewer.sync()

        u_prev = u.copy()
        
        # Diagnostics
        stick_angle = data.qpos[q_stick]
        max_angle = max(max_angle, abs(stick_angle))
        
        if step_count % 100 == 0:
            print(f"Step {step_count}: angle={np.degrees(stick_angle):6.2f}deg  v_stick={data.qvel[v_stick]:7.3f}  "
                  f"u_rw={u[0]:8.1f}  u_base={u[1]:7.2f}")
        
        # Detect crash (stick fell over)
        if abs(stick_angle) > np.pi * 0.5:
            print(f"\nCRASH at step {step_count}: stick fell over (angle={np.degrees(stick_angle):.1f}deg)")
            break
    
    print(f"\n=== SIMULATION ENDED ===")
    print(f"Total steps: {step_count}")
    print(f"Max stick angle: {np.degrees(max_angle):.2f}deg")

