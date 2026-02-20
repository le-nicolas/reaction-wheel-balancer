from pathlib import Path
import mujoco
import mujoco.viewer
import time
import numpy as np

# ----------------------------
# Load MuJoCo model and data
# ----------------------------
model = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[1] / "models" / "wheel_on_stick.xml"))
data = mujoco.MjData(model)

dt = model.opt.timestep  # simulation timestep

# ----------------------------
# PD controller gains
# ----------------------------
Kp = 300.0   # proportional gain (angle)
Kd = 30.0    # derivative gain (angular velocity)

# ----------------------------
# Wheel / torque limits
# ----------------------------
max_tau = 1000.0  # max actuator torque
max_wheel_speed = 100.0  # optional safety

# ----------------------------
# Initial conditions (small tilt)
# ----------------------------
data.qpos[0] = 0.01  # ~0.5° tilt from upright
data.qvel[0] = 0.0   # stick angular velocity
data.qvel[1] = 0.0   # wheel angular velocity

# ----------------------------
# Simulation loop
# ----------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # --- Read stick state ---
        theta = data.qpos[0]       # stick angle (radians)
        theta_dot = data.qvel[0]   # stick angular velocity

        # --- PD control for torque ---
        # Torque automatically changes sign depending on stick tilt
        tau = - (Kp * theta + Kd * theta_dot)

        # --- Saturate torque ---
        tau = np.clip(tau, -max_tau, max_tau)

        # --- Optional: reduce torque if wheel speed near max ---
        wheel_speed = data.qvel[1]
        if abs(wheel_speed) > 0.9 * max_wheel_speed:
            tau *= 0.5

        # --- Apply torque to wheel ---
        data.ctrl[0] = tau

        # --- Step simulation ---
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)

