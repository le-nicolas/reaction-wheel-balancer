import mujoco
import mujoco.viewer
import time
import numpy as np

# ----------------------------
# Load MuJoCo model and data
# ----------------------------
model = mujoco.MjModel.from_xml_path("wheel_on_stick.xml")
data = mujoco.MjData(model)

dt = model.opt.timestep  # simulation timestep

# ----------------------------
# LQR-style state-feedback gains
# ----------------------------
# These gains were tuned for your wheel + stick system
# u = -K @ x
# x = [theta, theta_dot, wheel_speed]
K = np.array([1200.0, 100.0, 10.0])  # [angle, angular velocity, wheel speed]

# ----------------------------
# Reaction wheel limits
# ----------------------------
max_tau = 1000           # max torque (matches actuator ctrlrange)
max_wheel_speed = 100    # max wheel angular velocity (rad/s)

# ----------------------------
# Initial conditions
# ----------------------------
data.qpos[0] = 0     # small initial tilt
data.qvel[0] = 0         # stick starts at rest
data.qvel[1] = 0         # wheel starts at rest

# ----------------------------
# Simulation loop
# ----------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # --- Read current state ---
        theta = data.qpos[0]
        theta_dot = data.qvel[0]
        wheel_speed = data.qvel[1]

        # --- LQR state vector ---
        x = np.array([theta, theta_dot, wheel_speed])

        # --- Gravity compensation ---
        m = 4.0        # total mass above hinge (kg)
        g = 9.81
        l = 1.75       # COM distance from hinge (m)
        tau_g = m * g * l * np.sin(theta)

        # --- State-feedback control + gravity compensation ---
        tau = -np.dot(K, x) - tau_g

        # --- Saturate torque ---
        tau = np.clip(tau, -max_tau, max_tau)

        # --- Apply torque ---
        data.ctrl[0] = tau


        # --- Step simulation ---
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)
