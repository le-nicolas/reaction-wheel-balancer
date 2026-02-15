import mujoco
import mujoco.viewer
import time
import numpy as np

# ----------------------------
# Load MuJoCo model and data
# ----------------------------
model = mujoco.MjModel.from_xml_path("wheel_on_stick.xml")
data = mujoco.MjData(model)

# ----------------------------
# PID controller gains
# ----------------------------
Kp = 300       # proportional gain
Kd = 30        # derivative gain
Ki = 0         # integral gain (set to 0 for now)

theta_integral = 0.0       # integral accumulator
dt = model.opt.timestep     # simulation timestep

# ----------------------------
# Reaction wheel limits
# ----------------------------
max_tau = 1000           # max torque (matches actuator ctrlrange)
max_wheel_speed = 100    # max wheel angular velocity (rad/s)

# ----------------------------
# Initial conditions
# ----------------------------
data.qpos[0] = 0.05      # small tilt for stick
data.qvel[0] = 0        # stick starts at rest
data.qvel[1] = 0        # wheel starts at rest

# ----------------------------
# Simulation loop
# ----------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # --- Read current stick state ---
        theta = data.qpos[0]       # stick angle
        theta_dot = data.qvel[0]   # stick angular velocity
        wheel_speed = data.qvel[1] # wheel angular velocity

        # --- PID control ---
        theta_integral += theta * dt
        tau = Kp*theta + Kd*theta_dot + Ki*theta_integral

        # --- Torque saturation ---
        tau = np.clip(tau, -max_tau, max_tau)

        # --- Optional: reduce torque if wheel is near max speed ---
        if abs(wheel_speed) > 0.9 * max_wheel_speed:
            tau *= 0.5

        # --- Apply torque to the wheel ---
        data.ctrl[0] = -tau   # negative to oppose stick tilt

        # --- Step simulation ---
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)
