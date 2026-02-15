import mujoco
import mujoco.viewer
import numpy as np
import time

# ----------------------------
# Load model
# ----------------------------
model = mujoco.MjModel.from_xml_path("reactionwheel_basemotor.xml")
data = mujoco.MjData(model)

dt = model.opt.timestep

# ----------------------------
# Control gains
# ----------------------------
# Reaction wheel (fast inner loop)
K_rw = np.array([800.0, 80.0, 20.0])   # [theta, theta_dot, wheel_speed]

# Base motor (slow outer loop)
K_m = 50.0                             # momentum dumping gain

# ----------------------------
# Physical parameters (approx)
# ----------------------------
m = 1.0       # stick mass
g = 9.81
l = 1.0       # COM height from hinge (half of stick length)

# ----------------------------
# Actuator limits (from XML)
# ----------------------------
MAX_TAU_WHEEL = 1000.0
MAX_TAU_BASE  = 200.0

# ----------------------------
# Initial conditions
# ----------------------------
data.qpos[0] = 2    # small initial tilt (rad)
data.qvel[0] = 0.0     # stick angular velocity
data.qvel[1] = 0.0     # wheel angular velocity

# ----------------------------
# Simulation loop
# ----------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # --- State ---
        theta       = data.qpos[0]    # stick angle
        theta_dot   = data.qvel[0]    # stick angular velocity
        wheel_speed = data.qvel[1]    # wheel angular velocity

        # ----------------------------
        # Reaction wheel control
        # ----------------------------
        x = np.array([theta, theta_dot, wheel_speed])
        tau_wheel = -K_rw @ x

        # ----------------------------
        # Base motor control
        # ----------------------------
        # Gravity compensation
        tau_grav = m * g * l * np.sin(theta)

        # Momentum dumping (drive wheel speed to zero)
        tau_base = tau_grav + K_m * wheel_speed

        # ----------------------------
        # Saturation
        # ----------------------------
        tau_wheel = np.clip(tau_wheel, -MAX_TAU_WHEEL, MAX_TAU_WHEEL)
        tau_base  = np.clip(tau_base,  -MAX_TAU_BASE,  MAX_TAU_BASE)

        # ----------------------------
        # Apply controls
        # ----------------------------
        data.ctrl[0] = tau_wheel   # wheel_spin
        data.ctrl[1] = tau_base    # stick_hinge

        # ----------------------------
        # Step simulation
        # ----------------------------
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)
