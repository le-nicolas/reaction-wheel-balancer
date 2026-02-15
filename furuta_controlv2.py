import mujoco
import mujoco.viewer
import numpy as np
import time

# ------------------------------
# Load model
# ------------------------------
model = mujoco.MjModel.from_xml_path("furuta_pendulum.xml")
data = mujoco.MjData(model)

# Joint indices
ARM_JOINT = model.joint("arm_joint").qposadr
PEND_JOINT = model.joint("pendulum_joint").qposadr

# Control parameters
torque_input = 0.0
torque_step = 2.0  # max torque applied per frame
paused = False

# ------------------------------
# Reset helper
# ------------------------------
def reset_to_key(key_name):
    key_id = model.key(key_name).id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

# ------------------------------
# Main loop using keyboard polling
# ------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    reset_to_key("down")

    while viewer.is_running():
        # ----------------------
        # Read keyboard state
        # ----------------------
        keys = viewer.get_keyboard()  # returns dict of key -> True if pressed

        # Default torque = 0
        torque_input = 0.0

        if keys.get(mujoco.mjtKey.mjKEY_LEFT, False):
            torque_input = -torque_step
        if keys.get(mujoco.mjtKey.mjKEY_RIGHT, False):
            torque_input = torque_step
        if keys.get(ord('R'), False):
            reset_to_key("down")
        if keys.get(ord('U'), False):
            reset_to_key("upright")
        if keys.get(ord(' '), False):
            paused = not paused
            time.sleep(0.2)  # debounce

        if keys.get(mujoco.mjtKey.mjKEY_ESCAPE, False):
            break

        # ----------------------
        # Step simulation
        # ----------------------
        if not paused:
            data.ctrl[0] = torque_input
            mujoco.mj_step(model, data)

        viewer.sync()
        time.sleep(model.opt.timestep)
