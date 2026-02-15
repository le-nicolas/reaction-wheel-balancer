import mujoco
import mujoco.viewer
import numpy as np
import time

# ------------------------------
# Load model
# ------------------------------
model = mujoco.MjModel.from_xml_path("furuta_pendulum2.xml")
data = mujoco.MjData(model)

# Joint indices
ARM_JOINT = model.joint("arm_joint").qposadr
PEND_JOINT = model.joint("pendulum_joint").qposadr

ARM_VEL = model.joint("arm_joint").dofadr
PEND_VEL = model.joint("pendulum_joint").dofadr

paused = False

# ------------------------------
# Controller
# ------------------------------
def controller(model, data):
    theta = data.qpos[PEND_JOINT]
    theta_dot = data.qvel[PEND_VEL]
    phi = data.qpos[ARM_JOINT]
    phi_dot = data.qvel[ARM_VEL]

    # Normalize angle [-pi, pi]
    theta = (theta + np.pi) % (2*np.pi) - np.pi

    # Physical parameters
    m = 0.15
    g = 9.81
    l = 0.3

    torque = 0.0

    if abs(theta) < 0.3:
        # --- LQR capture near upright ---
        x = np.array([theta, theta_dot, phi, phi_dot])
        K_lqr = np.array([80.0, 10.0, 2.0, 1.5])
        torque = -K_lqr.dot(x)

        #arm stabilization
        torque -= 0.1 * phi_dot
        torque -= 2.5 * phi
        
    elif abs(theta) < 0.33:
        # --- Energy-based swing-up ---
        arm_dot = phi_dot
        E_des = m * g * l
        E = 0.5 * m * l**2 * theta_dot**2 + m * g * l * (1 - np.cos(theta))
        k = 20.0
        sgn = np.sign(theta_dot) if abs(theta_dot) > 1e-3 else 1.0
        torque = k * (E_des - E) * sgn
        # Optional: arm damping to pump energy
        torque -= 0.1 * arm_dot
        

    # Clip torque to actuator limits
    data.ctrl[0] = np.clip(torque, -8.0, 8.0).item()


# ------------------------------
# Reset helper
# ------------------------------
def reset_to_key(key_name):
    key_id = model.key(key_name).id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

# ------------------------------
# Key callback (for launch_passive)
# ------------------------------
def key_callback(keycode):
    global paused
    # keycode comes as an int; use chr() if comparing text keys
    try:
        c = chr(keycode).upper()
    except:
        c = None

    if c == ' ':
        paused = not paused
        print("Paused" if paused else "Resumed")

    elif c == 'R':
        reset_to_key("down")
        print("Reset to down")

    elif c == 'U':
        reset_to_key("upright")
        print("Reset to upright")

    elif keycode == mujoco.mjtKey.mjKEY_ESCAPE:
        # return False to close viewer
        return False

    return True

# ------------------------------
# Main sim loop (passive viewer)
# ------------------------------
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    # initial reset
    reset_to_key("down")

    while viewer.is_running():
        if not paused:
            controller(model, data)
            mujoco.mj_step(model, data)

        viewer.sync()
        time.sleep(model.opt.timestep)
