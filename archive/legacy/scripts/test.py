from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[1] / "models" / "test.xml"))
data  = mujoco.MjData(model)
dt = model.opt.timestep

# --- Gains ---
Kp = 150.0
Kd = 30.0
Kx = 2.0
Kv = 4.0

# --- Initial condition (near upright) ---
data.qpos[1] = np.pi + 0.1

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # States
        x     = data.qpos[0]
        theta = data.qpos[1] - np.pi   # shift upright to zero
        xdot  = data.qvel[0]
        thetadot = data.qvel[1]

        # PD control (classic)
        force = (
            Kp * theta +
            Kd * thetadot +
            Kx * x +
            Kv * xdot
        )

        data.ctrl[0] = np.clip(force, -50, 50)

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)

