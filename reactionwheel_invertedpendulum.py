import mujoco
import mujoco.viewer
import numpy as np
import time

# Load model
model = mujoco.MjModel.from_xml_path("reactionwheel_invertedpendulum.xml")
data = mujoco.MjData(model)

# Small initial angle so it actually falls
data.qpos[0] = 0.2  # radians

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 10:
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
