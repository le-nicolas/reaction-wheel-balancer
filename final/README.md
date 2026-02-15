# Reaction Wheel Balancer (Non-Technical Guide)

This folder contains a simulated self-balancing robot built in MuJoCo.

If you are not an engineer, think of it like this:
- A body is trying to stand up.
- A spinning wheel inside it can create balancing torque.
- A sliding base can move under it to help catch falls.
- Software constantly measures motion and adjusts motors many times per second.

## What This Is

This is a digital twin of a balancing robot that uses:
- a base body/wheel,
- a stick-like upper body,
- a reaction wheel,
- and a controller that tries to keep it upright.

You can run it, push it, and compare different control modes.

Main runtime file:
- `final/final.py`

Main model file:
- `final/final.xml`

## How the Mechanical Design Works

### Parts and roles
- Base body/wheel: the lower mass that contacts terrain.
- Base X/Y slides: let the base shift sideways to assist balance.
- Stick body: the upright part that must be stabilized.
- Reaction wheel: spins to generate equal-and-opposite torque.

### Why this geometry can balance
- The stick has a center of mass above the base, so gravity tends to tip it.
- The reaction wheel can quickly create corrective torque.
- The sliding base can move under the stick to recover when wheel momentum gets high.
- The combination gives two recovery mechanisms: rotational (wheel) and translational (base).

### Visual-only vs dynamics-critical geometry
In `final.xml`:
- Dynamics-critical geoms carry mass and/or collision influence.
- Visual-only geoms use `mass="0"` and `contype="0" conaffinity="0"`.

That separation keeps appearance flexible without breaking physics.

## Physics in Simple Terms

A few key equations drive the behavior:

- Torque-acceleration link:
  - `tau = I * alpha`
  - More torque (`tau`) gives more angular acceleration (`alpha`) for inertia `I`.

- Wheel reaction effect:
  - If the reaction wheel accelerates one way, the body feels opposite torque.

- Control update law (simplified):
  - `du = -K * [x; u_prev]`
  - `u = clip(u_prev + du)`

Where:
- `x` is the estimated state (angles, rates, base motion, wheel speed),
- `u` is motor command,
- `K` is controller gain,
- `clip` enforces safe limits.

Terrain friction and contact also matter: traction changes how effectively base motion can recover balance.

## Control Law (What the software does every cycle)

Each control cycle in `final.py` does:
1. Read noisy, sensor-like measurements.
2. Update a state estimate (Kalman-style correction).
3. Compute command increments with delta-u LQR.
4. Apply safety shaping and actuator limits.
5. Send commands, simulate one step, repeat.

Safety shaping includes:
- wheel speed budget and high-spin handling,
- base authority gating near upright/recovery phases,
- motor torque/slew/rate limits,
- crash-angle protection.

## Why This Is Close to Real Life

The simulator includes practical effects used in real robots:
- Control rate scheduling (`control_hz`) separate from physics timestep.
- Command delay queue (`control_delay_steps`).
- Sensor noise (IMU angle/rate, wheel encoder, base position/velocity noise).
- Actuator saturation and motor envelope limits.
- Crash thresholds and stop-on-crash behavior.

So this is not just “perfect physics + perfect sensors”.

## ESP32 Relevance

This project has a firmware-oriented path for ESP32 in `final/firmware/`.

How simulation maps to embedded control:
- Fixed-rate outer loop (estimator + controller + guards).
- Inner motor/current loop handled by motor driver stack.
- Safety checks and latches mirror embedded expectations.

Useful files:
- `final/firmware/README.md`
- `final/export_firmware_params.py`
- `final/test_export_parity.py`

Parameter export keeps simulation and firmware constants aligned.

## Run It on Your Laptop / PC

### Requirements
- Python 3.10+ (recommended)
- A working graphics environment for MuJoCo viewer

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the simulator:

```bash
python final/final.py --mode smooth
```

Try robust mode:

```bash
python final/final.py --mode robust --stability-profile low-spin-robust
```

### Play mode ideas
Apply scripted push forces:

```bash
python final/final.py --mode smooth --push-x 4.0 --push-start-s 1.0 --push-duration-s 0.15
```

Try hardware-like constraints:

```bash
python final/final.py --mode robust --real-hardware
```

## Known Limitations

- It is still a simulation, not a full physical prototype.
- Contact/friction and sensor models are approximations.
- Real hardware needs integration, calibration, and safety validation.

## Where to look next

- Runtime/controller: `final/final.py`
- Mechanical/scene model: `final/final.xml`
- Firmware integration notes: `final/firmware/README.md`
