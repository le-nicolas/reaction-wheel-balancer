# Model Integrity Investigation Scripts

These scripts were used during the 2026-02-21 investigation that found two structural model issues:
- missing world weld for the base (`lock_base_to_world`)
- miswired `base_y_force` actuator target

They are retained for traceability, not active runtime use.

## Contents

- `analyze_base_y.py`: checks base-y controllability and subsystem structure from linearized dynamics.
- `diagnose_roll_changes.py`: compares roll-channel coupling terms before/after structural model fixes.
- `verify_base_y_stabilization.py`: verifies base-y mode/eigenvalue behavior under tuned weighting.

For maintained runtime and benchmarking, use `final/`.
