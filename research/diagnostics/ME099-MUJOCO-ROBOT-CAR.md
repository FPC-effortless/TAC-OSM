# ME099 frozen checkpoint MuJoCo diagnostic: articulated arm and car-like proxy

**Evidence tier:** SIMULATION_DIAGNOSTIC_ONLY. Not authentic real-world qualification, autonomous driving, robot safety, trained policy or causal action-effects evidence. The simulation does not train or edit the model.

## Original source contract

Reference: [ME099 original FIT index](../replay-bundles/ME099/FIT_INDEX.json) and [original actual robot data scores](../mathematical-engineering/me099/RESULTS.md). Default original seed: 99003, 103,380 learned parameters, with exact SHA256 verification. This is a pretrained **14D ALOHA joint-state and previous-recorded-action trajectory predictor**, not a feedback action policy and not a vehicle model. It was trained on real robot logs, evaluated only on exposed development episodes, with the overall real-world gate FAILED (one of five seeds failed uncertainty coverage). All originals remain immutable.

## Physics environments

1. MuJoCo articulated two-hinge, gravity/damping/motor arm with baseline proportional-derivative target controller.
2. MuJoCo planar thrust/yaw vehicle-shaped rigid body with inertia. **This is NOT a tire/wheel/Ackermann car, urban simulator, perception stack, or road collision model.** The baseline is a classic heading/speed controller.

The frozen ALOHA model receives shape-compatible, deliberately **untrained mappings** of two simulator environments into its original 14D inputs. All other original-domain dimensions are held at original FIT means. This creates an **out-of-distribution diagnostic**, not proof the original physics task generalizes. The simulator runs baseline versus bounded, supervised neural-advisory feedback and clean versus exactly 50 consecutive artificially missing sensor observations. The advisory is accepted only under conservative magnitude controls; if absent/out-of-distribution it falls back to classic control. The model never acts as a trained control policy.

Input semantics are crucial: only the **past executed action** goes into the trained model. It cannot select the best hypothetical current command: that requires new action-conditioned training. During the gap the model receives source-invalid observations with a carried recurrent state and no invented physical measurements; ground truth is accessed for scoring only after advancing MuJoCo. The model's invalid-observation path may produce degenerate distributions, and this must be reported rather than silently repaired.

## Run without hardware

Install CPU PyTorch, numpy and MuJoCo using pip, then execute:

    PYTHONPATH=src python scripts/me099_mujoco_robot_car_diagnostic.py --seed 99003 --steps 160

The associated GitHub Actions workflow installs dependencies in a Linux runner, executes the experiment, prints full metrics and uploads JSON as an artifact. GitHub Actions usage may be quota-limited even though MuJoCo is open source.

## Interpretation

Report actual original SHA256, model parameter count, clean/gap prediction MAE versus persistence, nominal-90% interval coverage, interval widths, goal tracking and advisory usage/fallback, CPU wall time and step count. Simulation-only performance is neither an independent authentic-data comparison nor evidence of trained robotic or automotive control.

To test an authentic robotic/vehicle brain properly, fit a source-authorized real action-conditioned model for each domain; independently validate calibration and horizons, zero-shot physical transfer, logged command authority and physical safety, then qualify with genuine held-out real outcomes and appropriate hardware supervision. **No autonomous vehicle or real-device action is authorized.**
