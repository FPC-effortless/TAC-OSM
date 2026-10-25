# SIM001 — frozen ME099 checkpoint in MuJoCo articulated arm and planar vehicle proxy

**Date:** 2026-10-25. **EVIDENCE TIER: SIMULATION_DIAGNOSTIC_ONLY / OOD ZERO-SHOT TRANSFER / NO ROBOT OR AUTOMOTIVE CONTROL QUALIFICATION.**
**Source:** [original ME099 checkpoint index](../replay-bundles/ME099/FIT_INDEX.json); [real source scoring](../mathematical-engineering/me099/RESULTS.md). All ME099 live neural weights, real-data source and score gates are unchanged. The physically simulated results below must not be included in real-world capability promotion or compared to protected unseen physical cohorts.

**Reviewed execution:** [GitHub Actions run #40185878515](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40185878515), 2026-10-25, completed **SUCCESS**, five checkpoint checksums validated, **40 conditions** (five seeds × two physics proxies × baseline/assisted × visible/blackout), 160 physics steps per condition, no model training. Original checksum receipts are SHA256 pinned in ME099 FIT_INDEX. GitHub Actions simulation-only JSON artifact ID **12596242383**, containing five per-seed records with per-condition metrics. No model/optimizer/sampling source byte was overwritten.

## Question and method

Can original ME099 physical world-state forecasts, trained solely from authentic ALOHA mobile-cabinet 14D joint observations and *lagged executed commands*, transfer **zero shot** to:

- A MuJoCo 2-degree-of-freedom articulated gravity/damping/motor **arm** tracking a fixed two-joint target; and
- A MuJoCo planar rigid body with force/yaw control acting as a **vehicle proxy** tracking a position target? **It has no real tires, Ackermann steering, road/perception/obstacle/other-agent physics and must not be described as a full car simulator.**

The *same* classical PD robot controller or heading/speed controller remains in charge. "Assisted" may consume only an individually bounded 25% hint from model output when sensor data is present; it is not a neural learned policy. 50 consecutive **synthetically** missing sensor observations at physics steps 60..109 are generated without changing the actual source checkpoint or accessing hidden simulator truth inside the model.

The checkpoint was trained on ALOHA robot input domains, not these task domains; a shape-compatible mapping uses original FIT channel mean/variance and maps the simulated task's first four physical coordinates into original 14D features, leaving unused channels at learned source means. Such a mapping is **not scientifically justified for car dynamics or even this arm**, and creates an out-of-distribution stress diagnostic only. Past executed action enters the source model as lagged input; proposed commands are NOT evaluable as causal counterfactual actions. Checksum+parameter checks ensure the *actual trained model* was called and never silently replaced by a hand-coded dynamics function.

### Blackout-contract correction

Initial run [#40185466418](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40185466418) and first five-seed run [#40185618805](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40185618805) evaluated a **padding/invalid observation output** during the blackout, yielding zero interval width. **INVALID AS A CALIBRATION TEST**: in ME099, valid=False explicitly masks invalid events and suppresses their forecast outputs. The earlier 0% interval coverage/zero-width figures are **withdrawn as blackout-calibration evidence** and retained here solely as test-audit history.

**Corrected run:** at time index 59, compute conditional coherent forecasts for horizons 1..51 using only the last actual observation and previously executed recorded action. For blackout steps 60..109, retain the last genuine memory state, never write hidden simulation truth, and compare h=2..51 predictions to actual *subsequent* physics outcomes. The distribution is no longer a padded output; it has nonzero width. It remains a forecast under unknown future controls, not a model of candidate-action effects. Original model code, weights and freeze stamps are unchanged.

## Actual corrected measured results

Mean over **five original frozen seeds**. Each seed evaluates the same deterministic two physics proxy tasks and conditions: do not regard 40 simulator runs as independent natural world episodes. Position errors are in the simulator's respective positional coordinates; robot angles and vehicle distances have different units and cannot be directly compared.

| Domain & condition | Controller | Model position MAE all | Persistence MAE all | Model MAE inside 50-step gap | Persistence gap MAE | Model nominal90 gap coverage | Gap mean width | Final goal distance | Mean goal distance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arm, visible | classical | 0.077814 | **0.076104** | n/a | n/a | n/a | n/a | **0.985709** | **0.852567** |
| Arm, visible | neural advisory | 0.078645 | **0.076937** | n/a | n/a | n/a | n/a | 0.995525 | 0.857243 |
| Arm, 50-step gap | classical | **0.606852** | 0.623623 | **1.763141** | 1.819614 | **1.00%** | 0.313485 | 0.690993 | **1.150608** |
| Arm, 50-step gap | neural advisory | 0.503252 | **0.482130** | 1.412066 | **1.347806** | **0.20%** | 0.377594 | **0.504653** | 1.249840 |
| Vehicle proxy, visible | classical | 0.010779 | **0.005424** | n/a | n/a | n/a | n/a | **0.489125** | **1.307102** |
| Vehicle proxy, visible | neural advisory | 0.010779 | **0.005423** | n/a | n/a | n/a | n/a | 0.489218 | 1.307380 |
| Vehicle proxy, 50-step gap | classical | 0.107019 | **0.059463** | 0.320358 | **0.179484** | **11.4%** | 0.270083 | 0.455121 | **1.294098** |
| Vehicle proxy, 50-step gap | neural advisory | 0.107002 | **0.059441** | 0.320299 | **0.179414** | **11.4%** | 0.270042 | **0.454936** | 1.294232 |

**Visible-sensor nominal90% interval coverage** is only ~1.94% for the robot classical condition and ~42.88% for the vehicle proxy classical condition. Calibration is severely invalid in both out-of-distribution settings. No confidence interval for across-domain natural ability is valid from these deterministic proxy environments.

**Action outcome interpretation:** the arm's final distance happens to improve under dropout with neural advice, but its **average** distance over time worsens. Clean robot control also worsens in both mean/final tracking. Vehicle changes are negligible, not a measured neural benefit. This is **NO RELIABLE DEMONSTRATED CONTROL BENEFIT**. Do not label a model that consumes past commands a trained action-conditioned policy. A control authority or safety proof is absent.

**Timing:** observed CPU host wall time including Python model, observation adapter and MuJoCo physics was roughly **3–4.5 ms per physics step** in this runner under tested settings, shorter inside gaps partly because the multi-horizon forecast was cached at blackout onset. This is NOT standalone model-only latency, full app/hardware jitter, an ECU measurement, a <1ms guarantee, or a CPU-time-matched architecture comparison. The original core is 608 FP32 bytes **exclusive** of extra metadata, model weights, simulator, host and runtime allocations.

### What the experiment establishes, and cannot establish

**Established:** original Git-owned real-trained weights were loaded; two physics proxy loops ran and produced finite outputs; source-index SHA256 was checked for all five; original recurrent state is causal; on these OOD proxies, clean one-step persistence is stronger and the genuine blackout forecasts are grossly undercovered; steering/arm tracking gets no reliable help from model advisory; the model still has no current-command/counterfactual mechanism.

**NOT established:** that TAC-OSM has a safe robot brain, automotive planning, obstacle perception, broad generalization, useful real-world sensor-gap calibration, sub-millisecond timing, or a deployable car driving product. Simulation is not a substitute for independent real measured action-conditioned results.

## Specific next experiment — real trained actuation rather than adapting the wrong checkpoint

1. **Audit source availability and actual action/observation meaning:** qualify permitted ALOHA real command/event data for a source-conformant command-conditioned model; for vehicles qualify actual independently sourced wheel, steering, IMU, GPS, camera/obstacle and observed outcome data. Keep device-lane outputs separate.
2. **Correct action causality:** the model must take *proposed current action* at the correct prediction time and predict conditional future distributions; train actual model weights with valid supervision, log predictions before effects and evaluate on held-out authentic outcomes. Do not call passive correlations do(a) identification.
3. **Use realistic simulator controls only after real model development:** multi-actuator MuJoCo arm with safe torque and limits; genuine vehicle model with tires/steering (e.g. validated CARLA task) and bounded perception, obstacles, emergency fallback. Unseen sim environments serve as stress diagnostics, not promotion data.
4. **Verify safety/uncertainty as hard gates:** missing-sensor causal forecasting at correct horizons, joint likelihood/CRPS/coverage, collision/cross-track/regret, stationary/moving regimes, real actuator latency and a classical controller backstop. Model must earn authority; start read-only.
5. **Preserve research rules:** primary real-world data/untouched external evaluation, exact six families, original checkpoint SHA frozen, 95k–105k deployed trained params and ≤1% paired parity for architecture superiority, matched actual training data/compute/tuning and honest peak memory. CPU runtime reported separately.

**Result:** SIMULATION DIAGNOSTIC EXECUTED / OOD TRANSFER FAILED / NO REAL WORLD CAPABILITY PROMOTION. Original ME099 full real-data gate remains **FAILED**.
