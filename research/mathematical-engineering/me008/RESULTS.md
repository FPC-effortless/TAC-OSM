# ME-008 — Equal-step native TAC vs GRU; real-world future shift exposes false-motion weakness

**Outcome:** Native TAC's originally suspected training-budget handicap was real, but 1,000 optimizer steps do not produce robustness: on the registered week March23–30, **every learned model loses to persistence**, including TAC and GRU. Research-only, no architecture promotion, no merge.

## Frozen original training, independent weights

Original source [workflow run 39382850579](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39382850579), artifact `12245779076` (`me008-original-native1000`), original fit commit `a0f7f90efd1adb332d3e02cdd2e3f1c291fa69b5`. Three fixed seeds 240401/2/3, original first-20days TRAIN only, unchanged genuine 12-channel ME004 input, 32 history reads and 12/60 future horizons, Huber beta=.2, AdamW same hyperparameters as matched GRU41, exactly 1,000 updates each. Original 450- and 1,000-step checkpoints; all six SHA256 digests **independently checked** against original result ledger.

- At 450 updates, initial two seeds reproduce the original ME004 TAC scores (0.2687311, 0.2714250) to recorded precision; third 0.2706702.
- At 1,000 updates, exploratory old validation: seed240401 **0.251992**, seed240402 **0.264861**, seed240403 **0.245172**; three-seed mean **0.254008** vs GRU41-1000 **0.257924**. This older validation was already exposed and cannot qualify a new success.
- Native TAC1000 took 16.18 sec CPU per fit vs GRU1000 6.86–7.10 sec on original recorded training. Matched optimizer steps, not matched computational budget or wall time.

## First-exposure frozen original time-shifted test

[Workflow run 39383282096](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39383282096), original first-exposure source commit `6fcf99bfcf1f18054e5858940729903256a4c254`, preserved archive `12245708188` (`me008f-first-frozen-equal-step-future`). Registered **March 23–30, 2020**, official UCI MetroPT-3 source digest unchanged, **53,283 genuine source observations**, **4,417 forecast windows**, no optimizer steps and no post-cutoff source label access. Original ME004/ME006/ME007/ME008 weights each SHA-bound before source exposure.

| Frozen model | Seed 240401 | 240402 | 240403 | Mean normalized forecast MAE |
|---|---:|---:|---:|---:|
| **Persistence** | — | — | — | **0.595477625** |
| Native TAC450 | 0.719128109 | 0.719332293 | 0.679451329 | 0.705970577 |
| Native TAC1000 | 0.729375123 | 0.725544007 | 0.647413378 | **0.700777503** |
| GRU41-450 | 0.769256894 | 0.769399185 | 0.771194608 | 0.769950229 |
| GRU41-1000 | 0.774648145 | 0.747377829 | 0.745961964 | 0.755995979 |
| Frozen TAC + ME006 direct adapter | 0.711241618 | 0.711208081 | — | 0.711224849 |

Native TAC1000 mean forecast MAE is **7.30% lower than matched-step GRU1000**, but **17.68% WORSE than persistence**. Every frozen learned model failed the real-world baseline on this later week. It would be invalid to claim broad TAC superiority from winning GRU, and invalid to claim that simply training longer solves world-state prediction.

## Change-event diagnostic
The real future window contains **9,603** hindsight-selected large physical-sensor changes, with **5,776 positive, 3,827 negative**. Change-conditional magnitude MAE (lower better): persistence **2.99694**, TAC1000 **1.65708**, GRU1000 **1.75436**, TAC+direct **1.83951**. Change-conditional sign agreement: TAC1000 **90.06%**, GRU1000 **88.52%**, TAC+direct **85.17%**.

**Interpretation:** TAC1000 carries meaningful conditional signals for actual transitions, while apparently making too many incorrect nonzero residual predictions on otherwise stable periods; this is a **hypothesis** requiring an error decomposition, not a proven causal regime-change mechanism. Direction scores remain conditional on knowing in hindsight that a change occurred, not a deployable advance warning metric.

## Scientific and mathematical decision
- **FAIL** any claim that ME004/ME006 native learned recurrent/association/adapter models have achieved a robust real forecasting advantage over persistence across future weeks.
- **KEEP** the unchanged native two-timescale architecture as an empirical research component because it learns nontrivial transitions and beats capacity/optimizer-step matched GRU on this one week, but computational efficiency and robustness remain unsolved.
- **NEXT HYPOTHESIS:** reliability-gated residual forecasting. Define `y_hat = last + alpha(history, uncertainty, dynamics, h) * delta_native`, `0<=alpha<=1`; optimize ONLY on available training/calibration records, compare to zero-delta persistence and fixed shrinkage, and validate on an entirely unopened later period **not March23–30**. Track false-positive drift on low-change observations, change recall/precision, MAE and calibration. Do not claim a convex mixture's pointwise error can never exceed both endpoints: it has no such guarantee.
- Retain every other unified-plan direction (CDL-PHR, BDH neuroplasticity, neural CASM, entity learned memory, VSA/executable operators, oscillatory gating, PNDS Lean) as independent research hypotheses, not automatic components. No synthetic success claims, no action-benefit claims without authentic action/outcome/decision-cost evidence.
