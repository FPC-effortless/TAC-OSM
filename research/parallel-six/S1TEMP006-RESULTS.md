# S1TEMP006 — original real 128 vs 512-step memory outcomes and exact weakness

**2026-10-26. Training workflow passed; ORIGINAL scientific gate FAILED 0/3 seeds.** [First genuine model fit and score CI run40291636780](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40291636780), [full original trained model+optimizer+RNG artifacts 12635849541](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40291636780/artifacts/12635849541). All six original checkpoint SHA and all first scores are committed in `S1TEMP006-ORIGINAL-LEDGER.json`; do not re-train after reading these DEV scores and pretend new data.

## Authentic recorded longitudinal evidence and matching
LeRobot ALOHA Static Coffee New dataset (distinct publisher release from prior Coffee), original historical v2 commit `4f8c712349b83a40e25ef43cbddfbcb9816e3e63`, verified publisher MIT license, 50 episodes×1500 chronological robot frame records=75000 original real frames. **All 25 FIT** full 1500-frame episode original file SHA256 committed before any fitting. True FIT-only normalizer and real high-motion-next-joint-change proxy (threshold FIT p90; NOT accident/collision/real danger). Train exact matching 3 seed pairwise original weights; 95,907 trainable weights, 166 FP32 float streaming state=664 bytes (plus count/flags/weights). 192 AdamW source-window updates ×batch4 per arm (64 each H32,H128,H512 interleaved), identical source window plans and 172,032 real frame exposures/arm, identical registered `rho=.1` gradient projection algorithm, same optimizer hyperparameters. All **6 original models, AdamW optimizer state and RNG checkpoint SHA frozen BEFORE any DEV file downloaded**. No TEST35..49 bytes downloaded. Same-publisher ALOHA workcell family, not demonstrated independent site or robot, complete measured training FLOP parity NOT established (although training wall CPU times per arm are comparable).

DEV real episode clusters **10**, preregistered terminal source indices `575,655,735,815,895,975,1055,1135,1215,1295,1375,1455`, total **120 genuine identical one-frame next-state targets per arm/seed**. For each target, an H128 vs H512 source prefix refers to the **same observed time and same factual next state**. Only the historical context differs. This is useful matched real observation evidence, not 120 independent robot trials. Measured high-motion proxy prevalence **7.5% (9/120)**, making auxiliary ranking weakly powered and noisy.

## Physical next measured 14D joint state MSE x 10^-6, lower is better

| Seed | Original additive H128 | Additive H512 | Persistent slow H128 | Persistent slow H512 | Constant velocity |
|:--|--:|--:|--:|--:|--:|
| 37001 | 2.584072 | 2.583737 | 2.583763 | **2.583709** | 2.602963 |
| 37002 | 2.577294 | 2.581854 | 2.576917 | **2.580452** | 2.602963 |
| 37003 | 2.570244 | 2.572447 | 2.570454 | **2.571395** | 2.602963 |

Slow H512 relative to additive H512 improves **0.0011%, 0.0543%, 0.0409%** for seeds 37001/2/3 respectively: ALL below pre-registered **>=1%**. Longer slow H512 relative to the SAME SLOW model H128 improves **0.0021%** in seed37001 but is WORSE by **0.1372% and 0.0366%** for seeds37002/3. Thus neither matched long-history context nor slow-state reallocation supplies the required positive benefit in this genuine source test. The physical MSE improvement over simple constant velocity (all seeds) is small and cannot be credited to superior temporal retention since the additive control also beats velocity.

## Observational high-next-motion AUPRC at same real H512 targets (higher better)

| Seed | Additive history | Persistent slow | Result versus >=90% retention guard |
|:--|--:|--:|:--|
| 37001 | 0.220817 | **0.176435** | FAIL, 20.10% drop |
| 37002 | 0.386284 | **0.406241** | PASS |
| 37003 | 0.083401 | **0.080190** | PASS, despite 3.85% drop |

Only **9 positive label frames** out of 120 in this limited same-task DEV window. No claim of genuine hazard detection or robust event classification; no independent real incident labels. Full study scientific performance gate **FAIL 0/3** due missing >=1% model/context advantages and sometimes event ranking regression. Source and model CI success does not supersede failed science.

## Exact mathematical design failure mode: memory decay timescale

The original fixed scalar real-time recurrence is
`s[t] = (15/16)*s[t-1] + (1/16)*b[t]`, with **NO trainable retention or source-authorized refresh mechanism**. Regardless of the neural network's learned weights, the carry contribution of one isolated old basis value through the 16D slow state is multiplied by **(15/16)^k** after k stream updates, absent new intervening terms:
- Half life **10.7401 frames = 0.2148 seconds at 50 FPS**.
- After 128 updates: **0.0002584** of original isolated information's slow-state amplitude remains.
- After 512 updates: **4.4596e−15** remains (the 10.24-second history test).

This is a provable architectural attenuation mechanism, not a trained learning failure in every representational path. The parallel fast72 mean can still contain old commutative signals and associative memory8×8 can retain order-dependent evidence. The **new slow16 channel**, however, is explicitly engineered for a much shorter time horizon than H512. This explains WHY the particular slow recurrence alone cannot be expected to preserve isolated evidence over 512 steps. It does not isolate the cause of the measured 0.04% point differences nor prove genuine H512 recall absence.

## Observatory + formal proof state
- Actual state is **genuinely persistent**, not a renamed transient slow activation: 72 additive memory floats, 16 decayed recurrent floats, 64 associative, 14 previous physical. Model is 95,907 parameters, 664 bytes floating carry in both arms, no new architecture family/seventh variant.
- Software [architecture & long 1100-frame exact-recursion CI #40290366015](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40290366015) passed, plus [actual generic slow16 adapter CI #40292525154](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40292525154) passed with named slow-state lesions, gradient connectivity and genuine parameter/stream-capacity accounting on constructed fixtures (not real trained capability evidence).
- PNDS draft mathematical companion PR #31 proves local exact-real recurrence boundedness with bounded inputs, input responsiveness, order-swap difference of a^2(y−x), 166-scalar/664-byte nominal arithmetic; **requires actual Lean CI verification** before claiming Lean results. Whole native PyTorch scan AD/FP32 refinement, hardware safety and action-intervention causal semantics remain unproved.
- Original frozen six-checkpoint independent SHA/re-evaluation cold replay is a separate no-fit check and must pass to claim independent reproduction; original first DEV scientific FAIL remains binding.

## Prospective successor — not yet evaluated
The next counterfactual is a **mixed-timescale slow bank** under the SAME 16D bank and active-parameter/state constraints: allocate four slow channels each to causal update rates `1/16, 1/64, 1/256, 1/1024`, preserving exact same 16D total and added state zero. Pre-register on a NEW authentic real recording/task before optimizer or scoring; evaluate matched H128/H512/1024 retention against additive and single-decay controls. This proposal is mathematical engineering, **NOT** retrospective correction to these already-opened DEV results. Detect source actor/demo duplicates, compare full compute (including projected gradients), and require explicit retention-relevant real temporal outcomes rather than just one-frame physical MSE. No promotion of trained slow model from this experiment.
