# S1OBS001 — trained System1 failure localization with the architecture-independent Observatory

**Original source forensic run (2026-10-26)**: [GitHub Actions #40286629873](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40286629873) **PASSED** source-level PR422 generic Observatory tests on **nine original frozen S1MOTION005 TAC checkpoints** (shared/isolated/projected × seeds 29001–3), with two SHA-pinned real FIT robot episodes and their original FIT-only scalers. Source anti-cheat ledger **PASS 5/5 required checks**, including the later raw-original-source future poisoning audit. Source original S1MOTION005 trained/first DEV experiment [run40283616062](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40283616062) full prereg scientific gate **FAIL (only 2/3 original seed groups qualify)**; **no new training or DEV/TEST scoring in this analysis**. Full generic diagnostic JSON original [artifact12634024236](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40286629873/artifacts/12634024236), including all 18 one/two real-FIT window checks and complete gradient atlases.

**Stronger source-poison repeat completed:** independent explicit no-training [GitHub Actions #40287379755](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40287379755) **PASS** with [artifact12635180654](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40287379755/artifacts/12635180654): the auditor changed the **authenticated raw FIT source frame13 future next-state measurement**, reran the genuine FIT-normalizer/source-episode window encoder, and verified that historical decision inputs (frames1..12 and previous logged action) remained unchanged while the actual future measurement changed. The five mandatory source anti-cheat checks all passed, nine frozen SHA model checkpoints × two source arrangements were tested, no optimizer updates occurred, and no DEV/TEST source file was downloaded. This verifies the tested decision-time source cut for this pipeline, **not** the absence of every other possible covert channel. The first #40286629873 receipt remains separately valid and is not misrepresented as having contained the later stronger test.

## 1. CONFIRMED local raw-gradient conflict at the affine associative-memory gates

All numbers below are **already weighted** .1×motion BCE and .01×Gaussian NLL gradient norms from **authentic original FIT episodes0 and1, B=2, 12 prefix frames**, evaluated on the nine original frozen model weights, with separate gradient graph per loss and **zero optimizer steps**. The three projected-model checkpoints are most relevant to the current S1MOTION005 model. No claims about whole training or global causal optimizer effects.

| Frozen projected TAC seed | point vs motion cosine | point vs uncertainty cosine | motion gradient / physical-point gradient L2 | uncertainty gradient / physical-point gradient L2 |
|:--|--:|--:|--:|--:|
| 29001 | **−0.921** | **−0.963** | **35.1×** | **68.9×** |
| 29002 | **−0.960** | **−0.979** | **57.5×** | **79.3×** |
| 29003 | **−0.518** | **−0.959** | **36.8×** | **95.9×** |

Additional **same-source encoder** gradient ratios on those three original projected TAC models: motion BCE relative to point **267×, 489×, 662×**; uncertainty NLL relative to point **454×, 761×, 202×**. These are gradient-norm ratios, NOT loss scalars or applied AdamW updates. Point readout gradient ratios are smaller; the worst conflict is in upstream world encoding and memory admission/gating, not a disconnected physical decoder.

**Exact mechanism:** the same native encoder and gate weights support two observational auxiliary objectives that can produce large, locally *anti-aligned* gradients relative to actual physical next-state MSE. Original S1MOTION005 **projects and norm-budgets** those auxiliary gradients BEFORE AdamW, so the raw conflicting gradients do **not** all enter the optimizer. This explicitly **does not prove** the remaining projected AdamW steps degrade the point forecast, or that gradient competition alone explains the original event proxy failure.

## 2. CONFIRMED graph wiring: observed velocity is NOT a direct motion-proxy input

On **all nine original checkpoints**, the generic PR422 input-leaf graph found:
- Physical next-state point forecast depends directly on `previous_measured_observation` buffered source input; its gradient norm to that buffer was **5.2915** on these B=2/14D point outputs (analytic velocity prior).
- The motion-proxy logit has **NO direct AD edge** to that buffered preceding physical observation (**connected=false**); this is confirmed by the actual `System1ParallelWorld._project(...)` implementation, whose head receives `fast, slow, old_read, predicted_core`, but not `x[t]−x[t−1]` as its own explicit feature.
- This **does not mean no past motion is accessible to the proxy head**: its recurrent `fast_running_sum` and `associative_memory` contain earlier observations, and the actual generic gradients and lesions show nonzero history influence. The narrowly proven omission is the direct one-step measured-velocity feature.
- Motion-proxy labels are actual high joint movement as FIT-thresholded measurements, NOT failure, accident, reward, contact state or causal `do(a)` effects. The original high-motion AUPRC for projected TAC original seed29001 was **0.17339 vs shared-TAC 0.32570**, violating the preregistered >=90% retention gate; lack of direct velocity feature is a **candidate mechanism**, not proven cause. A separate prospective, parameter/compute-matched feature-ablation test is needed.

## 3. CONFIRMED memory topology and temporal information bottleneck

Source review of `parallel_continuous_world.py`, `streaming_continuous_world.py`, `system1_parallel.py`, and `system1_velocity_anchored.py`:
- Full floating persistent state per instance **88** fast prefix-basis sum + **64** 8×8 associative matrix + **14** last observed physical state = **166 FP32 floats / 664 bytes**, plus counters/flag, weight/normalizer storage and runtime. Probe runtime returned **1328 bytes** for actual batch B=2. The 95,907 learned model weights are NOT included.
- The named `slow` features are **transient forward activations**, NOT a distinct persistent slow-state tensor. The source uses input-conditioned `slow_map` on current features and a mean historical fast context. The native streaming state has no dedicated persistent `slow` field.
- The fast historical summary is the cumulative **SUM of independently encoded per-step basis** and uses the prefix mean for context. Its carried value cannot distinguish reordering of earlier observed (state,already-logged-action) pairs in exact arithmetic. The **only order-sensitive learned carried history in these stream tensors** is the 8×8 associative memory. The last measured observation buffers only the most recent state. This is an **explicit architectural expressivity limitation**, not a trained proof that a specific task needs more capacity.
- The adapter now contains a constructed old-history permutation + zero-associative-state counterexample regression. That test is a source graph property, NOT a physically valid evaluation example or authentic robot outcome. The constructed old-history permutation + zero-associative test also **PASSED** with the later adapter CI [run40287590786](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40287590786); it documents a computational graph property, not real trajectory success.

## 4. ASSOCIATIVE MEMORY IS ACTIVE, not "dead"

The genuine B=2 original 12-frame FIT prefixes yielded positive source model local dependency gradients and deterministic zero-lesion output changes on **all three original projected TAC checkpoints**:

| Seed | Associative-memory zero-lesion absolute point change | Fast-history zero-lesion point change | Assoc-memory zero-lesion change to proxy **logit** |
|:--|--:|--:|--:|
| 29001 | 0.000210 | 0.002373 | 0.119 |
| 29002 | 0.000283 | 0.001369 | 0.177 |
| 29003 | 0.000355 | 0.001844 | 0.164 |

State lesions are local **out-of-distribution hidden-state perturbations**. Values across a physical forecast dimension and a logit dimension cannot be compared directly as if they had the same units. Within the point head, the associative-memory lesion changes are consistently smaller than those of the fast context on the specified real prefixes. This indicates **modest local point-forecast sensitivity** from associative storage, NOT memory non-use, long-term recall failure or a proof of why original DEV gating failed.

## 5. Provenance / runtime / observability / other weaknesses

The nine checkpoints are original, SHA-verified, and immutable; model parameters and buffers never changed. Original genuine FIT episodes 0 and1 were checked against pinned per-episode SHA. The original 5 required generic anti-cheat source checks passed. Entity isolation PASS only on an actual B=2 sample assembled from two separately authenticated episodes; single-episode entity isolation is **INCONCLUSIVE**, not passed. The first three failed diagnostic CI attempts are preserved, reflecting runner mistakes (single-entity coverage assumption, stale sample name, PyTorch tensor concatenation generator); original weights and research data remain intact.

On 3 original projected TAC checkpoints, **B=2, T=12 real-prefix replay + final query** ran at local single-thread CPU medians ~9.47–9.55 ms, not per-step latency nor a hardware/device SLA. Candidate-action permutation is **INCONCLUSIVE_NOT_DECLARED**, as the model has no candidate-action intervention/policy head; none of this qualifies autonomous decisions or safe actuation.

The current source is one same-publisher real ALOHA task. The original cross-architecture comparators did not establish fully equal training FLOPs. The original adverse-event-named head is supervised on high-motion proxy, so there is a **semantic hazard** if downstream software treats it as calibrated danger. Model `System1Forecast` explicitly refuses that authority, but the name should eventually reflect the source label ontology.

## Root cause classification and engineering priorities (NO new fit)

**Directly established architectural facts**: (i) shared auxiliary gradient conflicts at memory admission/gates and source encoder; (ii) motion proxy lacks a direct previous-measurement velocity feature; (iii) no distinct persistent slow-state, and fast prefix sum order invariance; (iv) true associative memory is numerically active but locally modest in the physical point head; (v) actual candidate-action dynamics/decision execution not implemented.

**Plausible but unproved explanations for original S1MOTION005 multitask failure**: risk classification's weak/indirect access to velocity, limited temporal-feature organization, and post-projection gradient budget tradeoff. These need a **new pre-registered authentic data source with equal parameters, original source exposures and measured full training FLOPs**, including a no-memory read-off, explicit-velocity-risk-feature matched-head, persistent-slow-memory matched-state, and auxiliary-gate controls. Do not re-fit to already opened Screw Driver DEV or examine its closed TEST to tune any of these.

**No promotion or merge.** This is actual instrumented diagnosis, not a proof of all future/other TAC-OSM architectures or a total proof of cheating absence. The original S1MOTION005 full scientific gate remains FAILED (only 2/3 seed trial conditions passed).
