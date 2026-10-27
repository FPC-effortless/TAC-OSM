# LATFUSE009 original real-event gradient survival diagnosis

**2026-10-27 — full recorded-source, FROZEN original LATFUSE008 checkpoint diagnostic SUCCESS. No new model fitting, no protected source, no claimed recall or causal policy.**

[Original source-verified CI run #40448135205](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40448135205) **PASS**. Source UCI501 original physically recorded Beijing station readings SHA256 d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8. Same **four first authentic May2014 PM2.5 high-event/ordinary station-hour matched pairs** from LATFUSE008 source-only cohort. Original two seeds80701/80702×two source-trained architecture arms, all four checkpoint SHA verified before/after, original 99,390 trained/deployed learned params and 6,800byte FP32 recurrent state unchanged; no optimizer or gradient update of original weights.

### Protocol and meaning

Original source observations for 24 hours before measured high/ordinary event and event timestamp processed under no-grad; frozen event-time state detached into independent differentiable leaf tensors. Process identical subsequent actual original 16/64/128/512 recorded hourly sensor/masks/elapsed/owner observations, then issue original model known past timestamp historic query. Only AFTER event logits are computed do we read original physically measured PM2.5 source event label for binary cross entropy. Compute **source-local gradients of genuine event BCE w.r.t event-time associative, fast and slow states and current query-time equivalent states**. No future target or event oracle key supplies model query; original learned parameter tensors are kept read-only.

Crucial: this is **dLoss/dEventTimeSTATE**, NOT direct dLoss/d original event write key/value *parameters* or real factual recall. Initial event state was detached so the model's original event-time encoder/write Jacobian is deliberately not part of this result. It tests suffix gradient transport from later source event query to earlier memory.

### Original associative-query model: two-seed mean event-time gradient / final query-time gradient

| Genuine source delay | Associative state gradient ratio | Slow state gradient ratio | Fast state gradient ratio |
|---:|---:|---:|---:|
| 16 hours | **0.8227** | 0.5785 | 0.01983 |
| 64 hours | **0.6529** | 0.3507 | 0.000233 |
| 128 hours | **0.4690** | 0.2683 | 4.62×10^-7 |
| 512 hours | **0.0842** | 0.1544 | effectively 0 |

Associative memory has a non-negligible EVENT-STATE historic BCE learning sensitivity to a future learned event-query loss over 16–128 authentic intervening source hours; at64h gradient is about65% of current associative query gradient norm, while the fast-state derivative all but disappears. This is **not** proof that actual key/write parameters are receiving sufficient information or that the history target is decoded correctly. At512h the associativity loss gradient shrinks to8.4%, consistent with earlier LATFUSE005 source-measured overwriting/erasure.

### Exact no-association control confirms isolation

For both actual source-trained baseline seeds, historical event-only BCE gradients wrt query-time and event-time associative matrix are **unconnected (None / zero as diagnostic)** at all four delays. This is expected: LATFUSE008 uses current fast×slow feature interaction to decode the same requested past clock without reading stored M. It does not establish that baseline has no associative memory in its ordinary +6-hour forecast path. Fast/slow gradients are connected; under original source they attenuate through long recurrence. All four trained model source checkpoint SHA and original source archive rechecked unchanged.

### What this falsifies

The simple hypothesis that LATFUSE007 failed to learn64h historic source recall **because NO BCE gradient can propagate through associative recurrent state over64 steps** is inconsistent with this measured checkpoint-local gradient path. The source-conditioned recurrent matrix is substantially connected, but predictions remain poorly ranked/calibrated. The more useful next failure mechanisms are *wrong event content encoded into values*, *historical query time/recipient address misalignment*, *large signal-to-noise from other writes/interference*, and *very short 32-update source supervised budget*. The entire source model was originally trained from scratch only32 steps, so attributing the failure specifically to one of these mechanisms would be speculative until independently checked.

At512h both original matrix perturbation and event-gradient survival become small; past LATFUSE005 showed directional beta overwrite is a substantial degradation mechanism. Further boosting alpha indiscriminately risks contaminating state with obsolete physical measurements. Controlled source-specific overwrite eligibility/utility remains a later mechanism hypothesis.

### Next source-falsifiable no-fit diagnosis

For each original source-written real PM2.5 event and source lag, capture actual source authorization/write key and value/gate at event without oracle future labels. Compute model-matrix event-change projection under its own learned historical clock query, and compare true event-vs-control logit/prediction after suffix with anchor read Lesion, matched same-clock source control and fast/slow-only alternative. Separately compute backprop from measured historic BCE into event-time write-key/value *functions* through actual source encoder; report sourced gradient ratio and whether gradient actually moves a useful source-visible write slot. Do not begin new adaptive training from current exposed May–June results. When new legitimate real data and complete FLOP parity available, preregister a historical-event aligned key/value write objective as its own arm, holding query constant. No modern Transformer KV cache required to infer this bottleneck.

**Status:** Genuine-source first trained query-only architecture negative, frozen gradient diagnostic adds a new insight. Full data/compute-matched architecture-superiority gate, native factual recall and calibration remain open; original work preserved unmerged.
