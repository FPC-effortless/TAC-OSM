# S1OBS001 — source-authorized architecture-independent System1 forensic pre-registration
2026-10-26 · NO training, checkpoint update, DEV/TEST rescoring, calibration or model promotion.

The architecture-independent Observatory from **PR #422** is reused as its
source-identical `research_probe.py`, `research_torch_basics.py`,
`research_interpretation.py` and `research_anticheat.py` modules in this
S1MOTION005-based research branch, because draft PR #422 and draft S1MOTION005
#423 are not merged together. This must not bypass final integration review.

### Source/cut
Use only the ORIGINAL previously frozen S1MOTION005:
- Original run 40283616062, artifact 12632762763 and first original SHA receipt
  `S1MOTION005-ORIGINAL-LEDGER.json`. Only seeds 29001/29002/29003,
  TAC shared, TAC isolated, TAC projected, all six? **Nine neural TAC checkpoints**.
  The separate six GRU comparisons remain as originally scored and are not
  diagnostically relabelled as TAC.
- Licensed and provenance-qualified authentic LeRobot static screw driver v2
  revision `0b873be59c9702f8d3de89022617121970db112f`.
- Fetch **ONLY FIT episode0 and episode1** source-locked SHA Parquet records,
  verify originals against 25 FIT SHA manifest. Use FIT-only real normalization
  originally saved **before any DEV file release**. No DEV or TEST file reads.
- Two B=1 real episode prefix windows T=12 (frame1..12), with source-known
  previous frame0, x[t] and prior recorded action[t-1]. No future
  targets/actions/labels can enter the adapter; actual FIT labels are used
  only to compute post-hoc supervised gradient diagnostics.
- Compile generic adapters with constructed tests, then run original frozen
  three seed TAC x three arm x two real FIT windows = 18 source diagnostics.
  Check every original model checksum, no optimiser calls or weights/buffers
  writes, true source-only hard boundary and coverage ledger.
- The generic engine reads actual 166 floating values/stream instance:
  fast running basis sum (88), associative 8x8 (64), previous observed 14D (14),
  excluding scalar count/boolean, weights, metadata or transient activations.
  Forecast is the genuine last-frame `VelocityAnchoredSystem1.step(...)`
  on the replayed T−1 step state, not a fake native PredictiveStateAgent.
  Motion head remains uncalibrated **observed movement proxy only**.

### Falsifiable questions / interpretation
A. **Memory use**: zero-patch only a named state block at terminal time, inspect
   point forecast, proxy logit, reported scale. Assess whether named memory
   lesions genuinely alter predictions. A local zero-patch can be OOD and
   absence of local effect is inconclusive for all tasks.
B. **Source dependencies / inadvertent paths**: independent AD leaves for
   current measured observation, truly past logged action, current carried
   fast sum, associative memory and true previous observation. Risk logit
   should NOT depend on preceding actual observation through the deterministic
   velocity prior; point MUST depend on genuine previous state. Detect
   unwanted interaction across batch items. Candidate permutation must remain
   INCONCLUSIVE; this model has **no causal candidate-action head**.
C. **Gradient issue**: local point physical MSE, source high-motion BCE, and
   dispersion NLL as separate weighted objectives on authentic FIT episode
   windows; use PR422 arbitrary objective name `gradient_conflict_atlas` and
   explicit gradient norm/cosines for true encoder, gating, point readout and
   auxiliary head. Do not confuse local gradient conflict or magnitude with
   AdamW's actual optimizer step or with a causal proof of predictive damage.
D. **Residual value**: divide measured learned correction from the strong
   velocity base; signed alignment and correction energy against genuine
   recorded one-frame residual. Numbers on tiny FIT prefixes are diagnostics
   only; they do not replace original source-unit DEV scores nor qualify any
   new predictor. No retrospective threshold changes or new training.
E. **Runtime / integrity**: measure CPU replay+query total, exact carried
   float state, model side effect and source check. Runtime NOT hardware SLA.

Outputs: original JSON and complete source anticheat ledger, one readable
honest findings report, automatic tests. Preserve original S1MOTION005 full
gate FAIL 2/3, earlier S1MOTION004 FAIL 2/3, and prior negative findings.

All diagnoses are architecture-independent **engine+architecture-specific trusted
adapter**, not global claims of the whole TAC architecture family.
