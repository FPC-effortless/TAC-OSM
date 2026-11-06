# ME-002: identity-first and two-timescale native TAC memory — numerical engineering

**Registration/implementation:** 2026-10-19.
**Evidence class:** software/mathematical gradient diagnostics with untrained random recurrent input, not a real-world model experiment.
**TAC-OSM remains a native predictive-executable world-model architecture, NOT an SSM conversion.**

## Claim being investigated
ME-001C showed an earlier bounded native recurrence can lose backpropagated sensitivity to very early state. We compare three intentionally simple formulations, not architecture winners:

1. Dissipative candidate: h_next = (1-g)*0.95*h + g*tanh(Uh+Vx), default gate bias 0.
2. Identity-first candidate: h_next = (1-g)*h + g*tanh(Uh+Vx), default gate bias -4.
3. Two-timescale candidate: identity-first fast half with learned gate, slow half with fixed 0.015 blend into the same proposal. This is a math control, not a learned time-scale discovery.

All use identical input sequences, initial-state probes, projection directions and matched U/V/write initializations per seed apart from deliberate gate biases. Five initializations, sequence lengths 8, 32, 128, 512, float32 CPU.

The identity-first and two-timescale equations maintain [-1,1] coordinate bounds assuming initial h lies in [-1,1] and the proposal stays in [-1,1], by convexity. This forward-invariance is independent of any claim that long-range gradients remain bounded away from zero. Local Jacobians include gate derivatives; state boundedness alone supplies no robust training guarantee.

## Observed local measurements (torch 2.10.0+cpu)
Mean L2 of terminal projected-state gradient wrt initial latent state over 5 fixed seeds:

| Length | Dissipative | Identity-first | Two-timescale |
|---|---:|---:|---:|
| 8 | 0.0420524 | 0.869782 | 0.878346 |
| 32 | 0.000225664 | 0.591299 | 0.613758 |
| 128 | 1.26225e-10 | 0.177610 | 0.199186 |
| 512 | 0 (all 5 seeds underflowed) | 0.00624168 | 0.00828432 |

Sixty seed × model × length records; all terminal states finite and within coordinate bounds. **Interpretation:** explicit identity retention preserves gradient sensitivity in this small untrained construction relative to stronger update/decay. It DOES NOT demonstrate better prediction, better temporal memory use, proof of nonvanishing gradients at arbitrary horizon, causal reasoning, emergent specialization, or training improvement. Note identity-first's less active writing may sacrifice rapid adaptation.

## Experiment decision
*Engineering feasibility*: KEEP identity-first and two-timescale variants as candidate ablations.
*Scientific promotion*: NONE. No accuracy benchmark nor real-world source entered.
*Next real-world test*: only after independent recorded-only replay of ME-001B; compare effective write/update rates, state reset, layer memory interference, and genuine rare-event forecasting under a NEW real-data qualification protocol, not further tweaking the repeatedly accessed heldout SCANIA cohort. Include GRU, persistence, strong drift/LightGBM and parameter/runtime-matched controls. Existing frozen ME-001B checkpoints and SEQ/DEV tests stay unchanged.

Retain all competing designs (explicit episodic memory, TCL/CDL-PHR, VSA operator binding, sparse graph, Hebbian plasticity, rhythmic gates) in the master registry; do not infer failure or success from this isolated gradient experiment.
