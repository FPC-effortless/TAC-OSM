# LACUNA-MATCHED002 — context-relevant historical contribution

This separately registered FIT-only follow-up addresses a limitation of001:
fixed-age events may be irrelevant to the current prediction. At each query,
select the most recent prior continuation in each age bin with the SAME observed
four-byte context. Selection uses only already observed bytes, never whether
its continuation matches the future answer. Original001 and its gate remain
unchanged; this is not independent generalization or a retrospective rescue.

Same three original640-update checkpoints and two PSF FIT owners, now observing
2,048bytes each. The cohort has190source/query/bin cells measured three times
(570measurements); bins can share targetpositions. No iid token/seed inference.

| Historical age bin | Distinct source/query cells | Historical continuation matches next byte | Event norm retained | ΔNLL removing matching event | Wrong event ΔNLL | Utility gate |
| --- | --- | --- | --- | --- | --- | --- |
| 8–31 | 12 | 75.00% | 92.35% | -0.0000301 | +0.0000332 | INCONCLUSIVE_LOW_SUPPORT |
| 32–127 | 60 | 56.67% | 73.06% | +0.0000026 | -0.0000735 | NOT_PASSED |
| 128–255 | 53 | 64.15% | 47.42% | -0.0000011 | +0.0000022 | NOT_PASSED |
| 256–512 | 65 | 61.54% | 22.41% | -0.0000049 | -0.0000034 | NOT_PASSED |

No bin passes. The8–31bin has insufficient support under the preregistered20cell
minimum. Other bins have support, but negligible/negative loss effects, far
below the0.001threshold. Useful past continuations exist in this observed-context
cohort, yet their individual slow-state contribution has not demonstrated useful
native prediction. This remains a fixed-query, fixed-other-state intervention,
not a full trajectory counterfactual or proof that memory can never work.

Current writer keys exactly match the selected historical writer keys
(maximumerror0): this is structural consistency of the source four-byte encoder,
not learned autonomous factual retrieval. Native query is separately learned;
its dot products cannot alone diagnose failure because sign/rotation/distributed
coding affect interpretation. Whole-matrix raw byte recovery at the supplied
historical key remains poor. All addresses and event tables are evaluator-only.

Source d20ece0828367bedcda0564025e8a3ab2dd323ec; successful original run:
https://github.com/FPC-effortless/TAC-OSM/actions/runs/41000661247
Artifact12952978880, ZIP SHA256
`df59a1a9fc7baa67fa5c48d419f8d6db5b0ca9836efe924bd552146adc134f61`.
Original artifact/JSON preserved here; all reported means independently
recomputed from saved rows without inference. CPU39.333seconds, no research
training, GPU, DEV score or TEST. Initial standalone import failure41000254179
occurred before cohort evaluation and remains recorded in ENGINEERING_LEDGER.
Existing PNDS087/88 conditional transport applies; full neural proof OPEN.
