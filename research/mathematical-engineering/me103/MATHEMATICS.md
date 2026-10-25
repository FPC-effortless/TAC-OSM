# ME103 math and source lineage note

ME103 continues the existing ME102 `SHARED-RECONSTRUCTION-ME102` model. It does not define a new architecture or computational graph. All 15 lineages retain the ME102 model graph, recipe-specific fixed scalar coefficients, tensor layout, and 98,495 trainable and deployed parameters. Each lineage starts from its own authenticated ME102 final model and optimizer checkpoint after 1,536 updates, restores that optimizer state, and receives one fixed extension of 1,536 additional updates: 3,072 cumulative updates. The original checkpoints and ME102 results remain immutable; ME103 continuation checkpoints and receipts are child artifacts linked to their exact parents.

## Inherited mathematical record

The implementation lineage is TAC-OSM source commit `bd7fe3c08f9979d34483d27dcd048eb5cec984e3`, file `src/tac_osm/variants/reconstruction_sensor_world.py`, and the existing independent serial reference is `foundation/v1/reconstruction_sensor_reference.py` in PNDS. The reference exports `serial` and `reconstruction_loss`; it layers the ME102 current reconstruction head, missing-current location update, variance maximum, and masked Gaussian NLL on the inherited ME101 chronological evaluator.

The PNDS mathematics is pinned to `FPC-effortless/pnds-lean` commit `424938148dd2f81ef03022baab0bc838943d9a81`, file `PNDS/ReconstructionSensor.lean`. Reuse its six reciprocal-manifest Lean symbols verbatim:

- `PNDS.ReconstructionSensor.futureVariance_at_least_floor`
- `PNDS.ReconstructionSensor.futureVariance_at_least_inherited_noise`
- `PNDS.ReconstructionSensor.futureVariance_positive_of_nonzero_noise`
- `PNDS.ReconstructionSensor.futureMean_unchanged_when_observed`
- `PNDS.ReconstructionSensor.ageFloor_monotone`
- `PNDS.ReconstructionSensor.readTrace_preserves_trace`

The defining laws remain those of the pinned ME102 record. In standardized log coordinates, the current estimate is `anchor + 2*tanh(head_mean)`, current variance is `0.0025 + softplus(head_scale)^2 + inherited_latent_variance`, future location adds the inherited affine response times the missing-channel current correction, and future variance is the maximum of the inherited modeled variance and the FIT-derived age floor. Age remains capped at 256 hours in the implementation. The six theorems establish only the stated scalar properties under their assumptions; they do not prove full neural-network behavior or empirical calibration.

ME103 changes training lineage and cumulative exposure, not the model equations. Its training-driver source commit must be pinned before registration: `fb50159be22f936f6fb0ebdc5fe1b9ab7805689f` is a placeholder and must be replaced with the immutable commit SHA. Record the exact ME102 checkpoint SHA for each parent and bind each continuation receipt to that parent, source commit, original optimizer state, restored RNG/sampler state, and the new 1,536-update budget. The continued model still has 98,495 learned parameters; continuation does not add parameters.

## Scope and open obligations

The existing reciprocal variant row remains `SHARED-RECONSTRUCTION-ME102`, parent `SHARED-BELIEF-SENSOR-ME101`. Do not append an ME103 architecture-variant row or alter the historical 17-version index and six active model families: ME103 has no new model graph. Add ME103 continuation evidence as a linked training-lineage record while preserving the ME102 source, checkpoint, optimizer/RNG, outcomes, and failures.

Before any continued-fit claim, audit all 15 fitted graphs against the unchanged implementation at FP32 and FP64, including outputs, state/trace, loss, and gradients, and verify restored continuation behavior. These required audits do not upgrade the existing formalization status: full neural and IEEE refinement, learned calibration, and real-world qualification remain OPEN. The one-extension study covers all 15 original lineages under the unchanged protocol and exposed 2015 development cohorts only. Keep 2016 and 2017 TEST cohorts closed and unscored. Results describe these continued lineages; they do not establish independent confirmation, generalization, or architectural superiority.

PNDS continuation record frozen before optimization:
https://github.com/FPC-effortless/pnds-lean/blob/b12a8d9/foundation/v1/ME103_CONTINUATION.md
