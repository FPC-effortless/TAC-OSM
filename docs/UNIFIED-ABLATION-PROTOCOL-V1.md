# Unified native-model ablation protocol v1

Status: registered research procedure. It does not modify historical experiment contracts or results.

## Required sequence

### Gate 0 — repository integrity

Before any scientific arm:
- all repository tests pass;
- mutable baseline metadata matches the live test collection;
- every experiment contract loads through ExperimentContract;
- no experiment is run when its contract is malformed;
- historical evidence files remain unchanged.

### Gate 1 — information sufficiency

For every learned mechanism:
- enumerate observational equivalence classes;
- test representability of the intended relation;
- test identifiability from learner-visible information;
- verify that no hidden target field crosses the pre-action boundary.

A failure is INSTRUMENT_INVALID for that measurement, not a negative capability claim.

### Gate 2 — predictive representation

Compare observation-only reconstruction, predictive next-state objective, action-conditioned prediction, and predictive plus action-sufficiency objective.
Measure held-out future prediction, action decoding/selection and latent causal sensitivity. Do not select the final representation by the test set.

### Gate 3 — structural discovery

Compare identity latent, learned slot/factor decomposition, and executable structural decomposition. The mechanism passes only if discovered factors improve registered predictive or action endpoints and remain identifiable under interventions.

### Gate 4 — addressing/proposal

Decompose selection into Address -> Proposal -> Route.
Measure proposal recall, conditional route accuracy given target admission, candidate/work counts, and state-addressing work separately.
No proposal miss may fall back to exhaustive hidden-truth selection.

### Gate 5 — execution

Compare exact executable semantics, learned execution-strength ablation, synthetic structural executor, and pinned CASM-S adapter. Execution must consume explicit topology. Internal topology inference from hidden truth is forbidden.

### Gate 6 — verification and repair

Compare no verifier, final-output verification, path verification, and bounded repair. Report initial success, post-repair success, repair attempts and invalidated structures separately.

### Gate 7 — persistent learning

Compare no writes, always-write, verified-write, confidence-gated verified-write, and verified-write plus retraction. The causal endpoint is future held-out performance after the original observation is removed. Memory occupancy and write cost are secondary.

### Gate 8 — operator learning

Compare raw experience replay, PST transition learning, StructMeans abstraction, AXON consolidation, SSA bounded operator retrieval, and SECA proposal plus independent verification. Operator-library additions are accepted only after independent verification and must improve held-out reuse.

### Gate 9 — planning

Compare one-step selection, depth-2 model-predictive search, and deeper bounded search. Every planning arm uses the same frozen transition model. Search compute is counted separately from representation and execution compute.

### Gate 10 — multimodal

Run separate modality arms first: language, image, audio. Then cross-modal image-text, audio-text, image-audio, and tri-modal tasks when the dataset supplies all three views. The shared core is compared with modality-specific transition controls.

## Statistical discipline

The unit of inference is the registered task or episode, not a pooled tensor element. Seed aggregation is performed from raw per-example counts or paired differences. Small denominators remain labeled unstable rather than converted into attractive ratios.

No threshold is tuned on test data. A failed primary endpoint does not create a new confirmatory arm without a new preregistration.

## Provenance

Every result artifact must contain experiment ID, contract fingerprint, code commit, checkpoint hash, train/validation/test seed IDs, dataset or synthetic-generator fingerprint, intervention configuration, exact metric definitions, gate outcomes, and raw counts needed to reconstruct summary metrics.

## Promotion

A component is promoted only when the relevant instrument gates pass, the primary endpoint meets its preregistered criterion, the result survives any independent seed or condition required by the contract, and the result's exclusions are copied into the evidence register.

Promotion of one component never silently promotes another component.

## Multimodal interpretation

Synthetic success establishes only synthetic structural modeling. Real-data success establishes performance on the registered real dataset/task. Neither establishes human-level understanding. Cross-modal transfer and persistent reuse require their own evidence.
