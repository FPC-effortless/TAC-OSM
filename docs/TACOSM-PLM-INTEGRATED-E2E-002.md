# TACOSM-PLM-INTEGRATED-E2E-002

## Purpose

E2E-001 established that the first jointly trained multimodal persistent-computation chain did not learn the task:

- mean held-out q2 accuracy: 0.5200;
- no-memory control: 0.5200;
- image-shuffle control: 0.5200;
- target-slot attention: 0.060937 with 16 slots.

E2E-002 therefore does not tune the failed model. It changes only the persistent-state address path and asks which interface is responsible.

## Interventions

### Explicit write

The observation entity identity selects the persistent-state write slot.
Read addressing remains learned.

### Explicit read

The query entity identity selects the persistent-state read slot.
Write addressing remains learned.

### Explicit both

The observation entity identity selects the write slot and the query entity
identity selects the read slot.

These entity identities are legitimate structural metadata. They are not answer
labels and are available before action.

## Fixed components

The multimodal encoders, shared representation, CASM Boolean execution,
verifier, optimizer, learning rates, training schedule, held-out compositions,
seeds, and evaluation generator remain fixed from E2E-001.

The E2E-001 measured result is retained as a frozen comparator. It is not
retrained or used to select hyperparameters.

## Diagnostic logic

- explicit write recovery with explicit read failure localizes the problem
  toward learned writing;
- explicit read recovery with explicit write failure localizes it toward
  learned reading;
- explicit-both recovery after both single-path interventions remain weak
  indicates write/read address coherence is the relevant interface;
- failure of all interventions pushes the diagnosis upstream toward the
  multimodal representation or downstream toward action/CASM.

No arm is ranked as a winner. The purpose is causal localization.

## Scientific limits

This remains a synthetic multimodal mechanism benchmark. It does not test
natural language, natural images, real audio, semantic retrieval, arbitrary
learned operators, or scaling laws.

The experiment will preserve all seed-level results and use the same bootstrap
convention as E2E-001 where uncertainty is applicable.
