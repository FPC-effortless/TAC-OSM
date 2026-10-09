# TACOSM-PLM-LEARNED-ADDRESS-002

Status: **PRE-REGISTERED — measurement workflow implemented**

## Scientific question

Can a generic learned associative address metric preserve the raw-dot reference
under isotropic Gaussian query noise while exploiting a structured anisotropic
query channel?

This is a mechanism experiment for learned addressing. It does not test the
full PLM capability gate, long-horizon persistence, semantic memory, or C5.

## Registered design

Stored address keys are independent, unit-normalized random 16-D vectors.
The query is derived from the target key and noise.

The isotropic condition uses:

\`q = normalize(k + sigma * epsilon)\`

with sigma in \`{0, 0.05, 0.1, 0.2, 0.4, 0.8}\`.

The structured condition uses:

\`q = normalize(D k + sigma * epsilon)\`

where \`D = diag([2]*8 + [0.5]*8)\` and sigma is in
\`{0, 0.05, 0.1, 0.2, 0.4}\`.

All seven memory populations M = \`{3,8,16,32,64,128,256}\` are measured with
10,000 trials per condition for each of ten seeds.

## Learned scorer

The scorer is a shared learned bilinear query/key metric plus a learned
candidate-key bias. It is initialized to the identity metric with zero
candidate-bias output. The initialization is explicit because the raw dot
reference is the correct floor for the isotropic condition; at inference the
scorer returns only the learned metric, with no raw-dot branch.

Training is fixed at 750 steps, batch size 512, M=32, alternating sigma 0.2 and
0.4, AdamW learning rate 0.01 and weight decay 1e-4.

## Decision rule

The isotropic criterion is non-inferiority: mean learned-minus-raw accuracy
must be no worse than -0.02 over the complete registered isotropic grid.

The structured criterion is evaluated in the non-saturated regime M=32,
sigma \`{0.2,0.4}\`. Learned-minus-raw mean accuracy must be at least +0.03.

The full M and corruption curves remain mandatory even when a criterion fails.

## Why the structured criterion was amended

The original preregistration required +0.05 at M=3 over the full structured grid.
A pre-measurement feasibility audit showed raw dot was already near the
accuracy ceiling there, making that threshold incompatible with the finite
[0,1] endpoint. The amendment moves the decision to M=32 and the high-noise
subset, preserving the same structured channel while creating observable
headroom.

No learned-address-002 model measurement existed under the previous criterion.

## Integrity controls

Every condition has a deterministic evaluation fingerprint based on the actual
keys, queries, targets, and independent payloads. Within and across all ten
seeds these condition fingerprints must be unique.

The scorer receives only query and stored keys. Independent payloads are
generated for provenance but never enter scoring.

Controls include raw dot on every condition, an exact target-key oracle query,
wrong-key corruption, and random address-order permutation.

No post-run model, seed, memory size, or sigma selection is allowed.

## Claim boundary

A positive result is bounded evidence for learned associative address-metric
adaptation on this synthetic channel. It does not establish semantic entity
resolution, content-derived addressing, open-world memory, multimodal
understanding, long-horizon persistence, or C5 scaling.
