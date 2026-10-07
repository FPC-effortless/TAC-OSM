# TACOSM-PLM-FULL-LEARNED-SWILA-001

Status: **PRE-REGISTERED — NOT YET SCIENTIFICALLY MEASURED**

## Purpose

This is the first construction lane whose target is the **full learned PLM**,
rather than a recurrent-controller proxy or a component-only benchmark.

The intended computation is:

`multimodal representation -> persistent MTSK state -> CDL relevance routing -> CASM computation -> action -> outcome -> verifier -> repair -> verified persistent update`

The model must learn these mechanisms jointly. Component ablations are
deliberately deferred until the integrated model is functional.

## Why SwiLA is used

The September 30, 2026 SwiLA paper introduces a recurrent state consisting
of multiple linear-regression components updated from prediction error using
online EM-style responsibilities, with input-dependent query routing. The
reported design keeps recurrent state fixed with respect to sequence length
while making the retrieval mapping piecewise linear. See:
https://arxiv.org/abs/2609.39034

TAC-OSM does **not** inherit SwiLA's benchmark results as PLM evidence. SwiLA
is an architectural antecedent for the MTSK state kernel only.

## MTSK mapping

The integrated PLM assigns six learned SwiLA-style experts into three
retention bands:

- fast: 2 experts
- medium: 2 experts
- slow: 2 experts

Each band has different initial retention/update dynamics. These are
architectural timescale priors, not encoded physical facts.

Temporal responsibilities provide sticky context across adjacent updates.
The CDL receives expert-specific persistent views and sparsely routes a
learned subset to CASM.

## Learned PLM mechanisms

### Representation

Text, image, and audio are encoded independently and fused into a shared
latent representation. No pretrained model, object ID, state label, or
hand-designed physical coordinate system is used.

### Persistent state / MTSK

The recurrent state is a fixed-size tensor of learned linear regressors. It
is updated continuously from observations. Verified experience is assimilated
through a separate post-outcome write path.

### CDL

CDL is a learned scorer over persistent expert views. The forward pass uses a
hard top-k subset with a straight-through training relaxation.

### CASM

CASM currently consists of a learned operator bank, a learned selector, and
adaptive halting. It is intentionally **not yet promoted as a structural
graph-execution result**. That is a known implementation limitation to be
addressed before any final PLM claim.

### OSM

OSM is a learned action-conditioned latent transition predictor plus learned
Hamiltonian energy. The action planner evaluates candidate actions through
the learned OSM rather than using a fixed physical transition rule.

### Verification and repair

The verifier sees post-action evidence. Repair proposes a new action from
post-action evidence when the verifier rejects the transition. Persistent
state commits are hard-gated by verifier output during inference.

### Plasticity

MTSK provides fast and slow persistent adaptation through its expert update
rates. Parameter learning remains an offline training mechanism; it is not
claimed to be the same thing as the online state dynamics.

## Privileged-information boundary

The only privileged world knowledge allowed in this lane is:

1. Hamilton canonical equations on a learned canonical latent.
2. Conservation of the learned Hamiltonian during passive intervals.

Forbidden model inputs include physical-state labels, entity IDs, fixed
operator semantics, oracle retrieval, target action, future reward, and test
outcomes.

## Scientific sequence

1. Verify architecture and leakage gates.
2. Run the integrated full-model capability test.
3. Repair any implementation or benchmark invalidity.
4. Only after the full model passes, freeze it.
5. Run preregistered ablations against that frozen integrated model.

A negative result means this integrated construction did not establish the
capability. It does not imply PLM is impossible.

A positive result licenses bounded evidence for this integrated model on the
registered benchmark. It does not automatically establish general
intelligence, general physical reasoning, real-world multimodal competence,
or the PLM scaling hypothesis.
