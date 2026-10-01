# Unified PLM Research Program

## Research position

TAC-OSM is now the experimental substrate for a unified PLM program. The architecture is a persistent computational system whose useful computation should scale with useful structure rather than accumulated structure.

The causal loop is:

S_t -> R_t -> C_t -> A_t -> O_t -> V_t -> S_(t+1) -> L_t

where:
- S_t is typed persistent state;
- R_t is relevance/state addressing (CDL);
- C_t is computation/operator selection (CASM);
- A_t is the external action;
- O_t is post-action outcome;
- V_t is verification/repair;
- L_t is delayed learning, consolidation and capacity lifecycle.

## Architectural fusion

### Persistent typed state

State is partitioned into WORLD, TASK, EXPERIENCE, COMPUTATIONAL, POLICY, EVIDENCE, PREDICTION, HYPOTHESIS and UNVERIFIED records.

Authority is separate from existence. Records have lifecycle status and provenance. Failed or uncertain outcomes must not silently become authoritative state.

### CDL

CDL answers: what information matters for this task?

The existing CDL teacher remains a supervision boundary because its per-candidate language-model scoring is not a sublinear runtime path. The previous teacher-to-cheap-router transfer was inconclusive and is not promoted.

The unified runtime therefore exposes a lightweight relevance layer plus a hierarchical structural address layer. Raw bucket population is reported separately from the admission cap.

### CASM

CASM answers: what computation should be executed?

The operator pool represents reusable computations. It supports operator identity, family preference, usage weight, bounded plans, adaptive halting, synthesis and pruning.

Operator lifecycle is a mechanism probe until added capacity earns held-out capability.

### Slow/fast plasticity

The Mini-AGI contribution is used as a falsifiable mechanism hypothesis: slow shared structure plus fast specialized adaptation.

The fused reference exposes separate slow and fast update rates. Fast adaptation is localized to specialist context; slow adaptation updates shared structural relevance.

This is a small interpretable analogue, not a reproduction or capability claim for the external model.

### Adaptive compute

CASM halts when plan confidence is sufficient or the declared depth budget is exhausted.

The endpoint is capability versus executed work.

### Verification and state stability

Verification is a state-transition filter. A verified outcome can create authoritative experience; failures remain uncommitted and may trigger bounded retries.

False acceptance is injectable as a stability falsification control.

## Bottleneck relocation

| Architecture | Principal bottleneck | Scaling variable |
|---|---|---|
| Transformer | context/KV attention processing | cached context/history |
| SSM | bounded recurrent state dynamics | sequence/history |
| TAC-OSM | addressing plus composition representation | persistent population M |
| Unified PLM | representation, addressing, computation selection and state stability | H, M, E |

For autoregressive cached attention, a single decode step is linear in cached context length; quadratic attention is the full-sequence attention pattern, not an assumption to apply to every cached decode step.

## Three scaling variables

H is accumulated history.

M is persistent memory population.

E is computational operator or specialist population.

The research target is not that H, M or E remain bounded. The target is that useful query-time computation does not automatically become proportional to all three.

Cost is decomposed as:

C_total =
C_representation +
C_address +
C_rerank +
C_operator_select +
C_exec +
C_verify +
C_write +
C_IO

## Four primary failure modes

1. Representation failure: the needed relation is not expressible in the learned feature map.
2. Addressing failure: the relevant state exists but is not admitted.
3. Computation-selection failure: relevant state is present but the wrong operator or depth is chosen.
4. Persistent-state stability failure: an incorrect transition is committed and contaminates future retrieval.

A fifth axis measures computational-capacity lifecycle: whether new capability can be localized to specialists rather than requiring global growth.

## Break-even addressing

An index is only computationally useful if its measured addressing and reranking work beats a strong scan baseline at equal capability.

Every addressing phase should report:
- raw bucket population;
- admitted population;
- candidates actually scored;
- index probes;
- admission recall;
- arithmetic proxy;
- wall-clock latency;
- memory and I/O where available.

An admission cap must never hide an oversized raw bucket.

## Anti-rediscovery rule

The unified phase consumes existing C5 evidence rather than re-selecting C5 winners. Fixed inputs include:
- dense rank scaling;
- explicit LSH recall/work measurements;
- persistent relational state results;
- reset/shuffle/corrupt controls;
- verifier-gated state writes;
- the representability gate;
- CDL teacher cost boundaries;
- the inconclusive cheap-router transition;
- the CASM execution contract and known failed learned-routing arms;
- exact solver/hash references as floors and controls.

A new experiment is justified only when it changes an architectural variable, combines previously isolated mechanisms, or measures a previously unmeasured interface.

## Scientific gates

Every learned phase maintains:
- anti-leakage;
- representability;
- calibration and held-out disjointness;
- exact/reference controls;
- provenance;
- separate routing and execution accounting;
- explicit negative controls.

Finite-range slopes are descriptive only.

Synthetic relational results are not semantic-language results.

Teacher agreement is not capability.

Lower candidate count is not lower total cost until addressing work is included.

## Research ladder

### P0: unified reference substrate

Implemented in src/tac_osm/plm_unified.py:
typed persistent state, authority-gated writes, hierarchy-aware addressing, cheap relevance, slow/fast plasticity, operator pool, adaptive CASM, verifier, bounded retry budget, lifecycle hooks and consolidation.

### P1: integrated C5 continuation

TACOSM-PLM-UNIFIED-PHASE-001 increases history and persistent population while holding task structure fixed. It reports representation, raw/indexed addressing, admission, execution, verification and bounded-state metrics.

This phase is a systems-fusion and bottleneck-localization phase, not a proof of the target scaling law.

### P2: verified state dynamics

Next: long-horizon state chains, delayed writes, contradiction and retraction, corruption dose response, contamination probes and verifier false-acceptance thresholds.

### P3: computation addressing

Hold state relevance fixed and scale operator population E. Measure operator selection, executed depth and compute.

### P4: continual specialization

Test fast specialist updates, slow shared updates, specialist synthesis/pruning and consolidation under repeated domain shifts.

### P5: strong baseline scaling

Only after P0-P4 gates pass should the unified system be compared with strong dense context, bounded-state recurrence and exact scan/hash floors.

## Long-term criterion

The target is:

H, M, E increase

while approximately preserving:
- task difficulty;
- relevant information;
- required computation;
- capability.

Desired empirical behavior is selective state addressing, selective operator addressing, bounded useful execution, stable verified state, and evidence-driven capacity growth.

This is the research target, not a current result.
