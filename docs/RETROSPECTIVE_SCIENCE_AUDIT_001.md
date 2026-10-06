# Retrospective Science Audit 001

**Scope:** all TAC-OSM research lanes and claimed evidence present on
master `66e7036510d4f92697401ca5a36018a175326c7d`, plus the active research
portfolio visible through 2026-10-06.

**Purpose:** apply the repository-wide research contract retrospectively,
without rewriting historical artifacts or pretending that a later gate existed
when an older run was executed.

## 1. Disposition rule

This audit separates four questions that were previously mixed:

1. **Did the historical runner execute?**
2. **Was the measurement instrument valid?**
3. **What scientific statement, if any, does the surviving evidence support?**
4. **Is the historical result eligible for use as a current confirmatory result?**

A historical result may therefore be scientifically useful while remaining
**LEGACY-BOUND** for current work. Conversely, a run can have passed hundreds
of tests and still be **INVALIDATED** when an audit finds benchmark leakage,
an unmeasured comparison, privileged supervision, or an incorrect metric.

The audit does not delete, overwrite, or recompute historical artifacts. It
changes only the authoritative interpretation layer and records the reason.

Disposition vocabulary used here:

- **LEGACY-AUDITED:** historical result survives the retrospective audit within
  its stated scope, but was produced before the universal gate and therefore
  is not a new current-gate confirmation.
- **LEGACY-BOUND:** evidence survives only with explicit exclusions that must
  travel with it.
- **INVALIDATED:** the measurement cannot support the registered scientific
  question because of an implementation, benchmark, leakage, provenance, or
  statistical defect.
- **SUPERSEDED:** a later corrected experiment replaces the scientific role of
  the earlier result.
- **UNVERIFIED:** the repository contains registration/design or an asserted
  result but not enough independently checkable evidence to promote it.
- **CURRENT-GATE-ELIGIBLE:** reserved for future runs that pass the universal
  G0-G10/P0-P7 process. No historical result is promoted to this state by
  this document alone.

## 2. Executive portfolio finding

The portfolio is substantially stronger than a simple pass/fail reading would
suggest, but several apparently positive lines were too broad.

The surviving evidence separates into:

- **valid bounded mechanisms:** causal synthetic temporal persistence (C1),
  small-population outcome-trained relational routing (C2), semantic state
  addressing at M=8 and bounded M-scaling (REP-006/007), bounded exact/approximate
  retrieval controls, and bounded execution-subset accounting;
- **valid negative/localization evidence:** REP-001, learned-state-index
  failures, F2/F3 large-H interventions, cosine-selective proposal failure,
  E2E-003, and the current G-CASM bridge failures;
- **invalid/superseded evidence:** the original E2E-001/E2E-002 benchmark,
  C5-001/C5-002 capability interpretations, 016C/016C-R1, the original
  product-key claims affected by unequal training/target exposure, and
  privileged AXON/StructMeans/online-learning interpretations identified in
  the earlier audit;
- **registered but unresolved lanes:** C5-003, VRS-001/REPRESENTATION-
  REFINEMENT-002, active-evidence policies, 017/017A/018/018A, E2E-004, the
  composition-budget lane, and the bacterial MTSK lane.

The broad C5 thesis therefore remains **unestablished**. The portfolio does,
however, contain narrower evidence that the architecture can create a
retrieval boundary and execute a retained subset in synthetic controls.

## 3. Retrospective gate findings

### G0 — evidence necessity

Historical experiments often arose as a continuation of a diagnosed blocker,
which is good practice. The recurrent failure was **lane continuity**:
stacked architecture ideas could make the previous blocker disappear from the
active plan without being scientifically closed.

**Correction:** every lane now has an explicit status, parent/successor,
blocker, and next action in `docs/RESEARCH_LANE_REGISTRY.md`. Priority changes
do not close lanes.

### G1 — contract and provenance

Later experiments have substantially stronger machine-readable contracts.
Older REP and early C5 experiments used registration/prose rather than the
current contract envelope.

**Disposition:** historical results are not automatically void for that
reason; they are LEGACY-BOUND unless a more specific defect is found.
Registration quality itself is not evidence of the hypothesis.

### G2 — security and supply chain

The current repository now pins third-party research checkouts, restricts CI
permissions, separates setup from measurement, and preserves artifact
provenance. The retrospective audit found no basis to invent a historical
security violation where the record does not establish one.

**Correction:** future confirmation must use the universal G2 contract even
when the mechanism itself is unchanged.

### G3 — benchmark integrity

This is the most important retrospective gate.

Confirmed historical defects include:

- relevance-circuit operands were once wired tautologically;
- exact Boolean semantics were once unreachable through the learned parameter
  path;
- gold bookkeeping was stale after candidate shuffling;
- verifier traces contained padded nodes;
- verifier tolerance did not match the output range;
- replay reference selection preferred query bits instead of the addressed
  state vector;
- a multimodal E2E generator ignored the q2 entity argument;
- several later C5 studies used repeated or insufficiently independent
  population identities;
- one early product-key comparison did not have matched training/target-code
  exposure.

These are not “small implementation details.” Any affected headline result
is invalid until a corrected run establishes the registered quantity.

### G4 — leakage and supervision

Two distinct historical leakage classes are now explicit:

**Interface leakage:** target identity, clean target descriptors, state
addresses, or exact answer-derived metadata reached a router that was claimed
to operate without that information.

**Supervision leakage:** training labels were logically equivalent to the
answer, even where the label was not literally stored in the runtime input.

The audit specifically reclassifies the AXON/StructMeans and PLM online-learning
claims where exact correct-record supervision or operator-family metadata was
available. Those experiments remain useful as demonstrations of a controlled
training interface, but not as evidence for autonomous discovery.

### G5 — representability

The §34 lesson is confirmed as portfolio-wide infrastructure.

The decisive correction was the move from a per-episode ideal vector to a
**single shared vector across all registered families and episodes**. The
earlier failure showed why an oracle arm cannot establish learned-map
representability.

Any old negative learned result without this gate is not evidence against the
learner; it is **UNVERIFIED/LEGACY-BOUND** until the hypothesis class is shown
to contain the target relation.

### G6 — model-state integrity

The HS-001 zero-weight evaluation is a confirmed historical
**MODEL_STATE_INVALID** run. It produced plausible uniform routing numbers
that were initially interpreted scientifically.

The current parameter-hash/trained-state gate is therefore mandatory. A
historical result that cannot identify the evaluated checkpoint is not
promoted by this audit.

### G7 — degeneracy/discrimination

The C5 series demonstrated that aggregate accuracy can conceal a bridge that
cannot separate the two candidates relevant to a query. C5-002's 0.84375
absolute output accuracy versus 0.2305 pair separation is the canonical
example.

**Correction:** capability bridges must be gated on the discrimination the
downstream decision actually needs, not on a proxy aggregate score.

### G8 — oracle/control validity

Oracle arms are retained as ceilings and diagnostic controls, never as proof
that a learned arm is causal.

The audit preserves valid oracle results, but strips all interpretations that
used oracle success to excuse an unverified learned representation.

### G9 — baseline/ablation integrity

The strongest historical C5 frontier result used a state-distinct executor
and exact pooled count conservation, but the downstream executor was still
descriptor-equality based rather than actual CASM execution. Its correct
disposition is therefore a **bounded synthetic state-admission/counting
mechanism result**, not confirmation of the full C5 execution claim.

Likewise, the REP and learned-state sequences generally test named
interventions well, but they do not constitute a universal ablation of the
eventual unified PLM. A future integrated model must use the common ablation
harness and matched-capacity/compute controls.

### G10 — statistical readiness

Two recurring corrections are now binding:

- a “pooled P90” must mean a pooled-trial quantile; averaging per-seed P90s is
  a different statistic and cannot be renamed after measurement;
- frontier eligibility must use exact count-conserving aggregation and a
  predeclared capability floor; truncated/averaged success counts can
  invalidate an otherwise plausible frontier.

## 4. Authoritative retrospective dispositions by research lane

### A. Foundational runtime / temporal / selective controls

**PR #1–7, TACOSM-TEMPORAL-001, TACOSM-SELECTIVE-001,
C5-EXEC-001, C5-END-TO-END-001**

**Disposition: LEGACY-AUDITED / LEGACY-BOUND.**

The temporal result is a valid bounded synthetic mechanism:
write at t, read after k through 32 decision boundaries, carry 1.0,
reset near 0.01, corruption 0.0. It supports causal state persistence in the
specific deterministic vector-store control.

The selective runtime and execution-subset experiments validly demonstrate
that a bounded retained set can be passed to the downstream synthetic
executor. They do not demonstrate learned semantic retrieval, actual CASM
execution, hardware wall-clock speedup, or the L4 C5 thesis.

### B. REP-001–005

**Disposition: LEGACY-AUDITED / LEGACY-BOUND, with negative conclusions
preserved only at their registered scope.**

REP-001 is a valid negative/localization result after its script correction:
analytic/oracle controls reached 1.0, while learned routing remained near
no-learning/chance. This supports a failure of the tested learned router under
that protocol, not a failure of representability or of the architectural idea.

REP-002/003 establish the identifiability progression in their synthetic
designs. REP-004's exploration result is a training-dynamics intervention, not
a mechanism falsifier. REP-005 is a preregistered temporal continuation and
must not be promoted beyond what its own artifact establishes.

### C. REP-006–009 learned semantic state retrieval

**Disposition: LEGACY-AUDITED / LEGACY-BOUND.**

REP-006's 0.7047 learned top-1 versus 0.1859 control at M=8 is valid bounded
evidence for semantic state addressing in the tested synthetic task. Its
secondary rank diagnostic remains quarantined.

REP-007 shows the same mechanism remains above its control through M=32, but
absolute performance falls from 0.9922 at M=2 to 0.3344 at M=32. This is
bounded M-scaling, not scale invariance.

REP-008 is an exact-index ceiling/control, not a learned capability result.

REP-009 shows that approximate state pruning can improve this particular
learned scorer while retaining the target in the tested noisy task, but its
original decision rule did not treat the large positive shift as the primary
success branch. The raw effect is informative; the preregistered verdict must
not be retroactively rewritten.

### D. Learned-state diagnostic sequence #24–33

**Disposition: LEGACY-AUDITED / LEGACY-BOUND.**

The sequence is scientifically useful as diagnosis of a specific learner:

- binary index retention was weak;
- more training did not eliminate the boundary;
- negative coverage helped early and plateaued around eight negatives;
- multiple positive views were harmful in this configuration;
- latent width saturated over 8/16/32;
- cosine training did not change the result when evaluation was held fixed;
- cosine inference improved recall from 0.460 to 0.700 on the same weights;
- the learned binary proposal then lost too much target state retention for the
  selective funnel (0.504 proposal retention; 0.428 selective state recall;
  0.707 end-to-end versus 0.833 exhaustive).

These results support bounded diagnoses, not a global statement that learned
semantic state retrieval has failed.

### E. C5 product-key / frontier sequence #34–49

**Disposition: split.**

The sparse-funnel design and frontier registrations are valid experimental
instruments.

Frontier-001/002/003 are **INVALIDATED** for their respective downstream
executor, aggregation, and truncation defects; their artifacts remain
provenance records.

Frontier-004 (PR #38) survives as **LEGACY-BOUND bounded synthetic evidence**:
factor-size 16/beam 6 achieved a mean states-scored/M of 0.1508 with minimum
capability retention 0.8519 on M=128/256/512 under exact pooled counting.
However, the later scientific audit established that its downstream executor
was descriptor-equality based rather than actual CASM. It therefore does not
support the full C5 claim.

PR #39 is **SUPERSEDED** by the count-conserving frontier definition.

The independent robustness and three-/five-factor continuation lanes must not
be treated as a single cumulative “best frontier” unless each run has its own
artifact and preregistered eligibility decision. No outcome-selected winner is
inherited automatically.

### F. C5 integrated / fused / noisy sequence #50–65

**Disposition: bounded and heavily qualified.**

The fused CDL+CASM result in PR #56 is a useful systems integration diagnostic:
CDL admission was above the eight-candidate K=4 chance level and exact CASM
oracle routing was 1.0, but final learned top-1 was only 0.2467. The claim is
operational integration, not C5 success.

PR #57's clean-target state-address measurement is **LEGACY-BOUND** because
the CDL student read an exact target descriptor from persistent state. Its
gamma=0 therefore cannot be read as evidence for solving noisy recovery.

PR #58's noisy funnel removes that leak and is valid within its synthetic
scope, but PR #59 later shows the fitted gamma=0.776 was not an asymptotic
sublinear law.

PR #59 must be read as a finite workload diagnosis. Its rank curve is affected
by repeated 64-class populations, and its “pooled P90” was a mean of seed-wise
quantiles. The corrected full-range/local-slope numbers are diagnostic only,
not an asymptotic representation theorem.

PR #60 survives as **LEGACY-BOUND bounded persistence/composition evidence**.
It shows persistent relation reconstruction is lost under reset and that the
state-aware router uses information absent after reset, while the learned
composition/routing problem becomes difficult at M=1024. Its gamma=1.0650
finite-range fit is not a complexity claim.

PR #61 is a registered audit, not a result.

PR #62–64 require the following correction: exact correct-record supervision,
stored operator-family metadata, oracle dense outcome supervision, non-held-out
OOD slices, and M-dependent unnormalised gradients prevent their original
strong interpretations. These lanes are **LEGACY-BOUND** for infrastructure
and diagnosis, but their stronger learning/autonomy claims are **INVALIDATED**.

PR #65 is itself a prior audit record. Its corrections are adopted here, but
its queued audit run is not treated as a new scientific result.

### G. AXON / StructMeans / SECA

**Disposition: LEGACY-BOUND infrastructure; autonomous-discovery claims
INVALIDATED/NOT ESTABLISHED.**

The integrated loop, verifier feedback, StructMeans, AXON, REGM/SSA and SECA
work is valuable architecture/instrumentation. However:

- goal-conditioned routing can construct the exact action mask;
- StructMeans purity includes operator labels and realized after-state;
- verifier feedback can reveal the exact target after failure and label
  unexecuted candidates negative;
- online update paths were not demonstrated as generic learned-router updates.

Those facts prevent the stronger interpretation “autonomous operator discovery
from execution feedback.” That claim remains unestablished until a clean,
target-blind, outcome-driven training path is measured.

### H. VRS-001 / representation refinement

**Disposition: PREREGISTERED/UNVERIFIED.**

PR #74 is a clean registration and implementation boundary. No confirmatory
result was claimed. The external scored-proposal artifact requirement,
dynamic-validity criterion, and population-independent calibration remain part
of the gate.

### I. Integrated multimodal E2E

**E2E-001:** **INVALIDATED / VOID FOR BENCHMARK VALIDITY.**
The q2 generator ignored the entity argument and targeted the same entity as
q1. Its numeric 0.5200 result is retained only for provenance and cannot be
used for tuning or claim support.

**E2E-002:** **SUPERSEDED / NO SCIENTIFIC RESULT.**
It inherited the malformed generator and did not establish the registered
diagnosis.

**E2E-003:** **LEGACY-AUDITED / LEGACY-BOUND valid corrected negative.**
With distinct q1/q2 entities and held-out composition checks, explicit-write,
explicit-read and explicit-both remained at 0.5200 mean, with seed-bootstrap
95% interval [0.5020, 0.5385]. This is a bounded negative for that synthetic
implementation. It does not establish that persistent state, multimodal
fusion, or PLM is impossible.

**E2E-004:** **PREREGISTERED / UNVERIFIED** until its confirmatory artifact and
post-run independent analysis exist.

### J. G-CASM 012–015

G-CASM-012–014 are registration/implementation lanes unless a later artifact
is explicitly named.

G-CASM-015 is **LEGACY-AUDITED / LEGACY-BOUND**. The activation-trace probe
produced 5.8933915 information bits at M=512 versus 0.9999956 for the scalar
row, with the registered finite-domain information-per-work residual and
bootstrap interval. This supports a finite observation-channel distinction,
not downstream capability or asymptotic retrieval.

### K. G-CASM 016C / 016C-R1 / 016C-R2 / 016D

016C and 016C-R1 are **INVALIDATED**. In both, scalar target evidence had a
different Python type from candidate evidence, forcing the compatible bucket
to zero. R1 also failed to implement the promised runner-level regression.

016C-R2 is **CORRECTIVE / UNVERIFIED**. It introduces one typed extraction
path, a runner-level same-type/nonempty-bucket invariant, generator-backed
tests, and total work accounting. The trace arm's own finite result remains
useful; the scalar-vs-trace capability comparison must await a valid R2 run.

016D is **LEGACY-AUDITED / LEGACY-BOUND** as a bounded null comparison:
the one-step budget-utility change was tiny with a seed-bootstrap CI that
included zero, and exact verified success was unchanged. It does not establish
that Shannon selection is globally optimal or that multi-step planning wins.

### L. Active-evidence / PLM-018 / multimodal future lanes

PR #100 is infrastructure only.

PR #101/#103/#102/#106/018A are registrations or implementation lanes unless
their confirmatory artifacts are independently present. Teacher agreement must
remain diagnostic; teacher-generated labels must remain clearly identified as
supervision.

### M. Bacterial-inspired MTSK

The MTSK experiment remains **PREREGISTERED**. It must be tested as one new
mechanism surrounded by standard controls, not as a replacement research
program. The paired-history/current-input causal test is mandatory.

## 5. Claim-ledger corrections required by this audit

The authoritative claims ledger must satisfy these conditions:

1. **Only one C12 exists.** The earlier E2E-001 negative entry is superseded
   by the benchmark-invalid disposition. Historical provenance belongs in the
   audit, not in two competing claim records.
2. **C5 remains UNTESTED.** The surviving frontier evidence is narrower than
   the core claim because its downstream executor was synthetic and
   descriptor-equality based.
3. Bounded REP-006/007, C5 execution-subset, and E2E-003 results may be cited
   only with their synthetic scope.
4. Historical evidence is explicitly LEGACY-BOUND unless it has a current
   G0-G10/P0-P7 run.
5. No invalidated result may serve as a tuning baseline or claim comparator.

## 6. Repository-document corrections

The following stale statements are scientific drift, not cosmetic issues:

- `docs/ARCHITECTURE.md` says `PersistentState.write` is unimplemented even
  though the hardened runtime and temporal path exercise verified commits.
- `docs/ROADMAP.md` says the temporal measurement is preregistered/no result,
  while TACOSM-TEMPORAL-001 has a named result and C1 status.
- Current lane registry lists only major portfolio lanes and omits most
  historical REP/C5/G-CASM/E2E successors, making silent abandonment possible.

These contradictions are corrected in the same change set as this audit.

## 7. Final scientific state after retrospective correction

The strongest defensible statement today is:

> TAC-OSM has demonstrated several bounded synthetic mechanisms and useful
> measurement infrastructure: causal use of explicitly persisted state across
> enforced boundaries, outcome-trained relational routing at small population,
> semantic state addressing in a bounded synthetic task, structured evidence
> acquisition in a finite observation channel, and execution of an explicitly
> retained subset. It has not yet established that learned persistent semantic
> addressing, learned selective execution, verified adaptive state formation,
> multimodal PLM capability, autonomous operator discovery, or the broad C5
> economic-scaling thesis generalize or remain valid under the current unified
> research protocol.

That is the baseline every future lane must inherit. Novel ideas add
experiments; they do not erase these blockers.
