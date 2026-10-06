
# Bacterial Adaptive-State Research Program

Status: preregistered research direction; no confirmatory result.
Branch: research/bacterial-adaptive-state-integrity-001
Base: master @ 66e7036510d4f92697401ca5a36018a175326c7d

## 1. Purpose

This research track tests whether computational principles observed in bacterial
information processing can improve the TAC-OSM / PLM computational substrate
without turning the project into a biological simulation or an LLM attachment.

The biological literature is used as a source of computational hypotheses, not
as evidence that TAC-OSM is biologically equivalent to a bacterium.

The uploaded source describes bacterial history-dependent behavior involving
iron state, stress priming, regulatory-network persistence, and fast/slow
ribosomal memory. Those observations motivate this program; they do not by
themselves establish a machine-learning mechanism.

The strongest current external anchor is Kratz et al., PRX Life (15 May 2026),
which reports experimentally and theoretically that single E. coli cells
integrate environmental history, show memory from minutes to hours, and can be
described by scale-free / power-law memory arising from heterogeneous ribosome
populations. DOI: 10.1103/5zbg-8vll.

Other relevant prior mechanisms include E. coli chemotaxis temporal memory,
bistable/epigenetic regulatory states, CRISPR adaptive memory, and population
bet-hedging. These are retained as separate hypothesis families and are not
collapsed into one biological claim.

## 2. TAC-OSM complement

The present TAC-OSM loop is:

S_t -> R_t -> C_t -> A_t -> O_t -> V_t -> S_{t+1}

The repository already provides:
- a persistent-state interface;
- conditional relevance routing;
- a structural execution substrate;
- verifier / repair loop shape;
- active-evidence interfaces;
- machine-readable contracts;
- leakage and model-state integrity gates;
- shared-weight representability checks;
- benchmark-integrity and evidence-layer discipline.

The current gaps are explicit:
- verified persistent write across a temporal boundary remains under test;
- end-to-end capability from structured evidence is not yet established;
- efficient total computation remains unresolved;
- several C5 instruments were invalidated and corrected rather than promoted;
- active evidence exists as infrastructure, not as a demonstrated capability.

These gaps make bacterial-inspired adaptive state a complementary block rather
than a replacement architecture.

## 3. Single theoretical bottleneck

The research bottleneck for this track is:

Can a compact persistent state with multiple characteristic memory scales preserve
decision-relevant history and improve adaptation to changing environments at
fixed or bounded computation?

Only this block is novel in the first phase.

Name: Multi-Timescale Adaptive State Kernel (MTSK).

The first implementation should be deliberately narrow:
- fast state trace;
- medium state trace;
- slow state trace;
- a normalized mixture over logarithmically separated decay scales;
- explicit state-cost accounting;
- explicit write / update boundary.

A later scale-free realization may replace the finite mixture with a power-law or
fractional kernel, but it is a separate experimental variant, not silently
introduced into the first comparison.

The surrounding system remains battle-tested wherever possible:
- existing TAC-OSM interface contracts;
- existing router controls;
- existing CASM structural executor;
- existing verifier / repair interfaces;
- standard residual / normalization / optimizer primitives when a learned
  neural scaffold is required.

No custom general-purpose matrix multiplication, optimizer, benchmark evaluator,
or unrelated architectural rewrite is allowed in this phase.

## 4. Biological-to-computational translation

| Biological observation | Computational hypothesis | TAC-OSM surface |
|---|---|---|
| Chemotaxis compares current input with adapted past state | temporal differences can be more useful than raw history | MTSK fast/medium traces |
| Ribosome heterogeneity creates multiple adaptation speeds | a bank of persistence times can approximate useful long memory cheaply | MTSK |
| Regulatory-network feedback creates persistent states | memory can be a dynamical attractor / state configuration | persistent state write/update |
| Iron-linked state biases future swarming | an internal resource variable can become a predictive latent state | state-conditioned routing |
| Oxidative-stress priming | sublethal evidence can preconfigure later computation | active evidence + state |
| CRISPR stores prior threat fragments and later matches them | some memory should be explicit, addressable, and updateable | later memory-lifecycle track |
| Bistability / bet-hedging | multiple policies or states may improve robustness under uncertainty | later stochastic-state track |
| Biofilm / electrical / quorum interactions | global behavior can emerge from local state exchange | later graph-state track |

Only the first four rows are in scope for the first MTSK capability experiment.
The others are future branches.

## 5. Research questions

RQ1. Does MTSK improve adaptation when identical current observations require
different actions under different histories?

RQ2. Is any gain caused by actual persistent history rather than parameter count,
additional compute, current-input shortcuts, or benchmark artifacts?

RQ3. Does memory benefit depend on the number and distribution of timescales?

RQ4. Can the useful state be selective and compact rather than a replay of the
whole history?

RQ5. Does verified state formation matter: does the system improve when only
validated outcomes are allowed to change persistent state?

RQ6. Does MTSK retain capability under increasing irrelevant history while
avoiding an O(H) state-read or state-execution requirement?

## 6. Core benchmark principle

Do not use a benchmark that simply labels the hidden regime.

Construct tasks where:

same current observation + different history => different correct action.

The environment must contain several temporal regimes:
- stable;
- periodic;
- bursty;
- anti-persistent;
- regime-switching;
- unpredictable control.

The regime identity is hidden.

The minimum task is:

H_(0:t-1) -> S_t -> action

with the current observation held constant for paired histories while the
optimal action differs.

The benchmark must also contain matched controls where history is irrelevant,
so persistence cannot be rewarded merely for changing outputs.

## 7. Primary experimental sequence

### Phase 0 — evidence and necessity gate

Before any compute:
1. query the evidence register;
2. identify what is already established;
3. define the new L2/L3 claim;
4. confirm the run is not merely re-measuring a settled row;
5. freeze the protocol and contract.

### Phase 1 — instrument construction

Build only MTSK plus the minimum integration adapter.

No result from this phase is a capability claim.

### Phase 2 — pre-run gates

Every confirmatory run must pass the repository-wide governance and run-gate protocol in
docs/RESEARCH_GOVERNANCE.md and docs/RESEARCH_RUN_GATES_V2.md.

### Phase 3 — minimal causal experiment

Compare:
- no persistent state;
- single-timescale persistent state;
- MTSK multi-timescale state.

Keep parameter count and execution budget matched as closely as possible.

Primary endpoint:
exact class-balanced task success on the held-out temporal benchmark.

Secondary endpoints:
- adaptation regret;
- recovery time after a regime switch;
- history-conditioned action gap;
- persistent-state footprint;
- state-write frequency;
- state-read work;
- total execution work.

### Phase 4 — bottom-up ablation

Ablate MTSK one property at a time:
- state off;
- reset every step;
- shuffled state;
- wrong-history state;
- corrupted state at fixed corruption doses;
- one timescale;
- two timescales;
- logarithmic multi-timescale bank;
- frozen state;
- unverified write;
- verified-only write;
- no active evidence;
- active evidence with no persistence.

The causal question is whether the capability change follows the state mechanism
rather than merely the presence of another module.

### Phase 5 — scaling

Freeze the model and benchmark family.

Increase irrelevant history and test:
- capability;
- state size;
- routing work;
- execution work;
- verification work;
- total work.

Do not call a flat execution term efficient while an upstream acquisition or
routing scan grows linearly and is excluded from accounting.

### Phase 6 — cross-condition

Hold the mechanism fixed and vary:
- temporal regularity;
- noise;
- switch frequency;
- delay;
- regime count.

The decision rule must be fixed before these results are inspected.

### Phase 7 — cross-domain

Only after the synthetic temporal benchmark is passed:
- language continuation with history-dependent latent regimes;
- image sequences with hidden state;
- audio streams with hidden temporal regimes.

The first cross-domain goal is not state-of-the-art accuracy. It is to test
whether the same state mechanism transfers without adding a domain-specific
memory shortcut.

## 8. What would count as evidence

A positive MTSK result requires all of the following:
1. benchmark validity;
2. no leakage;
3. representability of the tested relation through the actual feature/state
   map;
4. non-degenerate outputs;
5. valid model-state provenance;
6. successful oracle/control checks;
7. preregistered comparison;
8. a material capability effect at matched compute/parameter budget;
9. the effect survives state-specific causal interventions;
10. the effect is reproducible across registered seeds.

A gain in router accuracy, training loss, teacher agreement, or information gain
alone is not a positive system result.

## 9. What would falsify the hypothesis

The track must permit these outcomes:
- MTSK gives no material benefit over a matched single-timescale state;
- any apparent gain disappears under matched parameter/compute controls;
- history-conditioned behavior is not improved;
- state ablations do not change capability;
- the effect occurs only in one task family;
- the slow state merely memorizes task identity;
- verified-write and unverified-write perform identically;
- MTSK improves a diagnostic but not exact task success;
- MTSK improves capability only by increasing total computation.

A negative result closes only the registered hypothesis or intervention. It does
not justify a broader claim that persistent multi-timescale memory is useless.

## 10. Representation and memory tests

The benchmark must include matched-history pairs:

A: history H_A, current x
B: history H_B, current x

with:
x_A == x_B
optimal_action_A != optimal_action_B.

The model passes the basic history-use test only if:

action_A != action_B

under otherwise identical current inputs.

Then perform intervention tests:
- replace S_A with S_B;
- reset S;
- shuffle S across episodes;
- corrupt S;
- block writes.

A load-bearing state mechanism should cause a preregistered change in behavior
under these interventions.

## 11. State utility and compression

Do not optimize for reconstructing the entire history.

Measure whether state preserves future decision utility:

U_future(S_t)

while limiting:
- state footprint;
- update cost;
- read cost;
- retained irrelevant information.

A candidate objective for later work is:

maximize U_future(S_t)
minus lambda * state_cost(S_t)
minus beta * irrelevant_history_information.

This is a research hypothesis, not a locked loss for Phase 1.

## 12. Claims boundary

This track may claim:
- a tested computational mechanism;
- a measured history-dependent capability effect;
- a measured state/cost tradeoff;
- cross-condition or cross-domain transfer only after those experiments.

It may not claim:
- bacterial equivalence;
- consciousness;
- biological plausibility beyond the cited abstraction;
- general AGI from a synthetic result;
- asymptotic scaling from finite sweeps;
- hardware speedup without hardware measurements.

## 13. Source hierarchy

Primary biological anchors:
1. Kratz et al., Multi-Timescale Adaptation and Emergent Learning in Single
   Bacterial Cells, PRX Life 4, 023015 (2026),
   DOI 10.1103/5zbg-8vll.
2. Bacterial chemotaxis information-processing literature.
3. Bacterial regulatory-network irreversibility / persistent-state literature.
4. CRISPR adaptive-memory literature.
5. Bacterial bistability / bet-hedging literature.

Uploaded transcript:
Bacteria Are Forming Memories With No Brain or Neurons.txt is a secondary
summary used to identify candidate mechanisms. Claims from it must be checked
against primary literature before entering a scientific claim ledger.

## 14. Immediate implementation boundary

The next architecture should therefore be:

environment
  -> sensory/query interface
  -> MTSK persistent-state update/read
  -> existing TAC-OSM relevance routing
  -> existing structural execution
  -> action
  -> existing verifier
  -> verified state commit

The novel block is MTSK.
Everything else is held as standard/scaffold/control unless an experiment
explicitly changes it.

## 15. Required provenance

Every measurement must identify:

repository
-> branch
-> commit
-> experiment_id
-> contract_fingerprint
-> benchmark_version
-> generator_hash
-> dependency_lock_hash
-> model_checkpoint_hash
-> configuration_hash
-> seed
-> artifact_hash
-> metric

Invalid runs remain immutable in the evidence archive and are never silently
deleted or promoted.
