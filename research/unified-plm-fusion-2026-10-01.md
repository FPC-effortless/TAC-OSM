# Unified PLM Research Fusion — TAC-OSM × CDL × CASM × Mini-AGI

Date: 2026-10-01

## Purpose

This document preserves the unified research synthesis from the conversation: TAC-OSM, CDL, CASM, PLM, Mini-AGI, persistent state, selective addressing, reusable computation, adaptive capacity, plasticity, verification, repair, and consolidation are treated as components of one research architecture.

The central architectural hypothesis is:

> **PLM = a persistent computational substrate that separates slow shared structure from fast specialized computation, selectively retrieves relevant state, executes reusable operators with adaptive compute, verifies outcomes, and updates both persistent state and model capacity according to evidence.**

---

# 1. Unified architecture

~~~
                         CURRENT INPUT
              goal / observation / query / action result
                                  |
                                  v
                     +-------------------------+
                     |   SLOW SHARED SUBSTRATE |
                     | representation          |
                     | identity / roles        |
                     | structural schema       |
                     | routing priors          |
                     | verifier structure      |
                     | policy constraints      |
                     +------------+------------+
                                  |
                              query Q_t
                                  |
                                  v
                     +-------------------------+
                     |     CDL ADDRESSING      |
                     | coarse domain/expert gate|
                     | hierarchical index      |
                     | local rerank            |
                     | relevant state R_t      |
                     +------------+------------+
                                  |
                           selected state
                                  |
                                  v
                     +-------------------------+
                     |          CASM            |
                     | select reusable operator|
                     | select specialist       |
                     | choose execution order  |
                     | allocate compute        |
                     | adaptive halting        |
                     +------------+------------+
                                  |
                                  v
                     +-------------------------+
                     |        EXECUTION         |
                     | transition model         |
                     | tool computation         |
                     | planning/operator        |
                     | state transformation     |
                     +------------+------------+
                                  |
                                  v
                                ACTION
                                  |
                                  v
                               OUTCOME
                                  |
                                  v
                     +-------------------------+
                     |    VERIFIER / REPAIR     |
                     | evidence collection      |
                     | consistency checks       |
                     | outcome validation       |
                     | bounded re-execution     |
                     | contradiction detection  |
                     +------------+------------+
                                  |
                                  v
                     +-------------------------+
                     |  PERSISTENT STATE UPDATE |
                     | world / task / experience|
                     | computation / policy     |
                     | evidence / prediction    |
                     | hypothesis / unverified  |
                     | provenance / time / conf.|
                     +------------+------------+
                                  |
                                  v
                         CONSOLIDATION / LEARNING
                                  |
                   +--------------+--------------+
                   |                             |
              FAST PLASTICITY               SLOW PLASTICITY
              selected experts              shared substrate
              new operators                 routing structure
              local adaptation              verifier structure
                   |                             |
                   +--------------+--------------+
                                  |
                                S_(t+1)
~~~

Canonical loop:

\[
S_t \rightarrow R_t \rightarrow C_t \rightarrow A_t \rightarrow O_t \rightarrow V_t \rightarrow S_{t+1} \rightarrow L_t
\]

Mini-AGI supplies plasticity/capacity machinery; CDL supplies relevance; CASM supplies computation; TAC supplies persistent state and temporal continuity.

---

# 2. What each research line contributes

| Component | Core responsibility | Failure addressed |
|---|---|---|
| PLM persistent state | retain structured information across time | long-context dependence |
| TAC | state persistence and temporal continuity | forced re-reading / recurrent memory |
| CDL | decide what state is relevant | irrelevant-history computation |
| CASM | decide what computation to execute | monolithic computation |
| Mini-AGI slow trunk | preserve reusable shared structure | catastrophic forgetting |
| Mini-AGI expert pool | localize adaptation | destructive global updates |
| Dynamic expert growth | add computation when existing operators are inadequate | fixed-capacity bottleneck |
| Expert pruning/consolidation | remove unused capacity | unbounded model growth |
| Adaptive halting | allocate depth according to difficulty | fixed compute per input |
| Verifier | determine whether execution/outcome is trustworthy | persistent error accumulation |
| Repair | re-execute boundedly after failure | one-shot execution errors |
| Evidence/provenance | distinguish fact from hypothesis and outcome | state contamination |
| Learning controller | decide when/how fast substrate changes | unstable continual learning |

None of these should secretly become another name for the same module.

---

# 3. Mini-AGI: the important contribution is plasticity hierarchy

The important Mini-AGI result is the separation of slow shared structure from fast specialized computation.

\[
\theta =
\underbrace{\theta_{\text{slow}}}_{\text{shared structure}}
+
\underbrace{\theta_{\text{fast}}}_{\text{specialized operators}}
\]

with:

\[
\eta_{\text{slow}} \ll \eta_{\text{fast}}
\]

The fast modules should be interpreted in PLM not merely as language-model experts, but as learned computational operators.

Candidate specialist pool:

- relational lookup
- arithmetic transition
- temporal comparison
- causal update
- planning operator
- code transformation
- anomaly detection

CDL chooses relevant state.

CASM chooses which operator/expert should manipulate it.

The extracted hypothesis is that shared components can be made substantially less plastic than specialized components, improving continual retention. This should be tested inside PLM rather than assumed to transfer unchanged.

---

# 4. Persistent state is typed computational memory

Persistent state should not be one homogeneous vector store.

Represent a memory item as:

\[
m_i =
(
z_i,\,
k_i,\,
\tau_i,\,
type_i,\,
status_i,\,
evidence_i,\,
provenance_i,\,
operator_i,\,
outcome_i,\,
confidence_i
)
\]

State categories:

- WORLD
- TASK
- EXPERIENCE
- COMPUTATIONAL
- POLICY
- EVIDENCE
- PREDICTION
- HYPOTHESIS
- UNVERIFIED
- CONTRADICTION
- RETRACTED
- SUPERSEDED

A prediction is not a world fact. An unverified experience is not an authoritative transition. A verified outcome can become stronger state.

---

# 5. Bottleneck relocation

### Transformer

\[
x_{1:N}
\rightarrow
Attention
\rightarrow
representation
\]

Primary problem:

\[
\text{how to process increasingly large context efficiently?}
\]

### SSM

\[
h_t=f(h_{t-1},x_t)
\]

Primary problem:

\[
\text{how to compress history without losing future-useful information?}
\]

### TAC-OSM

\[
S_t
\rightarrow
\text{retrieve}
\rightarrow
R_t
\rightarrow
\text{execute}
\]

Primary problem:

\[
\text{how to address persistent state without scanning it all?}
\]

### Unified PLM

\[
\boxed{
\text{How do we retain, address, select, execute, verify, and learn over persistent structure while keeping each operation selective?}
}
\]

This is stronger than "long-term memory."

---

# 6. Routing cost remains unresolved

The intended decomposition is:

\[
C_{\text{total}}
=
C_{\text{address}}
+
C(|R|)
+
C(|A|)
+
C_{\text{verify}}
\]

However, current TAC-OSM evidence does not establish the desired asymptotic escape from population size.

The C5 measurements show dense P90 rank growth approaching linear over the measured range. Any measured sublinear routing exponent should therefore be treated as an operating-regime statistic, not as an established asymptotic theorem.

The key question is:

> Can a persistent index actually become computationally cheaper than a strong scan baseline at the population sizes relevant to long-lived agents?

---

# 7. Correct Transformer baseline

A KV-cached autoregressive Transformer does not perform a fresh quadratic attention operation over the whole prefix for every generated token.

For one new token, attention over the cached context is approximately linear in context length.

Therefore the correct baseline is not "Transformer always costs O(N²) per token."

Compare PLM against:

- dense KV-cache context access;
- strong sparse/selective attention where appropriate;
- full scan of persistent state;
- SSM-like bounded-state memory.

The question is whether the PLM index beats those baselines at equal capability.

---

# 8. Composition representation is an upstream bottleneck

The persistent relational experiments show:

\[
\text{persistent state works}
\rightarrow
\text{relation reconstruction works}
\rightarrow
\text{composition representation weak}
\rightarrow
\text{CDL selectivity degrades}
\rightarrow
\text{CASM receives too many candidates}
\]

Therefore the response is not simply "improve LSH."

The representation of compositional structure must first become sufficiently separable.

The representability gate remains foundational.

---

# 9. Representation gate

\[
\text{world relation}
\rightarrow
\text{representation}
\rightarrow
\text{addressing}
\rightarrow
\text{operator selection}
\rightarrow
\text{execution}
\]

If the representation cannot express the relevant relation, downstream routing and execution can appear broken even when those components are correct.

The unified gate is:

\[
\boxed{
\text{Representability}
\rightarrow
\text{Addressability}
\rightarrow
\text{Computability}
\rightarrow
\text{Verifiability}
}
\]

A failed representability gate must not be reported as a routing failure.

---

# 10. CDL should become hierarchical

Do not make CDL equivalent to:

\[
Q\rightarrow\text{score every memory item}\rightarrow top-K
\]

That recreates a scan.

Instead:

~~~
Q
|
+-- level 0: domain / namespace / task family
|
+-- level 1: persistent partition / expert family
|
+-- level 2: learned coarse address
|
+-- level 3: local metric search
|
+-- level 4: exact reranking
|
+-- retain K
~~~

Thus:

\[
M\rightarrow M_1\rightarrow M_2\rightarrow\cdots\rightarrow K
\]

with:

\[
K\ll M
\]

and the target:

\[
C_{\text{address}}(M)\ll C_{\text{scan}}(M)
\]

at a measured break-even population.

---

# 11. Break-even population M* is first-class

Let:

- \(s\) = cost to score one candidate;
- \(E\) = cost of executing one candidate;
- \(K\) = number of admitted candidates.

The index is useful only when:

\[
C_{\text{index}} < C_{\text{scan}}
\]

Therefore there is an empirical:

\[
M^*
\]

above which indexing becomes computationally worthwhile.

Report:

- MACs
- FLOPs
- wall-clock latency
- VRAM
- I/O
- retrieval recall
- reranking work
- admitted K
- end-to-end capability

Do not infer index superiority from candidate-count reduction alone.

---

# 12. Persistent memory and computational capacity are different scaling axes

Define:

\[
H=\text{history}
\]

\[
M=\text{persistent memory population}
\]

\[
E=\text{computational expert/operator population}
\]

Memory growth \(M\uparrow\) means more persistent experience/world state.

Computational growth \(E\uparrow\) means more reusable operators.

A persistent learning system should not solve every new problem by making the existing representation denser. It may instead add specialized computational capacity.

---

# 13. Dynamic expert creation becomes learning new computation

Lifecycle:

~~~
query
  ↓
CDL retrieves state
  ↓
CASM tries known operators
  ↓
execution fails / remains expensive
  ↓
verifier identifies structured failure
  ↓
Can existing operator be repaired?
     |
    yes → update specialist
     |
    no
     ↓
compose/create new operator
     ↓
register operator in computational state
     ↓
route future instances to it
~~~

This turns the expert pool into a library of learned computations.

---

# 14. CASM is the computational counterpart of CDL

\[
\boxed{CDL=\text{what information matters?}}
\]

\[
\boxed{CASM=\text{what computation matters?}}
\]

~~~
             QUERY / GOAL
                   |
             +-----+-----+
             |           |
             v           v
           CDL          CASM
             |           |
       select state   select operation
             |           |
             +-----+-----+
                   |
                   v
                EXECUTE
~~~

State selection and computation selection are distinct routing problems.

---

# 15. Adaptive halting belongs inside CASM

Use:

\[
h_{i+1}=F_{r_i}(h_i,R_t)
\]

where \(r_i\) is the selected operator.

Then:

\[
P(\text{halt}\mid S_t,R_t,C_i)
\]

determines whether another computation should run.

CASM controls:

\[
\boxed{
\text{operator identity}
+
\text{operator sequence}
+
\text{computation depth}
}
\]

---

# 16. Three learning timescales

### Fast

\[
\theta^{fast}_{t+1}
=
\theta^{fast}_t+\eta_f g_t
\]

### Medium

\[
S_{t+1}
=
\operatorname{Consolidate}(S_t,O_t,V_t)
\]

### Slow

\[
\theta^{slow}_{t+1}
=
\theta^{slow}_t+\eta_s g_t
\]

with:

\[
\eta_s\ll\eta_f
\]

Conceptually:

~~~
FAST
new experience
new operator
local adaptation
      ↓
MEDIUM
verified state
consolidation
operator lifecycle
      ↓
SLOW
shared representations
routing priors
structural abstractions
verifier structure
~~~

---

# 17. Verifier as a dynamical-system stability mechanism

The persistent loop is:

\[
S_t\rightarrow S_{t+1}
\]

A false write is therefore a state-transition error.

Define:

\[
V_t:
(S_t,C_t,A_t,O_t)
\rightarrow
\{\text{accept},\text{reject},\text{repair}\}
\]

and:

\[
W_t =
\begin{cases}
\text{authoritative write}, & V_t=\text{verified}\\
\text{quarantine}, & V_t=\text{uncertain}\\
\text{repair then re-evaluate}, & V_t=\text{failure}
\end{cases}
\]

The verifier is a state-transition filter and stability mechanism.

---

# 18. Error-state semantics

Use explicit state statuses:

- FACT
- OBSERVATION
- VERIFIED_OUTCOME
- EXPERIENCE
- PREDICTION
- HYPOTHESIS
- UNVERIFIED
- CONTRADICTION
- RETRACTED
- SUPERSEDED

Prevent:

~~~
incorrect transition
       ↓
persistent write
       ↓
future retrieval
       ↓
future computation
       ↓
new error
       ↓
persistent reinforcement
~~~

Instead:

~~~
incorrect transition
       ↓
verifier
       ↓
quarantine / repair
       ↓
no authoritative state mutation
~~~

---

# 19. Long-horizon credit assignment

The causal chain can be:

\[
R_t
\rightarrow
C_t
\rightarrow
A_t
\rightarrow
O_t
\rightarrow
V_t
\rightarrow
S_{t+1}
\rightarrow
R_{t+k}
\rightarrow
C_{t+k}
\rightarrow
A_{t+k}
\]

Potential sources of future failure:

- representation;
- retrieval;
- operator choice;
- execution;
- action;
- observation;
- verification;
- memory write;
- consolidation;
- plasticity.

The verifier/evidence graph should support both quality control and credit-assignment localization.

---

# 20. Exact baselines remain mandatory

Benchmark:

1. exact hash/semantic solver;
2. dense scan;
3. Transformer/KV baseline;
4. SSM baseline;
5. learned dense CDL;
6. learned sparse CDL;
7. CDL + CASM;
8. CDL + CASM + persistent state;
9. full PLM.

The exact solver is the algorithmic floor.

The full scan is the retrieval baseline.

Transformer/KV measures dense context access.

SSM measures bounded-state memory.

PLM must demonstrate value relative to all of them.

---

# 21. Unified cost function

\[
C_{\text{total}}
=
C_{\text{representation}}
+
C_{\text{address}}
+
C_{\text{rerank}}
+
C_{\text{operator-selection}}
+
C_{\text{exec}}
+
C_{\text{verify}}
+
C_{\text{write}}
+
C_{\text{I/O}}
\]

with:

\[
C_{\text{exec}}
=
f(|R|,K_C,D)
\]

where:

- \(|R|\) = retrieved state;
- \(K_C\) = selected operators;
- \(D\) = adaptive execution depth.

The real target is:

\[
\boxed{
C_{\text{useful}}
\approx
f(\text{relevant information},\text{required computation})
}
\]

rather than simply constant execution cost.

---

# 22. Three-axis scaling law

Analyze:

\[
C(H,M,E)
\]

where:

- \(H\) = history;
- \(M\) = persistent memory;
- \(E\) = computational expert/operator population.

Desired:

\[
H\uparrow
\Rightarrow
C_{\text{exec}}\not\propto H
\]

\[
M\uparrow
\Rightarrow
C_{\text{address}}\ll O(M)
\]

\[
E\uparrow
\Rightarrow
C_{\text{operator-selection}}\ll O(E)
\]

This is a stronger target than one-dimensional history scaling.

---

# 23. Two indexes

~~~
                       QUERY
                         |
              +----------+----------+
              |                     |
              v                     v
       STATE ADDRESSING      COMPUTATION ADDRESSING
            CDL                     CASM
              |                       |
          state index              operator index
              |                       |
              v                       v
          R_t states              C_t operators
              |                       |
              +----------+------------+
                         |
                         v
                      EXECUTE
~~~

One index answers:

> Which state matters?

The other answers:

> Which computation matters?

---

# 24. Dynamic capacity must be evidence-driven

A new specialist/operator should require:

1. repeated unmet computation demand;
2. inability of existing operators to solve within budget;
3. representability gate passes;
4. verifier confirms improvement;
5. held-out generalization confirms improvement;
6. promotion only after evidence.

This prevents uncontrolled expert proliferation.

Unused specialists can be pruned/consolidated after an explicit lifecycle policy.

---

# 25. State and expert consolidation should interact

A successful specialist can become a reusable abstraction:

\[
\text{experience}
\rightarrow
\text{specialization}
\rightarrow
\text{verification}
\rightarrow
\text{abstraction}
\rightarrow
\text{reusable computation}
\]

Repeated specialist success should eventually influence the slower substrate rather than remain permanently isolated experience.

---

# 26. Transformer / SSM / PLM bottleneck map

| Architecture | Accumulation | Selection | Computation | Persistent learning | Primary bottleneck |
|---|---|---|---|---|---|
| Transformer | context/KV | limited or attention-based | dense | weak/default | context processing |
| SSM | recurrent state | implicit gates | recurrent | weak/default | state compression |
| TAC-OSM | persistent structured state | CDL | CASM | explicit but immature | addressing + representation |
| Mini-AGI | recurrent latent + expert pool | MoE | recurrent expert computation | explicit | plasticity/capacity |
| Unified PLM | typed persistent state | hierarchical CDL | CASM + specialists + halting | fast/medium/slow | addressability + composition + stable learning |

---

# 27. Definitive fixed-difficulty scaling experiment

Use:

\[
H\uparrow,\quad M\uparrow,\quad E\uparrow
\]

while holding approximately fixed:

- task difficulty;
- relevant information;
- required computation.

Compare:

### Transformer/dense
Per-token context access grows with cached context.

### SSM
Per-step computation remains bounded, but test associative recall as information density exceeds bounded-state capacity.

### Scan baseline
\[
C_{\text{scan}}\sim M
\]

### PLM
Desired:

\[
C_{\text{address}}\ll M
\]

\[
C_{\text{exec}}\approx\text{constant}
\]

\[
K\approx\text{bounded}
\]

\[
\text{capability}\approx\text{constant}
\]

The strongest result would be stable capability as \(H,M,E\) grow while useful computation remains selective and the complete system remains below strong scan/dense baselines after a measured break-even point.

---

# 28. Unified benchmark ladder

## Phase 0 — Baselines and gates

Before learned experiments:

- exact semantic/hash solver;
- dense scan;
- Transformer/KV baseline;
- SSM baseline;
- representability gate;
- leakage gate;
- provenance gate;
- exact generated-answer evaluation.

No capability claim survives a failed gate.

## Phase 1 — Representation

Compare:

- ordinary vector;
- factorized/product-key representation;
- analytic relational basis;
- structured interaction representation.

## Phase 2 — Addressing

Measure:

\[
M=64,\ldots,8192
\]

and extend beyond this where hardware permits.

Report:

- dense cost;
- indexed cost;
- reranking cost;
- recall@K;
- P90 rank;
- K90;
- wall-clock;
- admitted K;
- end-to-end capability;
- break-even \(M^*\).

## Phase 3 — State persistence

Use:

\[
H=64,256,1024,4096,16384
\]

with relevant information and computation fixed while distractor history increases.

Controls:

- reset;
- shuffle;
- corrupt;
- persistent;
- verified persistence;
- unverified persistence;
- quarantined persistence.

## Phase 4 — CASM

Fix retrieval quality.

Vary:

\[
K_C=1,2,4,8,\ldots
\]

and adaptive depth.

Measure capability versus executed compute.

## Phase 5 — Plasticity

Compare:

- global fast update;
- global slow update;
- expert-only update;
- slow substrate + fast specialists.

## Phase 6 — Dynamic capacity

Allow:

\[
E\rightarrow E+1
\]

when existing operators cannot solve a recurring verified workload.

Measure new capability versus parameter count, resident VRAM, routing cost, and total latency.

## Phase 7 — Full loop

Evaluate:

\[
S_t
\rightarrow
CDL
\rightarrow
CASM
\rightarrow
A_t
\rightarrow
O_t
\rightarrow
V_t
\rightarrow
S_{t+1}
\rightarrow
L_t
\]

with no oracle retrieval or oracle operator selection.

Only here should the full PLM claim be evaluated.

---

# 29. Hypothesis dependency graph

~~~
H0  The task is valid.
 |
 v
H1  The required relation is representable.
 |
 v
H2  Persistent state can retain it.
 |
 v
H3  CDL can address it.
 |
 v
H4  Addressing scales better than brute-force search.
 |
 v
H5  CASM selects the right reusable computation.
 |
 v
H6  Adaptive compute preserves capability at lower execution cost.
 |
 v
H7  Verification prevents persistent state contamination.
 |
 v
H8  Fast specialists learn without damaging shared substrate.
 |
 v
H9  New specialists can be created when existing capacity fails.
 |
 v
H10 Consolidation compresses experience into reusable structure.
 |
 v
H11 Full system maintains capability as H/M/E grow.
~~~

A failure at H3 must not be called an H4 failure.

A failure at H1 must not be called a CDL failure.

A failure at H7 must not be called a retrieval failure.

---

# 30. Unified architecture definition

Stop treating TAC, CDL, CASM, Mini-AGI, and PLM as competing architectures. They are subsystems of one architecture:

\[
\boxed{
PLM
=
\text{Persistent State}
+
\text{Selective Addressing}
+
\text{Selective Computation}
+
\text{Adaptive Capacity}
+
\text{Verification}
+
\text{Continual Learning}
}
\]

More formally:

\[
\boxed{
PLM_t =
(S_t,\theta_s,\Theta_f,\mathcal{I}_S,\mathcal{I}_C,V)
}
\]

where:

- \(S_t\) = persistent typed state;
- \(\theta_s\) = slow shared substrate;
- \(\Theta_f\) = fast specialist/operator pool;
- \(\mathcal{I}_S\) = state-addressing index;
- \(\mathcal{I}_C\) = computation/operator index;
- \(V\) = verification/repair mechanism.

Transition:

\[
\boxed{
(S_t,\theta_s,\Theta_f)
\xrightarrow{CDL}
R_t
\xrightarrow{CASM}
C_t
\xrightarrow{A_t}
O_t
\xrightarrow{V_t}
S_{t+1}
}
\]

Learning:

\[
\boxed{
(\theta_s,\Theta_f,S_{t+1})
\xrightarrow{L_t}
(\theta_s',\Theta_f',S_{t+1}')
}
\]

---

# 31. Competitive unit

The architecture's competitive unit is:

\[
\boxed{
\text{Representation}
\rightarrow
\text{Selective Computation}
\rightarrow
\text{Action}
\rightarrow
\text{Outcome}
\rightarrow
\text{Evaluation}
\rightarrow
\text{Improvement}
}
\]

Mapping:

### Representation
Slow shared substrate + typed persistent state.

### Selective computation
CDL chooses state.
CASM chooses computation.
Expert pool supplies specialized operators.
Halting chooses how much computation.

### Action
Policy/harness converts computed structure into an action.

### Outcome
Environment returns evidence.

### Evaluation
Verifier decides whether the transition worked.

### Improvement
Fast specialist update + persistent-state consolidation + slow substrate update.

This is the core PLM identity.

---

# 32. Deepest unifying principle

TAC contributes:

> Do not compute over all history; compute over relevant history.

Mini-AGI contributes:

> Do not update all parameters; update the specialized capacity responsible for new information.

CASM contributes:

> Do not execute all available computations; execute the computation required by selected state.

Verifier contributes:

> Do not allow every observed transition to become authoritative state.

Dynamic growth contributes:

> Do not force every new capability into the existing computational basis.

Consolidation contributes:

> Do not retain every experience as an equally expensive primitive.

Together:

\[
\boxed{
\textbf{Intelligence should scale with useful structure, not accumulated structure.}
}
\]

This is the unifying research hypothesis.

---

# 33. Scientific status and caveats

The full-system PLM claim remains a hypothesis.

Current TAC-OSM evidence establishes useful primitives and exposes bottlenecks, but does not yet demonstrate the desired asymptotic scaling law.

Important caveats:

- current C5 evidence does not establish sublinear asymptotic addressing;
- composition representation is an upstream bottleneck;
- exact hash/semantic baselines can dominate synthetic tasks;
- routing cost must be compared against a strong scan baseline;
- wall-clock and I/O matter in addition to FLOPs/MACs;
- persistent-state stability is less mature experimentally than addressing;
- Mini-AGI's plasticity result is a compatible hypothesis source, not proof that the same mechanism transfers to PLM;
- dynamic expert growth must be evaluated for useful capability per added capacity;
- the full loop has not yet established long-horizon verified continual learning.

Therefore the next implementation should build on validated findings rather than repeatedly rediscovering failed or already-settled components.

---

# 34. Final research framing

The research is no longer:

**TAC-OSM vs Transformer vs SSM vs Mini-AGI.**

It is:

\[
\boxed{
\textbf{PLM}
=
\underbrace{\text{persistent structured state}}_{\text{TAC}}
+
\underbrace{\text{relevance addressing}}_{\text{CDL}}
+
\underbrace{\text{operator selection}}_{\text{CASM}}
+
\underbrace{\text{slow/fast plasticity}}_{\text{Mini-AGI}}
+
\underbrace{\text{dynamic computational capacity}}_{\text{expert lifecycle}}
+
\underbrace{\text{adaptive execution}}_{\text{halting}}
+
\underbrace{\text{verified state transition}}_{\text{verifier/repair}}
}
\]

The scientific objective is to establish whether this combination can produce:

\[
H,M,E\uparrow
\]

while:

\[
\text{capability}\approx\text{stable}
\]

and:

\[
C_{\text{useful}}
\approx
f(\text{relevant information},\text{required computation})
\]

rather than:

\[
C_{\text{useful}}\propto
\text{all accumulated information and capacity}.
\]

That is the test that determines whether PLM represents a genuine architectural shift or merely relocates the conventional sequence-model bottleneck.
