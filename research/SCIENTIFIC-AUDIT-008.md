
# Scientific Audit 008 — TAC-OSM / PLM Research Integrity

Date: 2026-10-01

## Audit purpose

This audit reviews the active TAC-OSM research stack for wrong methodology, invalid or misleading metrics, implementation errors, leakage and provenance failures, benchmark shortcuts and confounds, vulnerable inference paths, and unsupported complexity/scaling claims.

Historical artifacts are preserved. A finding here changes what a measurement can support; it does not erase the measurement.

## Executive status

**Needs revision for claims beyond the synthetic mechanism scope.**

The unified PLM substrate is operational, but several metrics and benchmark constructions are not sufficient for stronger conclusions. The most consequential correction is PR #59 high-M dense rank scaling: the apparent near-linear raw-rank growth is largely explained by repeating the same 64 semantic classes as M grows while class-level Top-1 remains constant.

Established mechanism evidence remains:

1. temporal persistence/reset sensitivity is real in the registered synthetic relational benchmark;
2. verifier-gated writes behave as intended under injected false-accept stress;
3. state and operator addressing can be represented as separate cost surfaces;
4. current routing quality is insufficient for large persistent populations in the hardest registered relational loop;
5. no asymptotic useful-computation law has been established.

Not established:

1. sublinear routing or useful-computation scaling;
2. representation collapse as the cause of routing failure;
3. superiority of product-key, late-interaction, or outcome-field variants;
4. true OOD/compositional generalization;
5. general learned CASM outcome-field training;
6. compute or memory advantage over Transformer/KV/SSM baselines.

## Material findings

### A1 — PR #59 raw-rank scaling is population-replication confounded (P1)

The M=64..8192 benchmark uses a fixed 64-class codebook and repeats every class equally as M increases. The reported best-valid candidate rank therefore contains an explicit multiplicity factor:

raw candidate rank is approximately class rank multiplied by M/64.

The code itself computes rank as better_classes multiplied by copies, plus tie position, plus one.

The reported Top-1 stays 0.7625 at every M, while mean raw rank rises from 1.25625 at M=64 to 33.8 at M=8192. P90 rises 2,3,5,9,17,33,65,129. This is consistent with approximately stable class-level routing plus increasing duplicate-class multiplicity.

**Correction:** raw candidate rank must not be interpreted as representation degradation in the repeated-class benchmark. Report class-level metrics separately and use unique-candidate populations for population-scaling claims.

### A2 — PR #59 quantiles named pooled are actually means of per-seed quantiles (P1)

The aggregation code averages the five per-seed P90 values. That is not the pooled P90 of the 320 trial-level observations.

**Correction:** reserve pooled for quantiles computed from concatenated trial-level observations. Keep mean per-seed P90 as a separate descriptive statistic.

### A3 — PR #61 representation arms are not capacity matched (P1)

Control, product-key, and late-interaction routers have materially different trainable parameter counts and different per-update computation. Equal update counts therefore do not imply equal training compute.

**Correction:** match parameter count and training/inference compute, or treat results as exploratory with explicit capacity and compute accounting.

### A4 — PR #64 outcome-field arm is not yet a CASM outcome field (P1)

The 007 arm creates labels directly from the exact relation computed from persistent state and compares every candidate against that hidden target. This is dense oracle supervision, not a learned field of post-CASM action outcomes.

**Correction:** preserve the 007 artifact as an oracle dense-label diagnostic. A true outcome-field experiment must derive labels from executed candidate actions and post-action outcomes, with label-collection cost measured.

### A5 — PR #64 OOD slice is not held out (P1)

The 007 evaluator filters a pattern that was also present in unrestricted training.

**Correction:** define training support restrictions first and generate evaluation exclusively from the excluded structural family. Call it compositional holdout or distribution shift only when overlap is machine-checked.

### A6 — PR #64 outcome-field gradient changes with M (P1)

The candidate gradients are summed while the learning rate stays fixed. Thus gradient magnitude depends on candidate count. The observed collapse cannot be interpreted as evidence against dense outcome supervision.

**Correction:** normalize by candidate count or use an explicitly class-balanced objective and report optimizer state and candidate exposure.

### A7 — P6 is narrower than a persistent-state reasoning claim (P1)

The corrected P6 uses a noisy target descriptor in the public query and computes the router embedding from that query alone with an empty temporal state. Persistent state provides identifier mapping/filtering but is not the source of the routing signal.

P6 also uses independent bit corruption with probability 0.0625 rather than the exact-one-bit corruption used in the main C5 workload.

**Correction:** retain the measured result but scope it as C5 admission transfer to a typed-state-backed index. Add a state-addressing benchmark where the answer descriptor is absent from the public query.

### A8 — Cost proxies are not end-to-end compute (P1)

Current PLM/C5 routing costs are unitless query-side arithmetic proxies. They omit index build/update, memory footprint, I/O, cache behavior, and wall-clock runtime.

In historical LSH sweeps, routing operations can exceed M even where rerank/M is small.

**Correction:** keep arithmetic proxy metrics as mechanism diagnostics. Require wall-clock, peak memory, and amortized index maintenance before systems-performance claims.

## Additional vulnerabilities

### B1 — Oracle-assisted downstream execution

PersistentRelationCASMVerifier derives the exact target relation from state and constructs the executable relevance program from that reference. This is useful as a controlled executor/verifier mechanism but is not evidence of general learned CASM reasoning.

### B2 — Hard-negative distillation is oracle-backed

PR #61 negative harvesting depends on the same reference-driven executor/verifier. It is therefore a training-interface test, not a general verifier-discovery test.

### B3 — Index construction economics are omitted

Query-side indexed cost can fall while build/maintenance cost dominates an online mutable-memory workload.

### B4 — Operator identity is stored in records

The unified PLM benchmark stores operator_family on records and uses it to prioritize operator plans. This lowers the burden on learned operator discovery.

### B5 — Tail metrics have small samples

Five seeds and 64 trials per cell give coarse P90 estimates. Bootstrap intervals are descriptive and should not be treated as high-confidence tail guarantees.

### B6 — Many arms and budgets create a multiple-comparison surface

Current studies are exploratory/descriptive. A claim of superiority requires preregistered primary endpoints and comparison rules.

### B7 — Synthetic codebooks are algorithmically structured

The fixed 64-class binary code with one-bit corruption is a valid routing stress test but can be solved by simple geometry. It is not a semantic intelligence benchmark.

## Correct metric contract

Every future routing benchmark must report: Top-1; best-valid rank; semantic/class-level rank when classes repeat; Recall@K; pooled trial-level P90; K90 from independent calibration; LSH admission recall; raw bucket population; reranked candidates; hash/probe operations; embedding-score operations; CASM executions; verification operations; state writes/retractions; index build cost; amortized build cost; wall-clock; peak memory; and I/O where applicable.

No single proxy may stand in for end-to-end compute.

## Correct benchmark ladder

### G0 Integrity

Machine-check training, calibration, and evaluation disjointness; population identity; target IDs; descriptor hashes; seeds; code/commit SHA; and artifact digest.

### G1 Representability

Use deterministic witnesses over the actual student feature class. For signed Boolean inputs:

XOR = -ab
XNOR = ab
AND = a + b + ab - 1
OR = a + b - ab + 1

The ranking gate must directly verify all four input combinations. Randomized search is not admissible in a scientific pass/fail gate.

### G2 Geometry

Measure effective rank, covariance spectral concentration, pairwise cosine distribution, and norm concentration across M and representation arms. A single geometry snapshot cannot establish collapse.

### G3 Generalization

Training must exclude a declared structural family, evaluation must use the excluded family, and overlap must be machine-checked.

### G4 Learning interface

Match optimizer budget, normalize candidate aggregation, report candidate exposure, and identify the label source. Distinguish exact-target oracle labels from post-action environment outcomes.

### G5 Discrete admission

Use one fixed index family per seed/M and evaluate prefixes of it, or use independent calibration/evaluation families with uncertainty explicitly reported.

### G6 Execution

Measure fixed execution budgets and an adaptive budget using a rule fixed before held-out outcomes are inspected.

### G7 Verified experience/state editing

Only after G0-G6 pass should experience priors, editable state, consolidation, or learned state operators be promoted.

## Corrected scaling benchmark

Primary M levels: 64, 128, 256, 512, 1024, 2048, 4096, 8192.

For every M:

- target included exactly once;
- distractors unique;
- five seeds;
- 64 held-out queries per seed;
- fixed relation distribution;
- pooled trial-level rank statistics;
- Recall@1/4/8/16/32/64;
- query-side routing operations.

The historical repeated-class result should remain only as a multiplicity control.

## Corrected compositional holdout

Training generator excludes local operand pair (1,1) from every operand vector.

Holdout generator forces the first four positions to (1,1) and samples remaining positions from the declared distribution.

The generator records and hashes its support constraints.

## Corrected outcome-field experiment

The learner receives, for every candidate actually executed during training:

(query, candidate, action_outcome).

The target identity is not supplied as a direct teacher label.

Compare:

1. exhaustive target-index supervision;
2. all-candidate environment-outcome supervision;
3. sparse outcome supervision.

Outcome collection cost must be reported. The current 007 artifact remains historical and is interpreted as oracle dense-label supervision.

## Corrected P6 state-admission benchmark

Keep the corrected same-population identity.

Next benchmark changes only the public interface:

- public query carries an opaque state address plus noise;
- state contains the candidate-defining structure;
- target descriptor is not exposed directly in the query;
- state retrieval is part of the causal routing path.

## Promotion rule

Promote a representation or routing intervention only when representability passes, held-out structural generalization improves, capability is preserved at matched compute, and routing/admission cost is measured end-to-end.

ID-only improvements are not sufficient.

## Final audit status

### Verified or substantially supported

- temporal persistence and reset sensitivity in the registered synthetic relational task;
- verifier-gated write stability under the injected failure model;
- composable state and operator addressing in the unified substrate;
- separation of addressing, admission, execution, and verification costs;
- lack of evidence for an established asymptotic useful-computation law.

### Retained with limited scope

- PR #59 LSH admission sweep;
- PR #60 persistent relational loop;
- PR #62 unified PLM P1-P5 mechanisms;
- corrected P6 typed-state-backed admission transfer;
- PR #64 representation geometry and oracle dense-label diagnostics.

### Invalidated interpretations

- PR #59 raw high-M candidate rank as direct evidence of representation degradation;
- PR #59 per-seed P90 means described as pooled P90;
- PR #64 oracle dense-label arm described as a true CASM outcome field;
- PR #64 current conditional slice described as held-out OOD;
- any asymptotic sublinear claim based on the existing synthetic cost/rank proxies.

## Central unresolved question

Can a representation/admission mechanism preserve task capability as persistent state and operator populations grow under unique distractors, true structural holdouts, matched compute, and complete cost accounting?

Until that passes, verifier, repair, plasticity, and state-editing results should remain mechanism evidence rather than evidence that the complete PLM hypothesis is validated.

## Additional LSH metric vulnerabilities

### A9 — Repeated classes also confound LSH bucket/rerank scaling (P1)

When a class is replicated, identical candidate embeddings hash to the same LSH buckets. Bucket population and rerank counts therefore grow partly because the benchmark has more copies of the same vector, not because the router is handling more distinct semantic alternatives.

Admission recall can remain high because any copy can satisfy the target-class condition while rerank work rises from duplicate entries.

**Correction:** use unique candidate descriptors for population-scaling claims. For repeated-class diagnostic controls, report unique-class bucket counts separately from candidate-copy counts.

### A10 — PR #58/#59 p2 is a sampled-decoy collision diagnostic, not a population-wide collision rate (P2)

The p1/p2 estimator samples a small number of random non-target decoys per trial and aggregates hyperplane collisions. It is useful for diagnosing LSH behavior, but it does not characterize the full negative collision distribution.

**Correction:** report it explicitly as a sampled pairwise collision diagnostic. For operating-point selection, also report candidate bucket-size distribution and target admission over the complete evaluation population.

### A11 — K90 is also sensitive to duplicate copies (P1)

The registered K90 counts candidate entries, not distinct semantic classes. In a repeated-class population, a fixed class-level uncertainty can require an increasing number of candidate copies to reach the same class-level coverage.

**Correction:** report both candidate-level K90 and class-level K90 where repeated classes exist. Use unique candidates for the primary population-scaling experiment.

