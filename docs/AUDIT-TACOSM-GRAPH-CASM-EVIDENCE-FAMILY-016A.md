# G-CASM-016A — Evidence Family Comparison

Status: PREREGISTERED / IMPLEMENTATION

016A is the tightened successor to the earlier 016 design. The confirmatory question is fixed to one primary statistical test at M=512, avoiding post-hoc selection across M levels and evidence families.

## Scientific question

Does any predefined externally observable structured evidence channel carry more deterministic hypothesis information per environment acquisition work than one scalar output?

Registered external channels:
- scalar output;
- paired output using the fixed perturbation Delta=(1,0,0,0);
- four-output one-bit neighborhood.

The six-bit activation trace remains a non-deployable sandbox ceiling.

## Primary analysis

At M=512, each seed contributes 32 paired tasks per arm. Within each seed, task means are calculated first. The three external arm-minus-scalar differences are then formed, and the primary statistic is the maximum of their five-seed means.

The 4,000-resample seed bootstrap recomputes that maximum statistic inside every resample. M=32/64/128/256 are diagnostics and do not select the primary claim.

## Downstream utility

After action selection, realized target evidence forms the compatible bucket. A deterministic public ordering retains at most eight candidates. Exact CASM verification is then applied to those candidates against the verifier rows.

Probe selection never sees target identity, target index, realized target evidence, or verifier-only labels.

## Cost accounting

Environment costs are fixed before measurement:
- scalar = 1 execution;
- paired = 2;
- quad = 4;
- trace ceiling = 1 execution + 6 trace reads.

Candidate cache construction and selector prediction scans are separately reported and amortized across the registered 160 tasks per seed.

## Leakage and split integrity

Each seed builds a 256-program training set and a separate 512-program evaluation library, rejecting structure and truth-table collisions. The selector uses only public candidate predictions.

## Interpretation

A positive primary result supports only a finite-workload observation-efficiency claim for a predefined external channel. It does not establish learned autonomous probing, optimal sequential probing, asymptotic scaling, multimodal competence, or hardware speedup.

## Architectural implication

If an external structured channel wins, the next PLM layer is an EvidenceCompiler:

probe -> structured evidence -> sufficient evidence state -> selective execution -> verification.

If no external channel wins, the next step is not another dense router. It is a search over richer task/environment interfaces whose evidence is more discriminative per unit cost.

## Scientific disposition

016A is retained as a preregistered analytical-null design but is not a confirmatory experiment. For deterministic B-bit evidence, mutual information is bounded by B bits, while the registered environment acquisition cost is at least B scalar executions for the paired and quad arms. Therefore their information-per-environment-work cannot exceed the scalar one-bit ceiling of 1 bit per execution. Running a full confirmatory measurement would not test a non-dominated hypothesis; the informative next intervention is a richer single-execution structured channel, which is tested by 015/016C rather than 016A.
