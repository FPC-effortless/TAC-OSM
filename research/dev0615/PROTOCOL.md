# DEV-061.5 — identity binding and rollout robustness

REGISTERED BEFORE FITTING. This is a new bounded study, not a DEV-061 checkpoint
fine-tune. Preserve all previous weights, negative outcomes and failed CI. Follow
research/autoresearch/program.md (Karpathy-style register/commit/fit/audit/retain).

## Question and boundaries

Can an identity-separated learned CDL/neural-CASM policy generalize to unfamiliar
public identity vectors and meet a twenty-training-seed reliability target after
dataset aggregation, EMA and curriculum? No simulator, teacher, public-state
parser, repair rule, analytical planner or stopping override enters deployment.

The new architecture uses separate identity and content streams, shared content
weights, learned similarity strengths and type gates, relation-biased
cross-candidate attention, dense learned operator mixtures (not sparse symbolic
programs), learned action-conditioned prediction and a learned action/stop head.
A public episode entity manifest defines temporary slots. Hard nearest-cosine
binding groups raw histories by those slots; learned cosine strengths implement
subsequent soft CDL reads. This is an explicit entity-association prior, not
learned identity discovery. No persistent categorical embedding dictionary exists. Typed
38-vector inputs and 19-vector action descriptors remain programmed interfaces.
Dense mixtures and float64 replay intentionally defer sparse-runtime claims.

The identity treatment bundles content/identity separation and fresh per-episode
random-vector augmentation. Its control retains identity coordinates in content
encoders and trains on static public codes. The training treatment bundles
DAgger, EMA and horizon curriculum; this does not isolate each constituent.
All TAC cells share the same new relation-read/graph/operator/prediction controller.
Two latent transformations represent one candidate action, not a two-action
planning horizon. Runtime re-encodes the bounded retained history at each decision;
no C5 scaling or incremental-state compute advantage is claimed.
It is a new architecture, not a causal comparison against frozen DEV-061 scores.

## Matrix and fair controls

Twenty independent training initialization/data-collection seeds 92501–92520.
Seven arms: fixed_bc, fixed_robust, split_bc, split_robust,
no_casm_robust, raw_gru_robust, raw_mlp_robust. The first four are a 2x2
identity-bundle x training-bundle factorial. No-CASM is retrained and parameter
matched; the raw-event GRU and pooled-event MLP have no latest-state or parent-bit
feature parser. All receive the same public records through an identical frozen,
learned record-type admission network; all retained fields remain raw. A second frozen learned type gate maintains a
separate public relationship-key stream for every arm, including raw controls.
This stream copies relationship pointers, never health bits, into manifest metadata.
The raw controls are therefore raw-event content encoders with the same public
binding interface, not completely unstructured recurrent baselines. This common
noise front end is explicitly disclosed, not called relevance discovery. Retain
unfiltered public logs and charge every incoming observation. Capacity64 uses
FIFO; overflow remains a possible source of errors and is measured, not a
learned retention result. Episodic retention remains deferred.

All robust arms within a seed group contribute student trajectories to a shared
teacher-relabelled union. Each student sees its own visited states and the others',
so this is multi-policy DAgger. They share the same aggregated canonical data and
batch IDs, teachers, collection budgets and update counts; identity augmentation
is the intended treatment. BC arms keep the initial corpus. The union's collection
cost is recorded once per seed group, not secretly free or reacquired per arm.
The twenty seed groups are independent; arms within a group are paired.

Use separate seed92500 for two-LR pilots (0.001,0.003), same full recipe/budget.
Select each arm using validation only: fewer faults, higher minimum identity-family
completion, lower economic cost, lower validation loss, then smaller LR. Record
both curves/rates, all20seeds, EMA/raw weights and every unsuccessful fit. No
favorable seed selection. Parameters matched within5% of split_robust using an
integer-width count search independent of outcomes. Exposure is matched, not FLOPs.

## Data and identity contract

Initial canonical corpus4096 snapshots, validation512, seeds6151001/6151002.
New acyclic cascading-parent graphs, N2/4/6, initial H16/32/64; exact public status
and receipts. This clean prerequisite remains solvable by a public rule and is
NOT the hidden-state benchmark. Coupling can form chains, but chain complexity
alone does not invalidate a public-state solver. Keep its ceiling and costs.

Public categorical universe100, training labels0–79, held-out labels80–99 (20%).
A seeded 16-dimensional unit-vector codebook has pairwise cosine<0.75. Static
controls see fixed codes. Augmented learners see fresh independent unit vectors
per episode, with the same separation condition; identity/content shared weights
have no label lookup. Held-out label names are not semantic attributes and have
no causal role. Report static seen, static held-out, and freshly generated-vector
families separately on the same physical worlds and actual public inputs for all
models. Novel vectors test identity rebinding, not semantic entity discovery.
Near-collision and unconstrained-vector stress are separately qualified OOD cases.

## Training and finite resources

One frozen shared type-admission pretrain:300steps, batch256, Adam0.01, input only
four public source bits, signal/noise and relationship-type labels; pretrain both
admission and relation gates. No gate threshold tuning.
Models use float32 training, AdamW(wd1e-4), clip1, batch64,1024updates. First128
updates train factual/current-state and read targets; then full actor/dynamics
loss. Teacher policy is a distribution over all safe-progress actions, not an
arbitrary single optimal index. Loss: policy KL/CE +0.5 current-state BCE
+0.5 current-completion BCE +0.5 one-action fault BCE +0.25 one-action completion
BCE +0.1 normalized action-cost Huber +0.25 public relationship-binding CE +0.05 safe-vs-
unsafe policy margin hinge(margin0.5). Proper probability losses are unweighted
by economic fault cost. Do not call them perfectly calibrated.

Robust arms use EMA beta0.995 from initialization (deploy EMA, no raw/EMA selection).
At updates256/512/768, roll out all five robust students on128 shared training
worlds/round/seed, horizons2/6/(3N+6), relabel every visited public history with the
training-only teacher; append the union. Initial-corpus half-batch and union
half-batch after collection, without favorable-example deletion. First256 robust
updates sample snapshots with at most2 prior interventions; later use the full
corpus. BC samples the full initial corpus throughout. All data/inspection/repair
collection costs and examples are recorded. The admission-only cost is shared.

Two CPU workers, one Torch thread each. New study fitting cap180CPU minutes,
including pilots, collection, calibration and per-fit validation. Cap600CPU seconds/seed group, including finalization;
retain capped/failed groups. Incomplete20-seed matrix forbids acceptance/test opening.
No automatic extension. Engineering smoke uses disjoint namespaces and SMOKE_ONLY.
Freeze protocol, model and fitter before registered fitting; no test-based edits.

## Replay, calibration and tie handling

Train float32; deployment/evaluation uses float64 and one deterministic Torch
thread. Dense attention/operator mixtures avoid operator-selection ties; manifest binding
still uses hard cosine routing, with separated identities except collision stress.
A registered generic decoder selects the first supplied valid candidate within
1e-6 of the maximum score. This is a numeric tie convention, not a repair or stop
rule. No privileged semantic identifier controls ties. Log top2 gaps, epsilon-
boundary distances, fraction below1e-6, CPU/software versions and numeric drift.
Fit positive policy temperature from validation only over grid0.5/1/2/4 using
unweighted teacher NLL; apply the tie convention to unscaled scores, so temperature
cannot alter actions. Report stop/completion calibration and early/late stopping.

## Untouched evaluation and gates

Holdout seed6151003, clean N2/4/6 x H32/128 x32worlds, three identity families,
576worlds/model. Primary: >=18/20training seeds achieve >=95% completion in EACH
identity family; novel static and fresh-vector completion must each match or
exceed seen completion per accepted seed. Report every-cell fractions separately,
all20paired rows, physical repair, residual faults, budget exhaustion, stop errors,
mean/min/max, intervals, actions/cost, and regret to the public teacher/reference.
Primary action budget3N+6; report tighter2N+4 as a separate diagnostic, not a hidden
completion improvement over prior studies. The reference shares the budget.

Promotion additionally requires zero residual physical faults across primary
heldout evaluation, no higher cost than the public reference, and all observed
cross-hardware float64 action choices matching exactly. Prediction telemetry
comparison toleranceatol1e-8/rtol1e-8; never widen it after seeing discrepancies.
A passed functional target does not establish commercial value or superiority.
Compare paired20seed split_robust minus tuned GRU/MLP/no-CASM using paired-seed
bootstrap2000 and Holm-corrected sign flips for three cost comparisons. No general
CASM claim without reliability, cost advantage and an information-matched task
with registered headroom. DEV064-CASM-LONG remains the later novelty deadline.

Extended H512/1024 N6 x16worlds/identity, near-collision and unconstrained Gaussian
N6/H128x16worlds each are separate secondary strata. Preregister seeds6151004/5.
No training changes after these are opened. Predeclared exported candidate seed92501,
not best seed. Save all weights, initial/aggregated data, curves, trajectories,
source hashes, priors, failed comparisons, parameter/state bytes and actual CPU
latency at float32 and float64; DRAM traffic UNKNOWN without hardware counters.
Independent restore replays all policies; native Veritas audit covers the fixed
first seed of all seven arms, not falsely claimed as full native coverage.

## Beyond this prerequisite

DEV062 needs public action provenance, missing states, noisy sensors, paid exact
inspection and exact information-matched belief controls. Public-posterior KL
training is stageA; observation/action learning without posterior targets is
stageB. Analytic VOI/filter/planner are references or training targets only;
deployed decisions remain neural. Commercial and complex-task advantage remains
UNESTABLISHED. Transformer depth, BDH dynamics and SSM replacements require
separate arms after these gates; retention awaits genuine historical dependence.

## Pre-fit implementation clarification

The initial registration commit preceded implementation. Before any fit or smoke
run, the final implementation explicitly adds the public manifest/relationship-key
priors and replaces proposed latest-record attention supervision with relationship
binding supervision. These changes are design corrections, not outcome-driven
tuning. The original registration remains in Git history. Raw baselines receive
the same added metadata. No privileged target is passed to deployment.
