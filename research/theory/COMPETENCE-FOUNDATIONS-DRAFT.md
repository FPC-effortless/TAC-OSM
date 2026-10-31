# Competence foundations — working research note

Status: **proposal and elementary decision-theoretic reasoning**, 2026-10-31.
This is a framework for falsifiable studies, not a validated theory of intelligence.
TAC-OSM is one experimental implementation; the definitions do not require its architecture.
No training result, new theorem, biological universality or capability gain is claimed here.

## Scope and measurement

Specify a task/environment distribution D, externally justified goal/score s,
legal observations and interventions, prior training/experience, and resource budgets B.
A claim applies to that scope; changing D, goals, priors or B can change rankings.
Do not assume that effective control, memory, prediction or continual updating alone
establishes a general capacity for intelligence. Compare the following profile first.

| Quantity | Operational measurement |
| --- | --- |
| Existing competence C0 | Expected externally scored performance before the registered new experience, under fixed prior training and permitted inputs. |
| Acquisition A(k) | Paired improvement Ck-C0 after k new experiences; compare informative, absent and irrelevant experience with matched opportunities. |
| Retention R(tau,I) | Post-acquisition performance after delay tau and registered interference I, relative to immediate performance and the original C0. Report both references. |
| Transfer T(S) | Performance on a declared shift S: new instances, entities, combinations or environments, with permitted adaptation fixed in advance. |
| Resource profile | Actual data exposure; training/adaptation work; acquisition, storage, retrieval and execution costs; latency and peak memory. Measure energy only with a stated instrument/hardware boundary. |

Report uncertainty over independent task/source clusters and seeds; repeated seeds
on the same cases do not create new independent cases. Keep absolute performance
alongside treatment differences. A frozen system can deploy existing competence;
parameter/state updates count as acquisition only when useful competence improves.
Learning an episode's new mapping is a legitimate adaptation result, but does not
by itself show acquisition of a new learning algorithm or an open-ended capability.
The score and scorer must track the intended external goal, not merely training loss.

## An information requirement, not a new intelligence law

Let Y take values in a finite answer set. Let X contain **all information available
to a restricted predictor at decision time**, including any task-correlated weights,
metadata, external stores or cached outputs. Assume its prediction depends only on
X and random bits conditionally independent of Y given X. With exact-match scoring,

$$
\Pr(\widehat Y=Y\mid X=x)
 =\sum_y\pi(y\mid x)\Pr(Y=y\mid X=x)
 \leq\max_y\Pr(Y=y\mid X=x).
$$

Averaging gives the ordinary Bayes accuracy ceiling
$\Pr(\widehat Y=Y)\leq\mathbb E_X[\max_y\Pr(Y=y\mid X)]$.
The inequality is a convex-combination bound; it assumes no neural architecture.
It is an attainable information ceiling for an unrestricted Bayes decision rule,
not a guarantee that a resource-limited learner can attain it.

For the balanced two-rule case, sample Theta uniformly from {0,1}, independently
of the entire restricted decision-information tuple X (including fixed prior weights).
Let Y=f_Theta(X), with f0(x) != f1(x) for every evaluated query and no other channel
identifying Theta at the decision boundary. Then each of the two answers has conditional
probability 1/2, so every predictor restricted to X has expected accuracy at most 1/2.
An identifying earlier observation H can change that ceiling if information from H
is available through retained state M. This establishes an information requirement,
not that a particular memory mechanism learns, preserves or uses H successfully.
This is the memory-free/no-acquired-history arm; raw past context is also memory.

The 1/2 ceiling fails when query content/metadata, retained parameters, task order,
feedback or external stores reveal Theta; imbalance also changes the ceiling.
Include every legal channel in X and audit denied channels at the actual actor boundary.
The ceiling concerns population expectation: finite samples fluctuate above 1/2.
Exceeding an estimated ceiling calls for a statistical and information-boundary audit,
not an automatic accusation of cheating. Reset/shuffle controls additionally need
natural no-evidence and sham controls because interventions can create unfamiliar states.

## Decision-relevant preservation

Preservation should retain information needed for the registered future decisions,
not every detail of history. In a finite, correctly specified POMDP, the posterior
belief over hidden state, together with known time/remaining horizon when relevant,
is a sufficient history statistic for optimal control [3].
That result assumes the observation/transition/reward model and task objective;
it does not prove that a learned finite state is sufficient, easy to acquire or cheap.
A representation can preserve optimal decisions for one objective yet lose facts
needed after the objective changes. Test stated sufficiency/approximation conditions;
do not rename ordinary state estimation or conditional prediction as a new theory.

## A falsifiable representation claim — proposed, unvalidated

Define the candidate representation before fitting: separately addressable learned
operations/arguments that can be reused across tasks and recombined without being
supplied test-time rule labels, programs or a solver. Specify the distributed rival
and all supplied priors; “structure” must not expand post hoc to mean all computation.

Proposed H1: within a preregistered task family and budgets, this representation
improves absolute success on unseen combinations by at least a predeclared meaningful
delta over the rival, while meeting an absolute acquisition/retention floor.
Also preregister the familiar-to-unseen performance change: an interaction can locate
a transfer benefit, but alone cannot rescue a candidate with poor absolute competence.
Choose delta, floors, uncertainty method and decision rule using a separate pilot,
then freeze them before the confirmatory evaluation. Do not select them from its scores.

Necessary evidence: verify direct-evidence/oracle solvability and support identifiability;
randomize hidden episode rules after prior training; separate support/query and held-out
combinations; match data/order, active parameters, optimizer/tuning access and budgets;
preregister a total-work cap or accuracy/cost frontier, since equal updates alone
do not establish equal compute. Report remaining resource differences and limit claims.
Seal the complete registered checkpoint cohort before scoring. Audit causal retention,
forbidden information, source provenance and cold replay; preserve missing/failed checks.
Use matched/sham representation interventions to test the alleged reuse mechanism,
including controls for generic corruption and alternative information paths.
A precise null excluding delta weakens H1 for that scope; a wide interval is inconclusive.
A positive result supports that representation/task/budget claim, not universal structure.
LANGCONTROL098 addresses an earlier objective confound and cannot by itself establish H1.

## Prior art and attribution

[1] Legg and Hutter (2007), *Universal Intelligence: A Definition of Machine Intelligence*:
https://arxiv.org/abs/0712.3329 — a prior formal approach to machine intelligence.
[2] Chollet (2019), *On the Measure of Intelligence*: https://arxiv.org/abs/1911.01547 —
separates achieved skill from skill-acquisition efficiency, accounting for scope,
priors, experience and generalization difficulty. This note does not establish a replacement.
[3] Kaelbling, Littman and Cassandra (1998), *Planning and acting in partially observable
stochastic domains*, especially section 3.2: https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf.
The Bayes bound above is reproduced elementary reasoning, not a newly discovered theorem.
