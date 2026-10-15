# DEV-018 — Memory attribution and horizon controls

Status: fixed adaptive-development protocol, committed before full training.
No outcome claimed. Parent: draft DEV-017 #173 at
`80e7fc42f73569fceee23c129f25d209a376f251`. Keep draft and unmerged.

## Question and single primary

Does the native TAC learned writer preserve generation-dependent cancellation
information beyond event-encoder learning and standard recurrent memories?
Primary: **pre-sensing target wait-future accuracy at H512**, pending/reissued
cancellation roles, at/after due, equally weighted registered delays32/128/512.
Opposite-past pairs give a **50% majority baseline**. Before-due cases have their
own class proportions; no universal 50% baseline is assumed there. Report all
seeds, not maxima across arms or endpoints. No test-set checkpoint selection.

## Controlled model interface

Ten arms: current-observation-only memoryless head, 768-scalar dense GRU,
768-scalar standard gated delta equation (four8x24 heads), learned indexed
schema41, frozen-random-GRU schema41, and four native TAC writer interventions
(joint, encoder-only, writer-only, reader-only), plus a full-history transformer.
All get identical initial event encoder16->32->12, public query encoder14->12,
and final prediction head44->64->2. Read adapters necessarily differ across
architectures. The four TAC controls have identical initial entire tensors.
Schema controls also have identical initial tensors. Gates assert fixed/trainable
component hashes and actual finite nonzero gradients.

TAC here uses the actual existing `DeltaMemory` writer, with an independent
common query-key reader and final head. It does **not** execute CASM and cannot
establish a gain for the complete TAC-OSM system. This isolates memory from
conditional executor capacity and writer/read-key coupling.
Standard gated delta: decayed state=alpha*S, then
S'=alpha*S+beta*(v-(alpha*S)k)k^T, normalized keys, per-head alpha/beta.
This is an equation-level baseline, not a reproduction of optimized Gated
DeltaNet kernels or a modern-baseline superiority claim.

Schema: public entity addressing is fixed, five learned dimensions per entity
plus one learned global dimension. No hidden bit/generation/timer assignment.
Frozen schema still has random recurrent dynamics; it is not memoryless.
Memoryless retains only12 encoded current-observation features. Transformer
retains the full Hx12 encoded prefix plus temporary attention tensors, with
one32D/four-head layer,64D FFN and query cross-attention. It is an external-history
reference, not a fixed-memory competitor. No unseen future tokens are supplied.

## Training and evaluation exposure

Three fixed seeds43/59/71,216 AdamW updates,lr.002,weight decay.0001,clip1.
Short recipe: H128,batch8. Long recipe: H512,batch2. Each sees **221,184
pre-sensing events**. Short sees1,728 episodes/query sets and training receipts;
long sees432. This matches pre-sensing event count and update count, **not**
label exposure, receipt count, FLOPs or wall time. A long-short contrast tests
these two exposure recipes, not a uniquely isolated length intervention.
Within each recipe every arm receives identical cases and receipts.

Both recipes train due delays32/128/512; evaluation uses the same delays.
Every update supervises current/wait/service outcome bits for all eight entities,
mean pre/post one common random inspection CE. Supervisor targets never enter
model inference. Schedule spans before/at/after deadline and all four roles.
No target mask, role, hidden world, generation label or exact tracker enters
learned state transitions or predictions. All arms start fresh.

Evaluation grid: H32/64/128/256/512 x due32/128/512 x before/at/after x
retention/acquisition/pending/reissued x four opposite-past pairs. **1,440
public episodes per checkpoint**,96 primary critical episodes. H is exact
public observation count. Due delay is remaining time after the last observation;
relevant-event age and prediction distance are distinct. Accuracies are exported
for every history/delay/deadline/role cell. Since short and long regimes each
have one training H, shorter test H also changes absolute time distributions.
A model winning at its training H establishes neither universal length robustness
nor impossibility of another architecture. Parameters and actual CPU timings
are recorded; no cross-model matched-FLOP claim.

## Physical decisions and action-selected sensing

On **all H512 cases**, cost=4*failures+services+.1*inspections, divided by8.
Service iff4*p_wait>1+4*p_service. Compare do-nothing, always-service and the
exact public-history controller, as explicitly hand-coded references. Export
failure/service/cost frontier at fixed thresholds0,.125,.25,.5,.75,1,2;
physical operating threshold.25 is not selected from test outcomes.

Sensing budget is **one**, intentionally separate from DEV-017's two-inspection
heuristic. For each entity simulate both possible public current-bit receipts
through the learned model. Weight ensuing minimum predicted action cost by
predicted current-bit probability. Inspect the largest strictly positive estimated
cost reduction after charging.1; otherwise stop. No actual receipt is consulted
until selection. Compare none and matched-count random sensing. This is a model-
based one-step information heuristic, not learned RL or guaranteed optimal VOI;
its16 hypothetical updates/decodes are charged in runtime reports. It cannot
magically observe an earlier generation cancellation through a current-bit query.

## Fixed decisions and interpretation

Reliability/cost success per arm requires **every seed** primary>=.80,
H512 selected failures<=.05, unnecessary services<=.10, cost<=.80, and mean
selected cost no greater than no sensing. TAC writer attribution additionally
requires joint-encoder and writer-reader primary gains each>=.05 with paired
three-seed95% Student-t df2 lower bounds>0, plus runtime state-swap drop>=.10.
Reset must give exactly50% on balanced primary; leaks invalidate results.
Full DEV018 TAC success is reliability plus writer attribution. Superiority to
GRU/delta is a separate reported contrast, never implied by writer attribution.
All intervals are exploratory; no claim of confirmatory multiple-comparison
control or independent replication. Three seeds remain a substantial limitation.

A schema win supports its structural prior; learned-versus-frozen transitions
answer whether learning dynamics adds value. A GRU/delta win supports the
simpler model on this task/budget, not universal architectural superiority.
A transformer-only win shows accessible complete history helps. Long-training
improvement implicates exposure recipe, not training as the sole cause.
If every model fails, training/readout/task adequacy remains unresolved.

## Evidence and integrity

GitHub CPU runtime pinned to Python3.12,torch2.14.1,numpy2.3.3. Save60
checkpoints and all86,400 raw evaluation rows. Reload every checkpoint and
regenerate every forecast, state intervention, chosen sensing decision and cost.
Separate stdlib auditor regenerates public worlds, targets, costs and selection
arithmetic, checks frozen component digests and computes all intervals/gates.
Tests/replay establish integrity, not capability. No SGD replication is claimed.
Publish `per-seed.csv`, `accuracy-by-history-delay.csv`,
`failure-service-frontier.csv` and compact analysis in the repository after audit,
so summary evidence is inspectable without Actions-artifact authentication.

## Research motivation

[H-JEPA](https://arxiv.org/abs/2610.06805) motivates future hierarchical
prediction, but its slow-feature collapse warning makes brief causal events
an essential probe. [RoboJEPA](https://arxiv.org/abs/2610.10515) motivates
examining training/deployment horizons; its multi-change training recipe is
not proof of a horizon-only effect. [DeepJEPA](https://arxiv.org/abs/2610.00368)
motivates decision-value-based computation, but adaptive depth is deferred.
[Harness-Zero](https://arxiv.org/abs/2609.24974) motivates later corrected
trajectory distillation, deferred until the verifier/controller is reliable.
[Nested Learning](https://arxiv.org/abs/2512.24695) precludes novelty claims
from multiple update frequencies alone. No hierarchy, adaptive depth or MTSK
novelty claim is part of this experiment.
