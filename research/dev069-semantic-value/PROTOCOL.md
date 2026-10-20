# DEV069 — freeze the world model; learn a probability-linear value readout

Preregistration. No actual fitting before this protocol and actual fitting source
are committed. This is a controller mechanism study, not whole-agent or commercial
qualification. DEV068's failed calibration/reliability evidence stays intact.

## Hypothesis and boundaries

DEV068 validation interventions on fixed first14seeds show that exact public
belief alone does not repair all failures, while replacing both belief and world
features with known-law values can substantially degrade completion. That is an
OOD/interface diagnostic, not proof of an optimal oracle or isolated policy fault.
Test whether a generic probability-linear readout is more robust than an equally
sized unrestricted MLP while holding every state/dynamics weight fixed.

MLP:30->64->64->1,6,209trainable parameters. Expected-value readout: public8->64->
64->22;21learned outcome utilities and one learned positive cost coefficient:

`Q = sum_j(probability_j * learned_utility_j(public_context)) + softplus(beta(public_context))*predicted_cost`

The expected readout has6,166parameters,0.7%fewer. It is affine in probabilities
at fixed public context and cannot prefer a higher predicted cost with all else
fixed. No physical class receives a programmed utility, no repair/stopping rule,
analytical posterior, transition law or search occurs in inference. These generic
inductive constraints, supplied21-class vocabulary and public typed schema are
explicit priors, not autonomous program discovery. Utilities may still learn
incorrect outcome orderings; performance must be measured.

## Frozen parents and arms

Reuse ALL20completed DEV068 outcome-derived encoder/world initializations, NOT a
selected successful seed. Parent sourcea6d8cf30ddf69fe3c14980c54e27ba47059a8854;
full parent status must be VALIDATION_REJECTED,88complete fits,20fresh groups,
with test worlds unopened. Source/individual checkpoint/data hashes are checked
and indexed before any fitting. Parent teacher-control models are excluded.

For each parent backend(TAC,GRU,no-CASM), retrain two policy heads(MLP,expected)
from fresh initialization: six arms, shared immutable predicted feature tensors
within each backend. Parent seeds96801–96820 pair with policy seeds96901–96920.
These are20new policy initializations across20existing parent memory/dynamics
initializations, NOT20new end-to-end world-model training runs. No memory, gate,
belief, actor, future or cost-head gradient/update is permitted; body hashes and
unchanged predictions must verify this for every fit.

Training consumes ONLY parent's immutable train.pt from public factual actions,
inspection measurements and verified terminal costs; no parent validation/test
rows, analytical future/policy targets or teacher-only posterior file is read.
Training collection fees, missing real prices and synthetic scope remain charged
and disclosed through the parent index. Do not hide prior encoder/dynamics/data
costs behind a cheaper head-only fitting number.

## Optimization and selection

Exactly8,192updates for every head; batch256 common sampled transition IDs across
all six arms, float64, one CPU thread/worker, AdamW decay.0001, clip1, EMA.995.
Same undiscounted DoubleDQN/Bellman MSE plus CQL.01 as DEV068; actual learning
cost is physical fees plus registered20noncompletion penalty. Proper state/world
probability models stay frozen. No probability objective or action teacher added.

Two pilot LRs.001/.003, policy seed96900, fixed parent pilot0(seed96800/LR.001).
Fresh validation worlds6691002,32worlds each H32/H128,error.1/inspection.1.
Each arm selects LR lexicographically: highest minimum stratum completion, fewest
faults, lowest total physical cost, smaller LR. Expected-only candidate entry
requires>=95%completion AND zero final fault bits in EACH stratum. Select eligible
expected arm by lowest total validation cost, lexical tie, before fresh fits.
No eligible expected arm=>VALIDATION_REJECTED; no fresh fits/test opening.

If entry passes, run the COMPLETE six-arm×20policy-seed matrix, including weak
arms. Candidate must pass the SAME screen in>=18/20fresh groups before any test
opening. Otherwise close VALIDATION_REJECTED. No test-driven recipe fixes or
seed omission. Both raw/EMA weights, optimizer/RNG state and all curves retained.

Two local CPU workers, per-group900CPUseconds, total fitting/preparation cap12,000.
Incomplete/capped/nonfinite fits cannot qualify. No paid hardware. Atomic full
training snapshots every1,024updates include raw/EMA/optimizer/generator/step/
source/LR state. Only executor interruptions may resume with unchanged code and
parameters; numerical/cap failures remain failed. Conservatively charge a full
group cap for each interrupted group's unmeasured lost interval, retaining all
snapshots and recovery records. Completed groups are reused, never refitted to
improve a score. A unit falsifier verifies exact interrupted/uninterrupted parity.
Smoke uses+10,000,000policy/evaluation namespace offset and4updates; excluded.

## Sealed assessment and gates

Primary worlds6691003,64paired fresh starts per H32/H128/H512/H1024,error.1,
inspection.1. Secondary6691004,32starts H128/H1024,error.2/inspection.4. All six
arms and20seeds evaluated. Parent test namespaces6681003/4 remain unopened.

Readout-component qualification requires>=18individual candidate policy seeds
with>=95%completion in EACH primary/secondary stratum; no additional final faults
against same-backend MLP for ANY seed/stratum; no primary or secondary residual
fault in the selected expected arm. Paired mean physical-cost noninferiority
upper<=+.02vs same-backend MLP, using10,000hierarchical paired bootstrap resamples
of20parent/policy seed pairs AND64world IDs(shared across primary histories),
RNG969003, one-sided95%upper. Report all histories separately, same-backend head
contrast and all backend controls. No CASM/memory superiority from this design.

Structural semantic checks: probability-mixture affinity and nondecreasing Q in
its own predicted-cost input, at fixed context, must hold on every checkpoint.
Known-world interface interventions use only public known-law beliefs/transitions
as EVALUATION-ONLY OOD probes, never targets/inference. Predeclared probe set:
same primary worlds H128/H1024. Require candidate known-world completion no more
than2percentage-points below predicted-mode in>=18seeds and no extra final fault
bits in any seed/stratum before calling the readout semantically robust. Failure
means no component promotion even if ordinary action accuracy improves.

Report stop precision/recall, false stops, healthy exhaustion, collateral repairs,
newly created fault bits, accumulated fault exposure, call counts, inspection/
repair/observation fees, top-two margins, p95/p99posteriorKL and per-seed curves.
The frozen belief bodies cannot demonstrate new belief learning or fix an old
fixed-history calibration failure by changing which histories the policy visits.
Whole-agent qualification remains FALSE here; parent calibration/generalization
gates need separate new evidence. Actual repair-economics/source contract still
unmet. Do not market a controller-component result as a commercial model.

## Reproduction

Commit fitting source/protocol before actual fits. Verify parent manifest and
every loaded checkpoint; preserve checkpoint/preparation/selection hashes and
failed runs. Replay every final checkpoint and public action trajectory, export
Torch-only first-declared policy seed per arm, verify frozen-body parity, replay
native Veritas effects on first-seed primary trajectories and record actual
CPU/memory scope. Veritas effect tables share synthetic physics and are not
independent real-world trials. All old DEV/REAL evidence and MetroPT July B stay
intact. Overall maintenance goal ACTIVE; each autoresearch rejection is archived,
not erased or retroactively changed into success.
