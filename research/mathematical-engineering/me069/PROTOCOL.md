# ME069 native enterprise event world model

This is an authentic enterprise observation lane under the master plan. It
does not claim robot forecasting, SLA intervention value, Woyengi production
readiness, revenue or fundability. Financial claims in the supplied discussion
are unverified and are not model evidence.

## Source and scientific split

See SOURCE_PROTOCOL.md, committed before original data access. BPI2012 is an
actual financial-institution loan application log with13,087 cases/262,200
recorded events. Exact publisher file identity was verified against its MD5
and size; an identical public research mirror supplied bytes after publisher
download timeouts. SHA256 is
`5cd9cc16b9bcb20bd4aae45666a5d87479ddbf47e6371618b6ad217174cecdf3`.
Raw traces remain external under the original4TU General Terms of Use.

Chronological source case cuts at60/80% of starts, with outcome-availability
exclusions, yield7,096 FIT /1,850 DEV /2,618 TEST cases. FIT excludes756 cases
whose final records cross the DEV cutoff; DEV excludes767 crossing TEST.
The single source file is ingested for case/time partitioning. This is not
sealed byte access; heldout scores/distributions are not inspected before the
model set freezes. All vocabulary, transition/time priors and proxy thresholds
fit only eligible FIT cases. IDs are split keys, never model features.

## Whole-model objective and comparison

Five paired seeds69001–69005,800 AdamW updates, batch32, lr.001,
weight_decay.0001, gradient clip1. Uniform FIT prefix endpoint sampling,
identical schedules per seed/arm. A window includes at most32 known events,
with dense next-event/time supervision at every nonpadded prefix. Full32-step
backpropagation, deterministic zero initial state and no future actions.
Mask padding in both recurrent state and loss. Count actual nonpadded tokens.
No DEV selection, early stopping, learning-rate sweep or checkpoint selection.
Widths are selected by original parameter/FLOP shapes only, before any fit.

Native state is fast64 + slow64 +8x8 associative delta memory. Activity and
resource embeddings are domain adapters; dimensions are configurable. State
size is a resource choice, not a claim that32D was proven inadequate. Elapsed
time affects native decay and is equally available to the GRU. Both use the
same transition-prior residual categorical head, zero-delay occurrence head
and positive lognormal waiting-time distribution. Recurrent state comes only
from actual past observed events. See MATHEMATICS.md.

Loss: categorical cross-entropy +.2 zero/positive Bernoulli score +.1 positive
lognormal score in log seconds. The target-only log-seconds Jacobian is
omitted from optimization equally in all arms. No fabricated failure reward,
transaction cost or canonical SLA is introduced.

## Scoring and acceptance

Commit all ten original model/optimizer/RNG checkpoint bytes before scoring.
Restore them from a separate clean checkout before final TEST inference.
Evaluate one deterministic midpoint prefix per eligible case, using the same
at-most32-event history and predicting the next recorded event. Last events
have no next-event target. This bounded protocol differs from published
all-prefix process-monitoring tables; no leaderboard claim is allowed.

Primary metric: case-mean next-activity log loss. Require at least2% lower
than the empirical first-order transition prior and1% lower than matched GRU,
paired case-bootstrap95% upper bound below zero and>=4/5 training seed wins.
Report activity accuracy and prevalence/current-activity repetition controls.
Report waiting-time MAE, nonzero-gap MAE, uncalibrated90% interval coverage and
width. A FIT75th-percentile positive gap defines a long-wait Brier diagnostic;
it is a proxy delay threshold, NOT a contractual SLA or actual failure label.
Also report prior/global-median and zero-delay timing baselines. No joint
intelligence qualification follows from activity accuracy alone.

Numerical parameter and common supported-operator FLOP estimates must match
within1% prefit and postfit. CPU speed is an efficiency result. Whole hardware
and unsupported-operator FLOPs remain uncertified. Preserve every original
checkpoint and negative result in research/replay-bundles/ME069. Clean-checkout
replay checks exact model/source identities and bounded score floats using
absolute1e-4/relative1e-5, with tighter1e-5 primary score tolerance. These
tolerances are fixed before scoring, not chosen after observing mismatches.

## Prior art and claim boundary

Tax et al.,CAiSE2017,doi:10.1007/978-3-319-59536-8_30, establish next-event
and timing prediction for business processes. Baytas et al.,KDD2017,
doi:10.1145/3097983.3097997, establish time-aware recurrent representations.
Delta-rule associative memory, Markov process priors, hurdle distributions
and predictive recurrent state are established. The research question concerns
their integrated native TAC utility under this authentic temporal case split.

No execution authority is derived from probabilities. Woyengi remains owner
of verified state, permissions and effects. Veritas can replay the advisory
contract or synthetic engineering fixtures; it cannot convert them into real
predictive utility, causal intervention value or commercial validation. The
robot tissue-box holdout remains completely untouched in this enterprise lane.
