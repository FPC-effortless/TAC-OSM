# DIAGTRAIN001 — training sufficiency and actual proposal effects

No new model fits. All five original ME109 checkpoints, 4,096 saved losses per
model, original AdamW moments, and three authenticated original FIT batches were
examined. No REPORT or protected observations were loaded. The source-to-output
anti-cheat adapter was rerun; its scoped probes pass and overall scope remains
INCONCLUSIVE. Models, moments, checkpoint bytes and step counters stay unchanged.

## Does the model need more training?

**Not established. More of the current recipe is not an evidenced fix.**
The run has no intermediate model checkpoints or fixed validation learning curve,
so convergence and the best stopping point cannot be recovered from final weights.
Current stochastic FIT losses are not validation estimates; different endpoint
windows contain different examples. Context-stratification removes the known
8/64/256 schedule confound, not all sampling noise.

Five-seed means, first versus last 256 records **within each context**:

| History | Full objective, early → late | Point MAE / FIT scale, early → late | Point change | Interval score change |
|---|---|---|---|---|
| 8h | 6.885022 → 6.548743 | 0.490734 → 0.528699 | +7.7364% | −34.1899% |
| 64h | 7.095490 → 6.650284 | 0.475956 → 0.531320 | +11.6322% | −38.0027% |
| 256h | 7.230063 → 6.731188 | 0.473836 → 0.520232 | +9.7917% | −36.8682% |

All 15 seed/context pairs improve the total objective while worsening point loss.
In the last half, objective slopes remain negative in 11/15 pairs; point-error
slopes are positive in 12/15. Interval training is making progress while point
accuracy moves in the unfavorable direction. This is an observed trade-off,
not proof of a unique causal mechanism or population trend.

Each model had 65,536 supervised origin draws and 7,163,648 historical token
exposures, but only 30,284–30,388 distinct station/hour origins. Repeat draws
comprise 53.63–53.79%; different augmentation/context draws are not identical
examples, and repeated draws are not additional independent outcomes. Origin
counts do not include all distinct events appearing inside overlapping histories.
Inherited parent training is additional to these new-update counts.

Historical clipping frequency is UNKNOWN: norm1 clipping was called, but its
preclip norms were not recorded. Therefore neither insufficient exposure nor
excessive clipping can be declared the cause from these receipts.

## Local first-order protection versus actual finite-step behavior

For each frozen model and each of three original FIT batches, form exactly one
original-rate AdamW candidate from its stored moments. Compare the original
average-point guarded candidate with the same unguarded AdamW candidate. Evaluate
through torch.func.functional_call, without applying either candidate to a model
or optimizer. The second panel is the next existing FIT batch modulo3, not a
validation or independent sample. No learning-rate search or candidate selection.

| Candidate and panel | Total objective worsens | Average point worsens | Gap PM2.5 worsens | General PM2.5 worsens |
|---|---:|---:|---:|---:|
| Guarded, originating batch | 0/15 | 0/15 | **10/15** | 0/15 |
| Unguarded, originating batch | 0/15 | 4/15 | **11/15** | 1/15 |
| Guarded, next FIT batch | 7/15 | 8/15 | **12/15** | 6/15 |
| Unguarded, next FIT batch | 7/15 | 10/15 | **12/15** | 7/15 |

These extend the old directional-dot audit to actual finite-step losses. The
average guard helps the metric it protects on its own batch, but does not protect
gap PM2.5 or another batch. Removing it is not supported by this probe.
Guarded same-batch gap mean delta is −0.00005846 despite 10/15 adverse cases;
next-batch mean delta is +0.00002640. Small effects and shared batches/seeds do
not support significance or long-run performance claims.

The guard retains a median parameter displacement norm of 0.00502282 versus
0.00564016 for raw proposals (medians are separate, not a paired ratio). It is
not stopping all learning. Historical block projection rates are33.34–36.68%;
applied parameter-rounding fallback rates8.60–9.63%. These are block events,
not percentages of whole updates rejected.

## Mathematical boundary and next correction

For current averaged point gradient g and a proposed displacement d, enforcing
<g,d>≤0 does not imply <g_gap,d>≤0, nor L_gap(theta+d)≤L_gap(theta), nor improvement
on another batch. It also does not make cumulative drift impossible when each
step uses a different minibatch. DIAGTRAIN001 changes no architecture, optimizer,
loss, gate or Lean proof. Existing limited PNDS contracts remain as declared.

Next substantive experiment should combine per-source-availability/per-channel
protection or selective routing with a registered budget comparison and fixed
FIT-validation learning curves. Save intermediate weights and preclip norms;
measure marginal gains at prespecified budgets rather than repeating a failed
recipe until a gate passes. Data used for those curves must have an explicit
training/validation boundary. Any budget arms must receive equal opportunities
and comparisons must charge inherited training and diagnostic overhead honestly.
All new fits require a new registration and actual live 95k–105k parameter gate.
ME109 remains FAIL0/5; no more ME109 fits are authorized by this diagnostic.

## Reproduction

`PYTHONPATH=src:. python scripts/diagtrain001_curves.py` reconstructs CURVES.json
from original checkpoints. `PYTHONPATH=src:. python scripts/diagtrain001_training_audit.py --replay`
replays the fixed finite-proposal probe against RESULTS.json under the original
relative1e-5/absolute1e-6 contract. Exact input/checkpoint hashes and immutability
checks remain compulsory. Neither command trains or scores protected data.

Independent detached-checkout replay at bbd056e3 reproduced all15 finite probes
and CURVES.json exactly on this host.81relevant tests pass. Original actual-model
Observatory replay also passes with its full-integrity INCONCLUSIVE unchanged.
Draft PR#425 and master-plan commitea966c5424da377dd3d5d77b4e9ed314a5ce4369
preserve the findings. GitHub replay status is recorded separately.

## Cross-host replay failure — do not treat proposal counts as portable

GitHub40287673335 FAILS the unchanged numerical contract at
checks[1].guarded_displacement_norm: CI0.003032 versus original0.003479,
absolute difference0.00044741, relative12.858526%. Original failure log preserved.
Clean same-host replay success does not establish cross-host reproducibility.
The table above describes original-host probes; portable agreement is OPEN.
Potential near-boundary projection/rounding branch sensitivity is a hypothesis,
not an established primitive cause. No threshold or original result is changed.
A versioned diagnostic now exports current-host reports and lists every failed
field before still failing strict replay. This is additional instrumentation,
not a replay pass or model fix.
