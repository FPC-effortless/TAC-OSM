# ME080: integrated distributional predictive-state learning

Prospective single-model capability development in the **physical** lane.
Read the latest master plan at commit51113071 and repository capability truth
standard. The old ME078/079 registrations remain closed; this is one new recipe,
five cold starts, no tuning or second attempt under this registration.

## Capability and falsifier

Build one native fast32/slow32/8×8associative model which predicts stationary
and genuinely moving physical futures together, rather than trading one for
the other. Keep causal observed state/lag1known command/velocity/acceleration.
Predict1/4/16frame coordinate distributions from eight observed frames.
Do not attribute gains to memory, claim an architecture winner, causal control,
autonomous decisions, arbitrary-history accuracy, safety or enterprise transfer.

The complete machine registration specifies all score gates. Primary gates
must hold simultaneously: H1MAE at least5%below frozenME079, movingH1 no more
than5%above ME078, stationaryH1 no more than10%above ME077. H4/H16 must remain
within5%of ME079, no taskH1 more than10%worse than persistence, coordinate
coverage at least85%and proper nominal90%interval scores no worse thanME079
at every horizon. Five-seed means, >=4/5paired-seed H1wins, paired40episode
bootstrap95%upper(new-ME079)<0. Otherwise record mixed/regressed/inconclusive.
A gate failure ends this registration; no extra fits or TEST tuning.

## Computation reuse without changing TAC identity

The mathematical model is a pure function `(observation, known_command, state)
returning `(distribution, new_state)`. The caller owns entity routing and cache
invalidation. The state includes fast/slow/matrix plus previous observed state,
velocity and first-observation flag. No global cache, no synthetic event and no
in-place mutation of caller state. When model weights/preprocessing change,
old states must be discarded/recomputed; inference caches cannot be reused
across optimizer updates without accounting for state staleness.

Training vectorizes observation-only encoder/address/input projections once
per chunk. Nonlinear native recurrence remains sequential. This is ordinary
common-subexpression reuse and explicit recurrent caching, NOT an associative
scan, category-theoretic novelty, compiler fusion or demonstrated GPU speedup.
Full/chunk/step/interleaved agreement is an engineering contract. Training and
real scores use eight-frame resets;40-step equivalence tests do NOT qualify
long-history predictive performance. No timing-based capability claim.

## Probability model

For each coordinate/horizon, predict a two-Laplace mixture: a zero-centred
quiet delta and a learned bounded motion delta. Learn both scales and mixture
probabilities using proper marginal NLL. The median and5/95%quantiles are
inverted from that same CDF. No separate unconstrained quantile heads or
post-hoc hard routing. A separate global factual motion head is supervised;
its event is mean14Dphysical displacement>=.025. It is NOT the probability of
the latent mixture component. Coordinate marginals are coherent, but they are
NOT a coherent joint temporal path or a cross-coordinate dependence model.

Published methods reused: Bishop1994Mixture Density Networks; Gated Delta
Networks; R2D2state handling. Mamba-2motivates stateful execution rigor but is
not used as a replacement architecture. No novelty/superiority claim.

## Source, training and preservation

Same lawful CC-BY-4.0BodenAI/RW-RL-HIL pinned original160episode subset,
120FIT/40already exposedREPORT. FIT-only preprocessing/servo/scales. Same
960H1,956H4,928H16observed future pairs. Future takeover masks target
eligibility only, never inputs. Episode-cluster uncertainty; same-session
operator/site independence unestablished. Protected tissue-box, sourceCAL/TEST
remain closed. No new source or external pilot claim.

Whole model99,816active/deployed parameters. Five seeds80001–80005,
2048updates×24windows×8prefixes, AdamW.0007,wd.0001,clip1. Fixed recipe,
all final checkpoints, no sweep/restart/warm start/REPORT selection. Same
importance-weighted proposal as ME078/079. Training differs in objective;
no equal-compute or architecture-superiority claim. Record actual exposures,
CPU and original optimizer/RNG/live guard evidence. All originals frozen in
Git BEFORE REPORT scoring; independent fresh remote clone restore+rescore.
Float32neural math; prospectively float64CDF inversion and scalar scores,
unchanged raw contract rtol1e-5/atol1e-6. Historical replay failures unchanged.

Preflight draftV1was generated before correcting the new final-verification
API call; preserved as NO_FITS evidence. Authoritative registration generated
again before any optimizer exists. No outcomes used to make this correction.

## Capability-to-Value Card

Skill: physical uncertainty/actual-motion forecasting with reusable native state.
Operator question: could an advisory forecast distinguish real movement from
stationary jitter? Customer/operator decision benefit UNKNOWN/NOT_MEASURED.
No customer, action authority, deployment or commercial readiness unlocked.
The separate enterprise/Woyengi lane is not qualified by this study.
