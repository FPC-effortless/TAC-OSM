# ME056 — Real physical-robot concept-swarm CED training

**Pre-registered on 2026-10-22, before new ME056 original robot training.**
**Evaluation: developmental original exposed TEST only; do not claim fresh generalization.**

## One integrated question

Can TAC-OSM's physically anchored native prediction become reliably better at
future-state prediction by adding a *persistent concept population* and a
*shared-weight recurrent CED-style latent refinement loop*, under the SAME
real original observations, actions, masks, optimizer schedule and a
competent active-parameter-matched GRU?

This is not evidence of active inference, a trained policy, do(a) intervention
effects, commercial value or novel Transformer/SSM mathematics.

## Architecture and exact test

- Retain ME054 fast32/slow32/8x8 associative belief, shared
  native encoder and bounded physical/onset verification.
- Add six learned persistent 15-dimensional concept vectors per episode.
- Concept-to-concept relevance-weighted message exchange.
- Four shared-weight decoder passes refining temporary action-conditioned
  concept states, while the persistent facts remain unchanged.
- Future joint-state output anchored at genuine measured x_t:
  predicted_next = measured_x_t + D_FIT * decoded_physical_delta.
  Both models' physical heads initialize exactly to persistence.
- Train only on real measured next 14D joint state and actually observed
  human-takeover onset, masked when not defined. No concept oracle, no
  unsupervised future-state label fabrication and no reward/head padding.
- Real physical feedback uses ME054 zero-surprise identity dynamics;
  concept-feedback updates are intensity-conditioned and bounded convex
  updates. A zero error must leave all concept and native state bit-identical.

## Data and controls

Original BodenAI RW-RL HIL, immutable source revision
9290f5a96dc91ea58256e81d9ce2dd9621890b51.
Original 325 FIT, 80 DEV, 100 TEST robot episodes with genuine 14D states,
14D actions and human-control flags. Dataset FIT original aggregate
SHA256 4aa30d5009bf897438194cdd2d3b9e9e6ec1d923ec5389c93a942f9b7046c96f.
Preserve original source-separated cohort hashes in FAIR_COMPARISON.json,
no replacement for missing episodes.

Controls: physically anchored **fully trained GRU**, width 44, and physical
last-state persistence (unmatched practical baseline). Parameters fixed
*before first fit* from model geometry rather than heldout scores:
candidate concept TAC 45,822, GRU 45,995 (~0.38% gap), subject to runtime
recount. Five independent paired seeds 56001–56005. Each model:
600 AdamW updates, same batch2, 256 true observed history frames, 16 genuine
gradient-bearing physical transitions, same FIT sampling/ordering per seed,
.001 LR, .0001 weight decay, clip1, takeover BCE weight .2, zero tuning trials.
CPU runtime is efficiency ONLY and need not match. The common evaluation
checks physical MAE/RMSE and real human-onset BCE with honest support counts.
Train and archive all original models BEFORE reading original DEV/TEST frames.

**Strict pre-training stop gate:** source FIT SHA and cohort membership,
actual per-arm active parameter count (<=1%), identical genuine data/source
windows, shared budgets, AND a prefit PyTorch profiler check of the whole
original-source-shaped 256+16 training update. The profiler's reported
operator FLOP relative gap must be <=1%; otherwise NO architectural
comparison model fits are started. This operator profiler does not account
for every nonlinear kernel or certify full true consumed FLOPs. Thus even if
the gate passes, report only developmental evidence and do NOT claim fully
matched scientific superiority from partial instrumented FLOPs.
Never pad either architecture with dummy parameters or wasted iterations.
The profiler executes one discarded FIT optimizer update per arm; this
must be counted as diagnostic profiling, not misreported as zero updates.

This uses the original ME054 source loader/trainer and scoring code; the model
registry and budget are replaced with the concept-CED and the width44 GRU.
Any partial failure or missing publisher data is a scientific stop; all
actual source receipts and original model checkpoint hashes remain in
GitHub Actions artifacts, not invented.

## Known earlier actual results that constrain claims

Original ME052 TEST was opened: native TAC 30.552486, GRU 2.613201,
physical persistence 0.066949. ME054 re-used previously exposed TEST
and obtained native approximately 0.0788, GRU ~0.0923 across five
seeds, still both behind physical persistence 0.066949. ME056 must beat
persistence before its extra recurrence can be considered useful; doing
better than ME052 does not demonstrate novel intelligence.
These comparisons of different run setups are developmental context,
not a fully controlled new ME056 result.

Refer to:
- ME054: https://github.com/FPC-effortless/TAC-OSM/pull/329
- Source concept-CED variant: https://github.com/FPC-effortless/TAC-OSM/pull/331
- NCM hierarchical discrete concepts: https://arxiv.org/abs/2610.14162
- Deep Equilibrium Models: https://arxiv.org/abs/1909.01377
- DreamerV3: https://doi.org/10.1038/s41586-025-08744-2

**No training result exists until the actual GitHub source/fit workflow
succeeds and trained weights/physical outcomes are audited.**
