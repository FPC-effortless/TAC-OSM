# ME-053: Original real-world trained-model physical-prediction failure forensics
Registered on 2026-10-22 AFTER ME052 actual TEST evaluation and BEFORE this
diagnostic's DEV source reads. This is a RESEARCH DIAGNOSTIC, not a new
performance-success claim, architecture bake-off or optimizer study.

## Evidence motivating next decisive research work

Actual real physical robot TEST 100 episodes showed TAC MAE mean 30.552486,
trained GRU 2.613201, and persistence 0.066949 original joint-state units,
all five paired seeds; TAC 0/5 wins, original episode bootstrap 95% interval
for GRU minus TAC [-29.95,-26.05]. Original training-data/parameter/update
fairness passed; CPU speeds intentionally NOT matched. Separate original
TEST human takeover outcome positives: only TWO in 1,282 sampled target
positions, insufficient to establish meaningful autonomous corrective skill.

The key scientific failure is not efficiency. **Neither recurrent model
learned a basic locally stable physical transition**, so further small
relevance gates, decays, or routing variations are not an appropriate use
of compute. Trained world models must at least compete with persistence.

## Mechanistic hypothesis grounded in the ACTUAL untrained architecture code

The native TAC physical-feedback update computes
`correction = .25*sigmoid(gate(z,a,r))*tanh(proposal(z,a,r))`,
where `r=tanh(observed_x_next - predicted_x_next)`.
Even with exactly ZERO observed prediction surprise `r=0`, ordinary
network biases and the existing recurrent state can produce nonzero
correction. Native scalar feedback similarly may update on zero
surprise. Across several hundred genuine physical frames this lacks
an exact neutral-error/no-op guarantee. This is an identified
mathematical invariant deficiency, not proof that it caused the
observed TEST gap. Independently, both systems predict absolute
physical `x_next` instead of anchoring to already observed `x_t`
and learning a residual `delta_x`. Correctly residualized physical
dynamics is well-established and NOT claimed as TAC novelty.

The original state normalizer includes two robot position channels with
training standard deviations ~37.97 and ~41.14, so normalized
prediction differences can be greatly amplified in original units;
the audited MAE calculations are still legitimate. Decompose to
determine whether recurrent belief instability and inability to model
low-movement dynamics contribute to this failure.

## Honest original-trained-model DEV diagnostic

No original weights are re-fit; use EXACT frozen original source SHA and
all original ME052 trained model checkpoint SHAs from run #39714369920.
Use two frozen original paired seeds 52001 and 52003 (chosen without
DEV results) and exactly 12 original physical DEV episode IDs selected
by SHA256 of (fixed study ID, published revision, original episode ID)
with NO reference to operator outcomes or model error. Do not reopen
TEST or access fake synthetic state machines.

For both original native TAC and fully learned GRU, keep the SAME actual
original 14D robot states, logged actions, masks, and prediction-before-
feedback conditions. Evaluate:
- **Full factual verification**: same original genuine trained-model
  evaluation path as ME052, both physically measured next state and
  next operator-takeover evidence arriving after prediction.
- **Physical-only actual feedback**: same original measurements and
  real prediction path; omit only human-takeover feedback.
- **No feedback beyond actual next observation**: still observe each
  genuinely measured state sequentially, but suppress all learned
  postprediction correction. This isolates the feedback mechanism
  while preserving same recurrent observation information.
For each mode, read the same genuine original DEV episodes and score
actual physical errors in intervals 256..511, 512..1023, >=1024,
recurrent state norms, and normalized forecast norms. State updates
happen ONLY after the original true next frame. No optimizer updates,
no synthetic physics, no reward or causal-action claim.

This is a posthoc diagnostic motivated by ME052 TEST. It therefore CANNOT
establish a new independent architecture superiority claim, but can
falsify specific failure-mechanism explanations on genuine DEV data.

## Decision next (not a micro-variation search)

If removing real feedback eliminates the extreme native error while
preserving the actual observed state inputs, the learned verification
update is the dominant proximate failure. If instability remains in
no-feedback path, native observe/retention dynamics or absolute-state
prediction must be addressed. Regardless, both architectures must
eventually use mathematically legitimate near-identity action-conditioned
physical prediction and genuine innovation-proportional corrections.
This known good engineering should be adopted as one coherent
world-model correction, not an ever-growing factorial test of trivial
variants. Then train both systems with the SAME real data, active
parameters and optimizer budget, at a newly preregistered fixed
budget large enough for convergence diagnostics. The already exposed
ME052 TEST cannot qualify that new architecture as independently
validated. An original unseen physical task or independently
reserved new cohort is required for a fresh confirmatory claim.

ME053 will preserve all genuine original model/data/source references
and any negative findings even if every candidate remains inferior to
persistence.
