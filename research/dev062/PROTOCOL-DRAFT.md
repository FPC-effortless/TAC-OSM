# DEV-062 draft: public belief inference, not another clean-state imitation win

**DRAFT / NOT REGISTERED / NO FITS.** Final numeric thresholds, data namespaces,
compute caps and model selection must be committed before DEV-062 fitting. Do
not silently reuse DEV-061 held-out worlds or extend its budget.

The first target is a learned action-conditioned posterior update under genuine
partial observability, anchored in DEV-060's public-posterior harness. Learning
which clean record is latest is insufficient. Public action provenance must enter
the state update: sensor report, exact paid inspection and repair receipt cannot
be treated as the same observation likelihood.

## Information contract to implement first

- Missing initial states, 10% noisy ordinary reports, paid exact inspections.
- Public record kind, identity representation, observed-value mask, timestamp,
  executed action descriptor and receipt provenance. No changed-entity-only
  receipt leakage. No private fault bits, simulator RNG, teacher belief, or
  environment object may enter model inference.
- Current public posterior is a **training-only** target in stage A and an
  evaluator-only reference in stage B. The exact filter has the same public
  history, known-law metadata and acquisition costs as the student.
- Keep a privileged teacher separately labeled. Its advantage can include extra
  information and cannot serve as an information-matched policy comparison.
- Start with the existing two-entity exact joint posterior. Expand entity count
  and coupled dynamics under a separate registered capacity comparison. Factorized
  marginal beliefs can lose correlations; measure joint as well as marginal
  accuracy before claiming that factorization is sufficient.

## Comparisons and staged objectives

Stage A reconstructs the exact *public* posterior with proper probability losses.
Compare structured recurrent memory, generic memory, tuned GRU/attention and a
public-belief feature MLP. No stop override or analytical planner enters the
learned-policy deployment artifact. Public filter/planner controls remain outside
it. Report KL/log loss, marginal and joint Brier, calibration and current-state
errors before policy outcomes.

Stage B removes posterior and hidden-fault targets. Train action-conditioned
public observation/receipt prediction, with public inspection outcomes as an
available grounding signal. This is not automatically label-free or JEPA, and
latent identifiability must be checked. Compare frozen representations and
controlled joint refinement on fresh worlds and changed sensor/dynamics laws.

Train learned acquisition/action/stop selection using charged inspections and
asymmetric decision risk, keeping proper probability objectives separate from
cost weighting. Evaluate at least ten fresh training seeds and retain all fits.
Do not call probability heads calibrated merely because BCE was used.

## Separate the three limits

1. **Information:** privileged teacher versus exact public-information oracle.
2. **Representation:** exact public posterior versus reconstructed posterior on
   identical histories; use a common decision evaluator for this diagnostic.
3. **Decision:** learned policy versus the same common evaluator given the same
   belief; also test the learned policy with public-oracle beliefs as a diagnostic.

State when an oracle's finite horizon makes it approximate. Beating a Bayes-
optimal known-law solution is not a meaningful expectation. Unknown-law/misspecified
reference experiments need their own disclosed information contract; oracle-only
law knowledge is privileged, not a fair baseline advantage.

Report completion and physical repair separately; early/late/censored stopping;
inspection counts and costs; actual action/observation cost; public and private
regret; training/data-acquisition cost; hardware latency/state/weight memory.
Keep sensor-law shift, dynamic-law shift, unseen identities, entity count and
history length strata separate. New cancellation and continuous-association
schemas require explicit controls rather than an undocumented feature adapter.

DEV-063 retention, DEV-064 direct outcome interventions/CASM-LONG attribution,
DEV-065 staged optimization and DEV-066 integration remain separate claims. A
working noisy-world prototype cannot reopen retired MTSK superiority or imply
symbolic CASM discovery, original CDL distillation, C5 scaling or commercial
validation. General text interfaces and layered TAC depth remain later lanes.
