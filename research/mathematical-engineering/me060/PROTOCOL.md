# ME-060: native trajectory experts — development registration

One integrated capability objective: improve factual action-conditioned robot
trajectories and provide measurable predictive uncertainty. Native TAC remains
a learned fast/slow/associative architecture. This does not substitute an SSM,
Transformer or pretrained world model for its recurrence.

## Source and exposure

BodenAI/RW-RL-HIL-Dataset, publisher revision
`9290f5a96dc91ea58256e81d9ce2dd9621890b51`, CC BY 4.0 per publisher card.
Use the existing `clean_stains_off_tabletop` metadata partition. Preserve whole
four-episode blocks. An outcome-independent hash of FIT blocks reserves 20
episodes for uncertainty calibration; 208 remain for training. DEV has 76
episodes. These populations have historical research exposure; all results are
developmental. The runner never loads old TEST trajectories, and no result can
be described as a fresh independent confirmation.

Normalize and identify the same physical servo prior from the 208 FIT episodes
only. Both models receive identical chronological 64-transition histories,
executed action/state/control records and 16 future logged action prefixes.
Future actions are factual supplied covariates, not actions guaranteed to be
known to an autonomous agent ex ante. Censor future errors from the first human
takeover. Histories may contain genuine past human observations. Robot-control
FIT origins are sampled uniformly with identical five seeded schedules.

Five paired seeds 60001–60005, 600 AdamW updates, batch eight, lr .001,
weight_decay .0001, gradient clip 1. The first 48 history steps are burn-in;
the last 16 receive gradients. Both architectures use the same 16-horizon loss
and 16 dense pre-feedback one-step losses, with the same .25 weighting. Loss
is normalized MAE plus .01 times a proper Laplace negative log likelihood.
No outcome-based width/learning-rate sweep or validation selection is allowed.
Widths were selected using counts and reported operator FLOPs only.

## Scientific gate and scoring

Run the machine preflight before any fit. Freeze all ten checkpoints and final
resource receipts before CAL/DEV scoring. Reprofile the final batch in every
arm. Report supported-operator FLOP estimates honestly: PyTorch does not count
all elementwise, norm, optimizer and hardware operations. This is a common
estimate convention, not a complete physical FLOP certificate. CPU times and
model states are efficiency dimensions, with no runtime matching requirement.

One deterministic midpoint robot-controlled origin per CAL/DEV episode; full
64-transition history and 16-step factual trajectory. Score original-unit MAE
per episode and horizon; include persistence and the same servo-only prior.
Average the five models' episode errors, then bootstrap episodes 20,000 times.
The exploratory development gate is >=2% lower MAE than GRU and both practical
priors, with TAC-minus-GRU 95% interval wholly below zero. It cannot promote
independent architectural superiority regardless of numerical parity.

Calibration uses the maximum absolute error divided by the learned scale over
all scored channel/horizon points per CAL episode. Apply finite-sample 90%
split-conformal quantile separately per fitted model, then measure whole-episode
and marginal coverage and interval width on DEV. No exchangeability guarantee
is asserted for this observational data. Forecast review uses caller-provided
physical tolerances; no cost-saving, policy-value or autonomy claim is supported.

## Prior art and specific architectural hypothesis

R2D2 supplies recurrent burn-in/replay principles; DreamerV3 establishes joint
robust world-model training; deep ensembles (Lakshminarayanan et al., 2017)
establish proper uncertainty scoring; split conformal prediction supplies
heldout calibration (Lei et al., JASA 2018, doi:10.1080/01621459.2017.1307116).
Direct multihorizon regression, physical residual shortcuts and associative
delta rules are established mechanisms. The new hypothesis is the utility of
their integration with TAC's bounded selective fast/slow state, direct
action-conditioned memory reads and horizon-conditioned residual experts.

Full independent qualification needs a newly preregistered authentic task/site
after development freezes. Preserve every failed fit and all negative scores.
