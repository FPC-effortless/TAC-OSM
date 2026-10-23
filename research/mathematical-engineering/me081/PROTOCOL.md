# ME081 — native horizon-query distributional state learning

**Prospective single-model capability development, physical lane.** One integrated
recipe, five cold starts, no tuning/restarts/warm starts. No new architecture
winner, causal action, general intelligence or deployment claim. Read the current
master plan (7e5ce85f), ME078/079/080 adverse results, real-world rules and capability
truth standard. Every previous registration remains closed.

## Target capability

Train a native fast32/slow32/8×8 matrix model on **32 actual observed frames with
full 32-step gradient credit**, rather than eight-frame reset supervision. Use a
shared nonlinear horizon-query decoder so the model learns one forecast family
conditioned on known 1/4/16-frame horizons. Train complete coordinate distributions
with properly scored CRPS, a small log-density term and factual motion supervision.
The purpose is useful quiet/moving-state and mid-horizon physical forecasting,
not isolated memory attribution or another execution-speed study.

There are 100,165 active/deployed learned parameters. No frozen encoder, dummy
padding, multi-model deployment or teacher model. Native state is bounded and
its update mathematics remains intact. The decoder uses continuous fixed horizon
features plus learned shared weights; this does **not** guarantee consistent
joint trajectories or accuracy at unseen horizons. Marginal distributions remain
a quiet/moving Laplace mixture; moving scale is at least twice quiet scale.

## Exact objective and established mathematics

For d=|moving location| and quiet/moving scales b,c (c>=2b),

E|L(0,b)-L(m,c)| = d + [c^3 exp(-d/c)-b^3 exp(-d/b)]/(c^2-b^2).

CRPS(F,y)=E|X-y| - .5 E|X-X'|, with exact mixture weights and Laplace absolute
moments. This formula is independently tested against numerical CDF integration.
Positive scores are averaged in physical units normalized by FIT displacement
and horizon; loss = normalized CRPS + .02 physical NLL + .2 genuine global
motion BCE. Scales are positive with no postfit upper cap. CRPS penalizes
unnecessary spread without suppressing honest uncertain outcomes. It does not
guarantee calibration or successful optimization.

Reuse Gneiting & Raftery (2007), strictly proper scoring rules,
DOI10.1198/016214506000001437; Bishop (1994), mixture density networks;
DeepAR (2020), DOI10.1016/j.ijforecast.2019.07.001; and established gated-delta
memory. Their established methods are not claimed as new TAC discoveries.
Shared horizon-conditioned forecasting, longer BPTT and proper scoring are
established principles. The unresolved question is whether this complete native
system learns more useful factual physical forecasts under the registered recipe.

## Authentic source and learning exposure

Unchanged original BodenAI/RW-RL-HIL source revision
9290f5a96dc91ea58256e81d9ce2dd9621890b51, CC-BY-4.0 with BodenAI/RW-RL attribution.
Original 160-episode archive: 120 FIT / 40 already-exposed REPORT, FIT-only
normalization/servo statistics. Protected tissue-box/source CAL/TEST remain closed.
No synthetic training or observations. Commands are recorded lag-one commands
known at each observation, not future actions or identified causal interventions.

Training requires 32 recorded autonomous context frames, last state t and targets
at t+1/4/16 for every actual prefix. Future takeover/source endings affect label
eligibility only. All features are causal, even when a training target also occurs
later in the same input chunk. Batch observation encoding and shared decoder have
no cross-time leakage; the native recurrence is causal.

Five seeds 81001–81005, 2048 AdamW updates, batch24, lr.0007, wd.0001, clip1.
49,152 windows and 1,572,864 prefix presentations per fit, repeated presentations,
not independent observations. This is four times ME080's observed-prefix exposure;
no matched-compute, sample-efficiency or architecture-superiority claim.
Nominal half-uniform/half-motion window proposal is importance corrected to
uniform eligible windows per task. Task round-robin component counts differ by
at most one draw; no claim of exact finite-sample target balance.

## Primary and secondary evidence are separate

**Primary:** same original 960 eight-frame REPORT queries, all available 1/4/16
future pairs, frozen original full-model predictions as references. This preserves
the actual old/new input/target/horizon task. Scores are means of individual fits,
not five-model ensemble predictions. Compare against ME077, ME078, ME079, ME080
and physical persistence/servo/clipped velocity without fitting new controls.

**Secondary:** score the same registered query origins where 32 actually observed
autonomous context frames exist. Freeze query IDs/denominators before fitting.
Report 32-frame accuracy, uncertainty and risk on every eligible future horizon.
This is a genuine longer-context task, but not memory superiority. Do not compare
different-input populations as if they proved architectural advantage. No new
independent-source evidence is acquired. Repeated seeds/histories do not add cases.

**All acceptance conditions** are frozen in FAIR_COMPARISON.json: primary H1>=5%
better than ME080, moving H1 within5%of ME078, stationary H1 within10%of ME077;
H4>=5%better than ME079; H16 within5%of ME080. No task H1>10%worse than persistence;
coverage>=85%and interval score no worse than ME080 at each horizon. Secondary
32-frame error beats persistence at each horizon and moving H1 beats clipped
velocity. Paired episode bootstrap95%upper(new-ME080 H1)<0 and >=4/5 seed wins.
Failure produces an honest mixed/regressed/inconclusive verdict, not extra fits.

## Preservation and replay

Machine preflight and actual whole-model budget guard precede every optimizer
update. Preserve original optimizer/RNG/schedule/code/source hashes, each final
checkpoint and actual exposure receipt in Git. Freeze all five originals BEFORE
both REPORT readouts. New final delivery includes original registration hash,
actual final tensors and all2048guarded updates. Independent fresh-clone replay
must restore and rescore both information conditions. Original numeric contract:
float32 neural weights, float64 quantiles/scoring, rtol1e-5/atol1e-6. No tolerance
relaxation or original model/score edits. All old replay exceptions remain visible.
The first no-fit preflight draft is preserved; only literature metadata was
completed before authoritative registration and before any optimizer existed.

## Capability-to-Value Card

Skill: genuine longer-observation, quiet/movement-aware physical distributional
forecasting with useful uncertainty. Operator question: could an advisory forecast
warn a person about actual movement without stationary jitter or overly wide
bounds? External pilot and operator economics: NOT_MEASURED. Customer relevance:
UNKNOWN. Business stage: research only. Physical evidence does not qualify the
Enterprise, Temporal, Edge or Counterfactual lanes. No autonomous action permission.
