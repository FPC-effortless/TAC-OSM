# ME037 — context-integrated native memory with bounded horizon control

Prospective registration before numeric parsing/fitting of the new household source
and before any ME037 fit. Architecture-first complete-system construction, not a
mechanistic ablation prerequisite. ME036 and BENCH-FM-002 remain unchanged drafts.

## Evidence interpretation

ME036 already conditioned its head on explicit level/scale. It encodes96inputs;
its720outputs are direct forecasts, not720recurrent rollout updates. Long-horizon
state dilution and a causal patch-stability explanation are unproven hypotheses.
This study adds a learned full-window context representation and a constrained
correction, rather than presenting existing level/scale features as new.

PR300/source d965964ad58f853465dda763e63106d9ea74caef supports task-specific
TimesFM/TabFM wins over its adapters, not universal forecasting dominance or proof
of TAC control advantages. Forecast targets there are+24/+96endpoints, distinct
from ME036 full sequences. Histogram boosting is not XGBoost; TAC tabular adapter
was supervised, not zero-shot;1.5s training+inference is not a microsecond streaming
measurement. No authentic action/inspection-cost intervention stream is added here;
no action-control, VoI or causal capability claim. Keep PR300 open/unmerged.

## Sources, cohorts and preprocessing

ETTh1/ETTh2/ETTm1/Weather/Electricity: exactly ME036 source checksums, channels,
chronological splits and96-step observation contract. Their test outcomes are now
**previously exposed development benchmarks**, not prospective qualification.
ETTh1/ETTm1 share physical equipment. Weather anomalies remain unchanged/disclosed.

New site: UCI374 Appliances Energy Prediction, DOI10.24432/C5VC8G, Luis Candanedo;
measured household energy and ZigBee temperature/humidity (Belgium),19735rows,
10-minute intervals. Publisher metadata/ZIP/unmodified CSV hashes in sources.json.
CC BY4.0 as declared by UCI (publisher metadata/page preserved). Original ZIP/CSV
remain unmodified; trained model/derived records are distinct artifacts.
Exclude rv1/rv2 (publisher-added random variables) and six airport weather fields
T_out,Press_mm_hg,RH_out,Windspeed,Visibility,Tdewpoint (merged weather availability
not proven causal). Keep20directly logged energy/temperature/humidity channels:
Appliances,lights,T1,RH_1,...,T9,RH_9. Preserve chronological row order, reject
missing/nonfinite values or unexpected cadence. Timestamp timezone unspecified.
Train first70%=int(.7N), test final20%=int(.2N), remaining middle validation.
Global mean/std fit only TRAIN, float64 estimation and float32 tensors; std floor1e-5.
Local normalization uses96observed values only. Inputs end before forecast origin;
all future targets stay inside their role. Histories may cross role boundaries.
No future covariates. Four full output sequences H96/192/336/720, all20channels.
Report energy-only (Appliances/lights) MSE/MAE separately from all-channel means.
Only this source's final20% is prospectively reserved for ME037; no claim that the
public dataset is globally unexposed to all researchers or foundation pretraining.

## Build and complete training

Four arms: tac_horizon, context_horizon, gru_horizon, ar_adapt. Native arm keeps
ME036 fast32/slow32 recurrence over12ordered8-step patches and adds a separate
96->42->64 nonlinear full-window encoder. The concatenated128coordinates plus
16explicit context features feed144->128 SwiGLU,64->32 basis and32->H readout.
Matched context control uses96->60->128 encoder; matched GRU replaces native memory
with GRU34->64. All three have active parameter gaps<=1%, tested before fitting.
AR-only adaptation is a deliberately unmatched practical reference, no dummy padding.

Forecast = mu+sigma*(AR(z)+alpha_h(c)*tanh(residual_h)).
alpha_h=sigmoid(k(c)*(log(tau(c))-log(h))), tau in[96,384], k in[2,6].
Initial tau256,k4. Gates are monotone in output lead for each fixed window;
tanh prevents a rescaled head from arbitrarily cancelling the attenuation.
For h>=384, |forecast-AR forecast|<=sigma/(1+(h/384)^2), in exact arithmetic.
This is an amplitude bound, NOT a no-regression guarantee or a proved failure cause.

Warm start every arm's AR identically from the same validation-selected ME036 AR
checkpoint for that dataset/H/base seed36001/36002. Original pretraining used1000
updates,4000windows and six validation endpoints; charge all original pretraining
CPU/exposure and record exact source checkpoint hash. No ME036 native/context
weights are imported. New household AR parents train from scratch with the same
1000-step schedule and six endpoints:25/50/100/250/500/1000; first100lr.001 then
cosine.001->.0001,AdamW wd1e-4,batch4,clip1,seed+1000 sampler,120CPU-second cap.

Each new arm receives500joint adaptation updates, batch4, new seeds37001/37002,
identical sampler seed+2000. AdamW lrAR.0001, otherweights.001,wd1e-4,clip1.
Loss=global standardized forecast MSE+10*mean-square AR parameter drift
(weight and bias means summed)+.01*mean-square applied correction in those units.
All learned components remain trainable; zero-initialized final correction weights
make initial outputs bit-identical to parent AR. First-step body gradients can be
zero; require finite active gradients subsequently. Step0 is an audit-only baseline,
not eligible for candidate selection or native predictive-value credit.

Save100/250/500endpoints with optimizer/sampler/model RNG/full learning trace,
validation stride24 and exact2window forecast probes. Select lowest validation MSE
per arm/seed among those3positive endpoints (earliest tie). No hyperparameter search
or replacement fit. Cap each500-step fit at120process-CPU seconds, single thread;
up to2concurrent workers within container2CPU quota. Preserve partials/failures.
6datasets*4H*4arms*2seeds=192adaptation fits;8new household AR pretraining fits.
Legacy parents reused only after hash verification, with costs charged per pipeline.

## Scoring, gates and claim limits

Freeze all parent and adaptation selections/hashes before final scoring. One scoring
pass per selected model, every eligible test origin (stride1), all future steps,
float64 error aggregation. Resume at individual model receipt boundaries if needed;
never reselect/refit based on scored outcomes. Collect per-channel metrics and
2first validation-window probes per endpoint. Persistence is an unmatched reference.
Historical ME036 scores are context only: parameters/exposure/pretraining differ.
No new foundation-model training or invocation; existing PR300 records are preserved.

Practical development gate: native macro MSE over the20legacy cells at least2%
below ar_adapt, and ETTh1/ETTh2 H720 no more than2% worse than ar_adapt, two seeds
averaged per cell. Prospective new-house gate, reported separately: four-H macro
at least2% below ar_adapt and H720 no more than2% worse; energy-only results must
be shown even if all-channel gate passes. These are engineering gates, not
statistical superiority tests; two fitted seeds/single household do not establish
population reliability. Never combine legacy and new-site outcomes as fresh evidence.

Parameter/data/exposure/selection/evaluation parity recorded per arm. Actual CPU
spend differs, so **no architectural superiority claim**, even if a score improves.
No causal/action/controller/world-model or commercial qualification. Window-dependent
normalization still requires full-window encoding; no constant-time or microsecond
streaming claim. A future action-conditioned study requires genuine observed actions,
costs/outcomes and its own registration, not invented simulator evidence.

All old1680checkpoints remain immutable. New same-CPU saved forecast replay must
be bit exact, including initial float32 AR equality; no retrospective tolerance
relaxation. Any new linear tolerance derives prospectively from unit roundoff and
gamma_n as in ME036; this study does not alter the old ME035B precision failure.
Export physical-unit native models; verify standalone loading and archive restoration.
