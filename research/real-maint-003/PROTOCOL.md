# REAL-MAINT-003 — measured thermal forecasting, registered before fitting

Question: can a small learned residual forecaster beat persistence and strong
autoregressive references on physical sensor histories outside its training dates?
DEV067's synthetic decisions are saturated and its calibration gate fails. This
study addresses real predictive learning before adding memory/conditional-compute
complexity. It does not repair or reinterpret REAL-MAINT-002's failed experiment.

## Source and fixed information contract

OEDI909, DOI10.25984/1862682, Kim/Leach/Im, CC BY4.0. SHA256 of measured experiment
zip: `1e6cd3df991abf865f7567e6cc0ac246968299eb02dfe9464d887727f930dde0`.
The downloaded archive has 63 equipment CSV days, 96 records each, plus five
weather files. The published 49-scenario count is not assumed to equal the CSV
day count; some setting IDs repeat on different dates and 11 files are from2019.
This cohort's unit is explicitly a CSV experiment-day. An exact mapping to all
published scenarios is not established, and no63independent fault configurations,
machines or facilities are claimed.
Only measured FRP data is used, not the separate simulation product.

Targets are ALL ten `T_Room_{102..106,202..206}_Avg` channels, at 2 and 4 steps
(30 and 60 minutes). Inputs: eight past 15-minute observations of these channels,
`T_out_Avg`, `Net_Total_Power`, and sine/cosine of PAST minute-of-day. No fault name,
intensity, filename, calendar date/year, future weather/control, post-target
measurement, label or simulator enters a model. Target selection is a room-thermal
service scope, not a claim about every analog channel in the archive or MetroPT.

Split by published equipment filename date: training before 2018-08-01 (all
2017 plus early 2018); validation the rest of 2018; held-out test all 2019 files
(11 days). Train/validation values have not been parsed before this registration.
Source hashes, filenames, first-line schemas and line counts were inventoried;
2019 sensor values have not been parsed/scored. No weather CSV data is used.

Windows stay within one actual calendar day and CSV, with strictly contiguous
900-second steps through the furthest target. No interpolation, imputation,
value clipping or outcome-based censoring. Nonfinite selected-input/target or
noncontiguous windows are dropped with EVERY count preserved. Reject data quality
if any selected day has fewer than 32 valid windows, training fewer than 12 days,
validation fewer than 4 days or test fewer than 6 days. Never delete a channel or
move dates to rescue a result. Training scale/naive MAE must be finite and >1e-6
for every input and every target cell; otherwise preserve rejection.

## Models, references and fixed budget

Four independently fitted arms cross linear/MLP with static/RevIN normalization.
MLP has one 64-unit GELU hidden layer. All output heads start EXACTLY zero: output
is last reading plus a learned change. This is an initialization, not a persistence
fallback. No output-channel disabling, shrinkage selection, confidence gate or
post-test mixture is allowed. Static normalization uses TRAIN past-input statistics.
RevIN centers/scales each room's past eight readings (population standard deviation
floor1e-6), reverses that scaling on its predicted residual, and uses TRAIN scaling
for the two past exogenous channels. All targets remain degrees Celsius.

Loss: mean over the 20 room/horizon cells of absolute predicted-change error,
scaled by that cell's TRAIN persistence MAE. Raw target changes are formed in
float64 before float32 fitting. AdamW, weight decay1e-4, gradient clip1, batch64,
1024 updates for EVERY fitted arm. Final EMA in float64 with decay.995 is the
ONLY evaluated checkpoint; raw weights also preserved. No intermediate selection.
All arms share minibatch indices within each initialization. Inference uses
float64/thread1 and numerical replay tolerance1e-8.

References: persistence, last-minus-first drift, univariate ridge AR and an
information-matched multivariate ridge on the static arm's inputs. Both ridge
families predict changes, with unpenalized intercepts and lambda{.001,.1,10}.
All six ridge fits remain preserved. Reference selection uses validation mean
MASE, then worst relative MAE, then name, over all candidates including persistence
and drift. They receive no test data. Closed-form references use all TRAIN rows;
their different fitting compute is reported, not misrepresented as matched SGD.

Pilot initialization97300, LRs{.001,.003}, all four arms retained at both rates.
Choose LR per arm by minimum worst validation-cell relative MAE, then mean MASE,
then smaller LR. A pilot recipe is eligible only if no validation cell has point
MAE worse than persistence and mean relative MAE≤.95. If none qualifies, close
VALIDATION_REJECTED with all eight pilots and keep 2019 values unopened.

Choose the candidate BEFORE fresh fitting: best eligible final pilot by worst
relative MAE, mean MASE, parameter count, lexical arm. Then fit the ENTIRE four-arm
matrix at20 fresh initializations97301–97320, including bad arms/seeds. Candidate
must pass the same validation point gate in at least18/20 seeds before any test
parsing. Otherwise close VALIDATION_REJECTED; no alternate-arm substitution.

Two single-thread CPU workers, ≤60 CPU seconds per four-arm group, ≤1800 CPU
seconds total including preparation, references, pilots and fresh fits. A failed,
capped or incomplete required fit leaves test closed. No extra updates or paid
compute. Smoke uses synthetic tensors, 8 updates, one LR/one fresh group with
seed offset+10000000, excluded from ALL performance claims.

## Sealing and held-out gates

After all80 fits and the validation gate pass, freeze candidate/LRs and every
checkpoint SHA256. Write a permanent TEST-EXPOSURE marker BEFORE parsing any 2019
sensor value. Set `test_opened=true` immediately, including partial failures.
Only then generate test windows and evaluate all80 candidates and every reference.

Qualification requires ALL:

1. Candidate has no point-MAE regression on ANY of the 20 test cells in at least
   18/20 individual seeds, and overall mean relative MAE≤.95.
2. Simultaneous one-sided95% upper MAE difference against persistence≤0 on EVERY
   cell. Use2000 hierarchical paired bootstrap draws, resampling20 initializations
   and circular moving TWO-DAY blocks to the observed number of test days; RNG973003.
   Upper quantile1−.05/20 (Bonferroni20). Days, seeds and control pairing preserved.
3. Mean TRAIN-scaled MAE difference versus the validation-selected conventional
   reference has ordinary one-sided95% upper bound≤0 under the same bootstrap.
   This is the comparative point-forecast gate, not an architectural claim.
4. Complete source/manifest/raw/final-checkpoint/validation/test/standalone replay
   within1e-8; every seed/channel/horizon/day and failed outcome retained.
5. First predeclared seed97301, batch1, float64/thread1 median decision CPU latency
   ≤10ms after5 warmups/100 repetitions, with parameter/buffer storage reported.
   This is a one-host engineering limit, not a production or universal SLA.

Report degrees-C MAE, TRAIN-scaled MASE, persistence/drift/AR gaps, all per-seed
and per-day outcomes, learning curves, preprocessing/fitting/inference CPU and
memory. Undefined test-relative errors stay undefined; they cannot become a gain.
Small, adjacent test days and one building limit population inference despite
blocking; 20 random initializations do not constitute20 independent facilities.

## Decision and boundaries

Only ALL gates permit QUALIFIED_ROOM_FORECAST_ONLY. Any failed complete matrix
closes NO_PROMOTION; a validation rejection leaves sensor test values unopened.
No tail/metric/date/channel changes, test-winner selection, persistence-only release
disguised as learning or revival of REAL002 is allowed in this ID. A next question
requires a new source-frozen registration and genuinely untouched evaluation.

This is teacher-free observed-future point forecasting, not calibrated hidden-state
probabilities, action-conditioned repair discovery, CDL/CASM advantage, learned
indexed retrieval or a fully autonomous commercial agent. No repairs are executed.
Actual inspection/repair prices and independent field outcomes remain unqualified.
Preserve every prior study and MetroPT July B. Overall goal remains ACTIVE.

Frozen fitting/selection/gate files: this protocol and protocol.json;
`src/tac_osm/development/thermal_forecast.py`;
`scripts/data_real_maintenance_003.py`, `run_real_maintenance_003.py` and
`analyze_real_maintenance_003.py`. Commit these before real preprocessing/fitting.
Source hashes, pre-fit tests and excluded smoke runs are retained separately.
