# ME038 — innovation memory coupled to native TAC dynamics

Registration before any ME038 capability fit, numeric parsing of Tetouan records
or held-out scoring. Architecture-first construction; no component-attribution
gate. Keep all ME034 stopping rules, ME037 failed household gate and original
artifacts intact. Do not reopen the household test or closed KuaiRec cohorts.

## Architecture and scope

A proposed new TAC variant, not a claim of globally novel primitives: native
fast32/slow32 states interact with an addressable8x8 matrix. Normalized queries
read the matrix into BOTH fast and slow updates. A normalized key and bounded
value write the discrepancy between the proposed value and the retained matrix's
current prediction. Native gates, residual slow update and timescale initialization
remain; this is a change in the state transition, not only the forecast head.
Shared contemporaneously observed channel context enters tokens and all neural
arms. Preserve ME037 joint AR, zero-initialized SwiGLU correction, normalization,
monotone horizon attenuation and bounded tanh. Full96-step replay remains needed.

Memory write: M'=rho*M + beta*(v-rho*M*k)*k^T, ||k||<=1, beta in(0,1),
rho=.95+.0499*sigmoid(r), v=tanh(W_v[f;s]). Its fixed-signal matrix operator
has norm<=rho. This is NOT a full coupled-system Jacobian contraction proof.
Fast/slow states remain in[-1,1] from zero initialization. Matrix state adds64
float32 coordinates: total128coordinates/channel, twice ME037. It is not a free
capacity improvement. Delta-rule/associative-memory ideas are established;
originality here is the implemented TAC coupling, not first invention of them.

Four arms innovation_tac/context_cross/gru_cross/ar_adapt. All neural arms see the
same normalized channel history and across-channel mean history, with identical
static context, head/gate and AR. Candidate context192->26->64; memory9562weights.
Context-only192->50->128. GRU16->44->64 with context192->20->64. Active parameter
counts for native/context/GRU at H96/192/336/720:
49462/49332/49422;61942/61812/61902;80662/80532/80622;130582/130452/130542.
Gaps<1%, no dummy padding. AR is an unmatched practical reference; persistence
is compulsory. Previously trained ME037 and compact-modern baseline results are
historical context only, not new parameter/compute-matched opponents. No new
TimesFM invocation or published/SOTA comparison is claimed.

## Authentic sources and roles

Legacy ETTh1/ETTh2/ETTm1/Weather/Electricity: original ME036 CSVs, hashes, channel
order, standard chronological split and train-only normalization unchanged. All
are exposed development evidence. ETT resolutions share equipment. Keep Weather
sentinels/duplicates/cadence caveats disclosed; no source cleaning based on tests.

New UCI849 Power Consumption of Tetouan City, DOI10.24432/C5B034, Abdulwahed Salam
and Abdelaaziz El Hibaoui. Publisher ZIP, API and license page SHA/bytes in
sources.json; CC BY4.0 publisher page link checked. No numeric records parsed
before this registration. Original CSV preserved, all8numeric measurement columns
in original order; DateTime used only for chronology/cadence verification.
Forecast all8, with power-only LAST3columns the primary application metric.
Weather fields are past recorded context only; no future weather/calendar inputs.
Three distribution zones within ONE city, not three independent datasets/sites.
Power units are as published; no assumed Wh/kW conversion or measured savings.
Publisher collection description is the provenance, not independent instrument
authentication. API instance count may include header; record actual row count.

Verify finite values/no missing, monotone600-second cadence; reject anomalies
rather than silently impute. Parse timestamp with M/D/YYYY H:MM. Split by original
row order: train int(.7N), validation until N-int(.2N), final20% reserved test.
Mean/std population estimates float64 on train only, tensors float32, std floor1e-5.
Window-local statistics use only96observations ending before prediction origin.
All targets lie within the role; input history may precede its start. Source
record dates and actual split indices recorded after registration. This is a new
same-city temporal test after training on its past, NOT zero-shot/multi-city transfer.
A repository search found no prior Tetouan study; this is scoped to recorded
repository exposure, not global absence of public-pretraining overlap.

Beijing UCI501 was screened by metadata/header only, then rejected for fresh-test
qualification because ME013 already used it. No Beijing fit or numeric outcome
analysis in ME038. Its downloaded bytes are separate unused source records.

## Complete fits and resource lanes

Seeds38001/38002, corresponding AR-parent seeds36001/36002. Five legacy sources
reuse exactly ME037's validation-selected AR-only parents (no native/context
weights), all six parent endpoint hashes checked. Charge full1000-update/4000-window
parent exposure and historical training/selection CPU to each pipeline; deduplicate
cached cost when reporting actually executed work. Eight new Tetouan AR parents
train1000updates, batch4, AdamW wd1e-4, clip1, lr.001 first100then cosine to.0001,
sampler parentseed+1000; save25/50/100/250/500/1000, validation stride24,
select lowest validation MSE earliest tie, cap120training CPU seconds.

Exposure lane:6sources*4H*4arms*2seeds=192fits, exactly500joint updates, batch4,
max120training CPU seconds. Save100/250/500; select lowest validation MSE among
positive endpoints. Step0 audit-only, never selectable.

Compute-budget lane: ETTh1/Weather/Tetouan only, H96/720, four arms/two seeds,
48separate fits. Same initial weights and sample-stream prefix as exposure lane.
Budget5process-CPU seconds for model construction/initial equality audit and
optimizer work; finish the update crossing the deadline, record the overshoot
and last-update CPU. Max10000updates; never pad with sleeps or dummy work.
Save at first crossed1/3,2/3,fullbudget; three validation opportunities, same
lowest-validation selection rule. Record endpoints, actual CPU, updates and
examples. If the update ceiling prevents spending the budget, report it; no
claim of equal actual spend. CPU timing is environment specific, not bit-exact.
Model-independent worker/optimizer import startup and validation/serialization
are separately measured, excluded from the5s optimization budget and INCLUDED
in total-cost reporting. Thus this is a matched training-CPU-ceiling lane with
unequal data exposure, NOT full end-to-end architecture parity. Never pool its
metrics with the exposure lane or conceal additional examples of cheaper arms.

Both lanes train all learned paths jointly from the identical AR parent;
AdamW lrAR.0001, otherweights.001, wd1e-4, clip1. Identical per-seed sampler
seed+2000 across arms. Loss=global-standardized full-sequence MSE +10*mean-square
AR parameter drift (weight/bias means summed)+.01*applied correction square mean.
First zero-output step may yield zero body gradients; require active finite
memory gradients afterwards. All attempted fits/partials/failures retained,
no replacement, tuning search or post-test architecture change. Two workers,
single-thread each, same CPU/quota; no paid compute. No synthetic capability fits.

## Frozen scoring, thresholds and integrity

Freeze ALL selected parent/new checkpoint/source hashes before any final scoring.
Full sequences H96/192/336/720, every eligible test origin stride1, float64 error
aggregation after float32 prediction subtraction; MSE/MAE per channel. Report
power-only and all-channel Tetouan scores separately, and both lanes separately.
Resume only at completed per-model score receipt boundaries; do not reselect.
Persistence all sources/H; preserve all seeds, failures and unfavorable results.

Exposure lane development target: native macro MSE>=2% below AR and>=1% below
matched context across20legacy cells. Prospective Tetouan PRIMARY power-only
four-H macro>=2% below AR and>=1% below context; H720 no more than2% worse than AR.
All-channel result also shown, without substituting for the power gate.
Compute lane power-only two-H macro uses the same margin criteria, separately.
These are engineering targets, NOT significance claims. Two seeds and one city
cannot establish population reliability. Current prospective test becomes exposed
after scoring; do not retune on it or relabel it as fresh in a later study.

Report parameter/data/exposure/compute/selection/evaluation PASS/FAIL per lane/arm.
Exposure lane actual CPU differs; compute lane examples differ and end-to-end
cost differs: NO ARCHITECTURAL SUPERIORITY CLAIM in either, regardless of scores.
Report profiler supported-FLOP subtotal and unsupported operators honestly;
no full FLOP/energy parity. No mechanism attribution, independent physical
control, VoI, causal world-model, commercial or microsecond streaming claim.

Bit-exact same-CPU validation-probe replay across ALL parent/new endpoints,
optimizer/RNG clocks and source-hash verification. Export selected native models
with raw-input normalization; standalone loading and archive restoration without
refitting. Record test-score checks separately from saved-probe replay: do not
imply that all aggregate test scalars were stored/replayed. No historical tolerance
relaxation; ME035B exception and all earlier failed gates remain unchanged.
