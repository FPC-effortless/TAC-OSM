# REAL-MAINT-002 — fresh-recording, persistence-start baseline qualification

REGISTER BEFORE FITTING. New user-authorized question; original REAL-MAINT-001
failures, source and checkpoints remain unchanged. Existing MetroPT-3 test targets
March2020 onward were opened; they cannot be sealed again. Per-channel reanalysis
there is explicitly retrospective and cannot qualify a new candidate.

Fresh source: MetroPT2.csv, Zenodo record7766691 DOI10.5281/zenodo.7766691,
authors Veloso/Gama/Ribeiro/Pereira, CC BY4.0,1209663443bytes, publisher MD5
056e04fd1874eaef4697d6b6657502c3. Prior to registration only metadata and first
16KiB (Apr28 training rows) were inspected; no model or held-out sensor values.
Download complete file and verify MD5 plus independent SHA256 before preprocessing.
Do not substitute dataset_train.csv or the 2020 recording. One operational APU,
not an independent fleet or causal repair trial. Published failure dates are
metadata; do not use failure labels in fitting or source-specific thresholding.

Fixed eight analog channels, schema-selected: TP2,TP3,H1,DV_pressure,Reservoirs,
Oil_temperature,Flowmeter,Motor_current. GPS and digital channels excluded before
outcomes, not based on performance. Ten-minute trailing bins; >=300 source rows/bin
(nominal1Hz); contiguous24-bin contexts predict the next ten-minute bin at hourly
target endpoints. No imputation across gaps; only past observations enter context.
Source timestamps are local-naive. Bin availability is the right endpoint.

Train targets before2022-06-01; validation June1 throughJune14; sealed A June15
throughJune30; sealed B July1 onward. Preprocessing may stream/count/hash all bins
but never prints held-out values or metrics. Store B separately; fitting/audit A
must not load B. A is opened only after the complete20-seed matrix, selection and
source/checkpoint manifests freeze. B remains unopened even after A results; any
later advanced architecture needs a separate committed protocol before B opens.
Sealing is an operational access/provenance guard, not an independent data custodian.

First-stage2x2 factorial: linear vs two-layer MLP(width32), each static normalization
vs non-affine RevIN. All output last_observed + predicted_delta * scale. Zero-init
final head makes the initial model exactly persistence. RevIN statistics use only
the current24-bin context; static moments use only training contexts. No analytical
physical dynamics or repair policy. Fixed persistence and train-mean references.
No GRU/TAC/SSM superiority comparisons in this baseline-qualification stage.

Training objective: mean channel absolute error divided by each channel's TRAIN
forecast-opportunity persistence MAE. If scale<=1e-6, use1 for optimization only;
report MASE as undefined, retaining raw MAE and the all-channel gate. AdamW wd1e-4,
gradclip1,batch64,512updates,EMA.99. Save raw/EMA and all curves. Validation every64
updates; choose trained checkpoints64..512 by minimum worst-channel relative MAE,
then mean MASE, then earlier update. The initial zero head is a baseline diagnostic,
NOT a candidate that can count as learned improvement.

Pilot seed93000: LR.0003/.001 per arm, identical updates; select LR by the same
validation criterion then smaller LR. Freeze selected LRs before twenty fresh
seeds93001..93020. Same sample draws per paired seed. Predeclare export93001, not
best outcome. Float32 fitting; float64 CPU/thread1 evaluation. Two workers; total
new fitting/preprocessing/selection cap20CPUmin; group30CPU seconds, no extensions.
If the complete matrix cannot fit, preserve INCOMPLETE and leave A/B closed. Smoke
namespace10093001 synthetic data is excluded and retained separately.

MASE denominator is TRAIN persistence MAE, not test data; persistence test MASE is
not necessarily1. Also report raw MAE and test model/persistence MAE ratio per
channel, retaining old NMSE as descriptive where applicable. Zero test persistence
MAE gives ratio1 on exact ties, infinity on positive model error. Never epsilon-floor
undefined test scales into a favorable score. Include per-channel source units.

Primary gate: selected arm must have>=18/20 validation seed passes (all channels
MAE<=persistence). After A opens, all eight analog channels must be no worse than
persistence with a one-sided simultaneous day-block bootstrap upper bound<=0 on
raw MAE differences (2000resamples, Bonferroni8). Average across20 fitted seeds for
each paired day; also report every seed's channel results/ranges, no fleet or
training-population claim. Practical improvement requires>=5% mean relative-MAE
reduction plus a paired day-bootstrap interval excluding zero. No initial zero
checkpoint or per-channel fallback can satisfy learned improvement.

Only if all-channel and practical gates pass may advanced memory comparisons be
registered against sealed B. If gates fail, retain NO_PROMOTION, keep B sealed and
choose a new registered question rather than tuning against A. Record real CPU,
parameter/state bytes, data preparation and all model work; physical DRAM/FLOPs
UNKNOWN without counters. Forecasting alone cannot qualify repair decisions:
actual inspection/action prices and independent outcomes remain separate gates.
