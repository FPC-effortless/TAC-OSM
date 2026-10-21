# ME036 — patched native long-horizon forecasting

Registered before numeric parsing of the new source files or fitting. Architecture-first
bounded systems development; no architectural-superiority or published-table claim.
The old ME035/035B checkpoints, negative results and 1e-6 initial-equivalence failure
remain unchanged. No KuaiRec access. No deployment or merge.

## Sources and information

Original unmodified ETT CSVs from Zhou et al./ETDataset (CC BY-ND4.0).
Weather and Electricity are the THUML Time-Series-Library benchmark mirror,
whose dataset card declares CC BY4.0; underlying measurements are MPI Jena weather
and UCI electricity consumption. Preserve mirror attribution and distinguish its
license statement from independent verification of original instrument-data rights.
Do not redistribute those CSVs in this PR. Exact URLs, byte sizes and SHA256 in
sources.json. Original bytes were downloaded/hashed, not numerically parsed before
registration. Dates/timezones and missingness will be reported from schema checks.
ETTh1/ETTh2 have prior development/test exposure; no new independent-population claim.
No evidence of earlier ETTm1/Weather/Electricity code use was found by repository search;
this is not a guarantee of globally pristine benchmarks.

History96 steps; separately trained full output sequences H=96,192,336,720,
all measured numeric channels, no future covariates. ETTh split8640/2880/2880;
ETTm1 splits34560/11520/11520. Weather/Electricity chronological70/10/20 percent,
following TSL integer split arithmetic (train=int(.7N), test=int(.2N), rest val).
Train-only population mean/std (float64 estimation, float32 tensors, scale floor1e-5).
No missing-value imputation: nonfinite input aborts that dataset with failure recorded.
Publisher finite sentinel values remain unchanged, explicitly disclosed.
Each target sequence lies wholly in its split; history may overlap previous split.
All training starts from scratch: no ridge/pretrained warm start. No interpreting
ME035's21-output checkpoints as long-horizon forecasts.

## Models and engineering

Native TAC: ordered nonoverlapping8-step patches,12 native fast32/slow32 updates,
residual slow innovation and trainable multiscale rates, retained level/scale,
SwiGLU correction with32 learned horizon basis features and joint trainable
channel-shared96-to-H AR path. Patch processing reduces recurrence depth8-fold;
this is a new patch-time model, not exact acceleration of old hourly recurrence.
Train the integrated system; no mechanistic ablation phase.
Controls: AR-only; same AR plus context-only nonlinear branch; same AR plus GRU34
and64-coordinate projection. TAC/context/GRU counts must differ by <=1% before fits;
all parameters active, no dummy padding. Other practical baselines: channel-shared
DLinear (25-step moving average), compact PatchTST-style (patch8,stride8,d32,
2 layers,4 heads,FF64,flatten head), compact iTransformer-style (variable tokens,
96-to32 embedding,2 layers,4 heads,FF64). Independent compact implementations;
not official published/tuned configurations. Transformer arms intentionally unmatched
capacity. All models use identical available observations; channel-independent
architectures need not exploit cross-channel interactions available to iTransformer.

## Locked budget and selection

Seeds36001,36002; five datasets times four H times seven arms =280 neural fits.
100 AdamW updates each, batch4 full multivariate windows, lr.001, wd.0001,
clip gradient norm1. Deterministic identical sample indices per dataset/H/seed
(seed+1000) for every arm. Save checkpoints25/50/100 with optimizer/RNG/trace.
120 process-CPU-second training ceiling per fit; partial/error checkpoint preserved
and no replacement tuning. CPU single thread. Actual spend, RSS, data presentations,
counts and budgets reported; neither matched caps nor steps prove equal compute.
No hyperparameter search. Validation every endpoint, anchors stride24; select the
lowest validation MSE endpoint independently per fit (earliest on tie). Validation
is development data; not a sealed outcome. Freeze every selected checkpoint/hash
before final test scoring. One final scoring pass, test anchors stride1, all horizons
and channels; aggregate scalar squared/absolute errors in float64 streaming without
storing enormous prediction arrays. Persistence included as unmatched reference.
Retain2 first validation windows' predictions per endpoint for exact replay.
All fits must finish or record failure before any final test metrics are read.

## Claims and correctness

Practical success: selected TAC mean test MSE >=2% lower than AR-only and within5%
of best baseline averaged equally over20 dataset/horizon cells; both seeds shown.
This engineering threshold is not a statistical-superiority test. Negative results
remain primary records. Parameter/data/compute/selection/evaluation parity reported
per arm. Actual compute differs, modern baseline sizes differ: **no architectural
superiority claim**, no public SOTA/world-model/causal or commercial qualification.
Full benchmark split/horizon metrics do not establish paper parity: lookback96,
compact architecture,100-update training and sparse validation differ from papers.
No test-based next run in this registration.

Numerical tests use analytical identities, causal inputs, all active gradients and
shape/parameter checks. No empirical tolerance repair. New dot-product checks use
float32 unit roundoff u=eps/2, gamma_n=n*u/(1-n*u), and the forward-error bound
(gamma_n+2*u)*sum(abs(a*b)), allowing coefficient rounding prospectively.
Checkpoint replay on same CPU must be bit exact; no tolerance exception accepted.
No float64 ridge initialization in this experiment. Rolling statistics do not
make window-normalized nonlinear state reusable; full-window serving is reported.
