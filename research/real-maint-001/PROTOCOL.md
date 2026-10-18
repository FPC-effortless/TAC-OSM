# REAL-MAINT-001 — external operational sensor forecasting

REGISTER BEFORE preprocessing outcomes or fitting. Independent external-data lane;
not a replacement for DEV063 retention and not a repair-policy validation.

Primary source: MetroPT-3, UCI DOI10.24432/C5VW3R, CC BY4.0, real operational train
compressor sensors. Complete source zip218381995bytes SHA256
aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a,
CRC verified. Preserve source URL, hash, schema and acquisition script; don't ship
partial downloads. Four company-reported failure intervals are evaluation labels
only, not training features. The report's May29/30 event says maintenance30Apr;
retain this inconsistency, never silently correct it or infer verified recovery.
No intervention choice set, prices or independent repair-success labels exist.
Commercial causal benefit remains UNKNOWN, not zero cost or 100% repair.

Question: does learned temporal/CDL/neural-CASM prediction transfer from an initial
operating period to later real sensor dynamics better than retrained controls?
Inputs are fifteen measured sensors, excluding index and timestamps as learned
features. Deterministic ten-minute *trailing-bin means* are a declared preprocessing
prior. Never average future values into a current feature. Windows24 bins (past4h)
predict the next observed ten-minute bin, endpoints once per hour. Require complete
contiguous bins; don't impute through outages. Fit normalization only Feb1–21;
train targets beforeFeb22, validation targets Feb22–29, untouched test March1 onward.
Context may precede a target split; targets never overlap splits. The initial
month is an assumed reference period, not independently certified fault-free.

Four parameter-matched arms within5%: temporal GRU + soft CDL + dense neural CASM,
retrained no-CASM, plain GRU, ordered MLP. Same raw windows/targets and train-only
normalization, Gaussian mean/variance heads and unweighted Gaussian NLL; declared
log-variance bound[-6,6]. No fault labels, teacher or physical rules in training.
EMA.99, AdamW wd1e-4, clip1, batch64,512updates. Twenty fresh initializations
92701–92720, pilot92700 rates.001/.003, choose each arm by validation NLL only,
then smaller LR; save all pilots, curves and raw/EMA weights. Export firstseed92701.
New fitting cap20CPUmin, group60CPU seconds, two workers/one thread. Incomplete
matrix forbids test opening. No extension or test-based fitting changes.

Evaluate float64 on untouched later windows: per-feature MSE, normalized MSE and
Gaussian NLL, actual latency/memory, paired20seed cost of prediction vs GRU/MLP/
no-CASM with bootstrap2000/Holm3. Persistence (last observation) and training mean
are strong fixed references. Claim forecast advantage only if paired-seed MSE
interval upper<0 against persistence AND GRU/MLP with corrected significance;
this is conditional on this one machine, not independent fleet replication.

Anomaly diagnostic only: squared standardized prediction residual, threshold99th
validation percentile fixed before tests; report four failure windows' observed
detection and24h pre-onset alerts plus out-of-window alert rate. Outside reported
windows is UNKNOWN health, not an independently verified negative. No statistically
powered physical-reliability claim from four events or twenty training seeds.
Do not use test intervals for thresholding/model selection. No fabricated repair
actions, economic savings, sensor inspection costs or counterfactual outcomes.
