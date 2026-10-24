# ESWAM007 — preregistered real multi-agent selective-memory intelligence test

Status: DESIGN / NO TRAINED RESULTS. This protocol follows source-qualified ESWAM005 and authentic fixed-prior ESWAM006, preserving all negative results.

## Research question
Does TAC-OSM's native exchange-structured, identity-addressed memory improve genuine held-out vehicle future-position prediction over a matched recurrent and causal-attention control, *especially* when 128–256 real observations are available, without losing the strong instantaneous-velocity baseline?

## Source and integrity
Use the SHA-256 pinned original INTERACTION TC_BGR_Intersection_VA recorded tracks, unchanged raw observations and native timestamp/frame/track identities. FIT recordings 000/001/002; exposed DEVELOPMENT recording 003. These are recordings from the SAME intersection, not an independent-site TEST. A separate location, never inspected for model selection, is required before external-generalization claims. All source histories must be strictly consecutive, with real co-observed actor identities; missing rows must not be filled with synthetic trajectories. Audit the number of actual >=256-frame *simultaneously visible* 2- and 4-agent windows before any training.

## Prediction problem
At each eligible original frame t, encode observed x/y, vx/vy, measured source time gap, actor identity and relative observed neighbor position/velocity for all actually present actors; never use future identity, future motion, or any label-derived future feature. Predict each selected real actor's actual x/y at +10 and +30 original frames. Report Euclidean L2 in meters, x/y MAE, moving/stationary stratification, actor-grouped uncertainty and per-episode reliability. Compare direct physical forecast AND residual forecast around the measured instantaneous-velocity prior, with identical residual scaffold available to every neural family.

## Candidate architectures and compulsory fairness
A: Native ES-WAM exchange-structured fast/slow persistent memory with identity-keyed reads and collision-safe updates.
B: Parameter-matched GRU with identical source observations and physical prior.
C: Parameter-matched causal attention Transformer with identical source observations and physical prior.
D: ES-WAM with key/collision protection disabled, other components retained and parameter-matched, to isolate the proposed exclusion effect.
E: ES-WAM with long-history read disabled (8-frame effective context), parameter-matched, to measure memory contribution.
Parameter counts within 1%; same original FIT source recordings, same eligible source windows and order, seeds, batch, optimizer, update count, and target/normalization. Explicitly measure actual per-step training FLOPs/total FLOPs and document any mismatch; no fair architecture-superiority claim unless comparable. CPU inference time may differ; it is NOT a compulsory equal-time gate. No model is allowed a privileged future observation or a privileged source-derived feature.

## Required untrained references
P0 last-position persistence, P1 instantaneous measured velocity, P8 average last8 actual velocities, P256 average last256 velocities. ESWAM006 previously found P1 strongest on 876/788 real DEV cases, and P256 significantly worse than P8. Preserve these negative results. Avoid retrospectively tuning the same exposed DEV cases; prospectively freeze split, hyperparameters and stop rule.

## Success criteria, frozen before fitting
1. Real +10 and +30 future position L2 must improve over P1, evaluated separately, with paired original-actor-cluster bootstrap 95% interval excluding zero on a predesignated primary horizon.
2. The native model must outperform matched GRU and causal attention at the primary horizon with a paired source-actor confidence interval excluding zero before claiming a family-specific advantage.
3. Native long-history versus native eight-frame ablation must improve actual source outcomes without increasing stationary error, and identity-collision ablation must demonstrate a measurable effect. Otherwise no associative-memory benefit claim.
4. Forecast calibration and real motion/stationary quality must be reported; movement-only improvement does not qualify a full model upgrade.
5. All negative outcomes, source manifests, actual trained checkpoints, optimizer/RNG state, original raw case predictions and independent no-fit replay are archived.

## Restrictions
This observational traffic source cannot prove do(action), intervention effects, decision costs, autonomous robot safety, customer ROI or causal policy intelligence. Synthetic key-recall is allowed only as a diagnostic, never real-world qualification. ESWAM004 remains failed for stationary/calibration despite descriptive moving gains. No code/model has been fitted by this protocol commit.
