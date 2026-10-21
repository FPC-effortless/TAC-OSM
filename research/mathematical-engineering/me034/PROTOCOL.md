# ME034 — Stable-input native recurrence, with a once-only new-user stop rule

Registered 2026-10-21 before any ME034 fitting or user-remainder-6 outcome
interpretation. Research only; no merge, promotion or commercial qualification.

## Question and evidence boundary

Can original native fast/slow dynamics learn a useful prediction correction when
the R-pretrained input embedding is immutable? ME033 established an adaptation
penalty under one recipe; input-coordinate drift is an unproved mechanism, not
an isolated root cause. Overfitting and learning-rate mismatch remain alternatives.

Existing ME031/032/033 pretraining is already rotation-specific and R-user-only.
Audit its original source, role sets, parent hashes, embedding equality, R-only
duration normalization and item popularity before each fit. Do not describe this
as rerunning pretraining or as proof against every possible form of leakage.

## Authentic source, roles and final population

KuaiRec, Gao et al., CIKM2022 DOI10.1145/3511808.3557220, CC BY4.0,
official Zenodo18164998 `KuaiRec.zip`, `big_matrix.csv`.
ZIP SHA256 c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c,
MD5 261550d472c48eff4990fb13c0e5bcf7.
Original ME022 snapshot SHA256
fea9675380d501e81b740a0c1e0cd8c99893863ba6bed9a7e6619a572cc21177.
Training uses only the original 10,776 FIT histories /376 users, user%16=0,
target before 2020-08-01, exactly128 real preceding events, watch_ratio>=1
binary targets. Missing/invalid/negative/nonfinite measurements excluded as in
the original parser; duration>0, 0<=item<16000, timestamp>1e9. Metadata remains
log1p(duration)/12, observed feedback, historical gap, including original gap
resets at positions0/96. No fabricated labels or padded histories.

Reuse three entity-disjoint roles `(uid//16)%3`: R=r,S=(r+1)%3,T=(r+2)%3,
all three rotations and seeds14001/2/3. Inherit only original R-only embedding,
R-only common head/context normalization/popularity. R's native300 and common500
historical steps are charged to every arm, never described as new physical fits.
S trains downstream models; T supplies already-used FIT cross-validation only.
Freeze ALL63 neural models and9 trees before ANY new T metric. No DEV fitting,
scoring, selection or access to previously opened user1–5 evaluation cohorts.

Final: exact user%16=6, targets 2020-08-14 inclusive to2020-09-06 exclusive,
source-only two-pass deterministic stride, maximum6000 windows, minimum1500
windows and100 users. All prior observations of these users before the end date
may supply causal128-event histories. Source download/hash does not constitute
model training. Skip other users before interpreting outcomes; skip end-date
records before outcomes. No replacement cohort or outcome-based exclusions.
The committed remote-ref/reader inventory finds past KuaiRec exposure only in
remainders0–5 across331 remote refs/593 unique scripts at inspection time.
This is repository-record provenance, not proof against undocumented activity.
Remainder6 is now consumed by this study and can never again be called unopened.
No final label is parsed until all model payloads are hashed and verified.
Create an access receipt before parsing; one registered final pass, later
replays only. No retraining, tuning, changed arm or checkpoint after final access.

For final prediction, average the three rotation models' probabilities within
each training seed. Report each seed separately and mean metrics across seeds.
No best rotation/seed/model selection. This is a new-user and later-target test,
not a new independent dataset. FIT pretraining ends before final target dates,
but later observed feedback inside a final user's causal history is legitimate.

## Locked complete arm matrix

All native arms preserve the original32+32 recurrence and split RMS epsilon.01.
Learned memory correction is unchanged ME032 centered nonlinear readout:
delta(h,C)=w'h+v'[tanh(Bh+AC+a)-tanh(AC+a)], zero at h=0,1260 active coefficients.
Common C has the identical26 R-normalized causal channels and prior32 logit.
The inherited R common head stays frozen except common_adapt.

| Arm | Active downstream parameters | Treatment |
|---|---:|---|
| native_frozen |8236| R embedding frozen, original dynamics+centered readout trained |
| native_adapt |136236| identical initial embedding/dynamics/readout; embedding also trained |
| history_frozen |8772| ME033 frozen-input pooled11->56->56->64 correction |
| common_adapt |1205| ME033 common-head adaptation reference |
| gru_frozen |8244| learned11->24 event encoder, standard GRUCell24->36, fixed injective36->64 map, identical centered readout |
| ewma_mlp |225| ME021's12 causal rate features and16-unit tanh MLP, evaluated/trained at all four real prefixes |
| current_only |517| trained10->43->1 head: current R-item embedding8,duration,popularity; R-global prior, no personal history |
| boosted | non-neural | HistGradientBoostingClassifier on common26+prior logit; unmatched practical reference |

Every student embedding starts at the same original R table. Native frozen and
adapted start with identical body/correction weights. GRU's fixed output map is
a seeded64x36 orthonormal-column matrix, buffer2304 values, not learned capacity
or padding; its recurrent state has36 coordinates, native64. All6984 GRU body
coefficients are active in event processing. Native body6976. Parameter gap
native_frozen/GRU is0.0972%, within1%. Native frozen/pooled gap exceeds1%.
Freezing128000 coefficients deliberately changes active parameter count; never
call freeze/adapt or native/common parameter-matched. Count stored weights,
buffers and historical pretraining separately, including duplicate R embeddings.

Per-window dense forward memory MACs: native128*6848=876544;
GRU128*(11*24+3*36*(24+36))+4*36*64=872448, gap0.4673%.
This is a dense arithmetic ledger, not full FLOPs. Record warmed B8 four-target
forward/backward profiler operator subtotals before fitting and after replay,
with2% target tolerance for the native/GRU subtotal. Unsupported fused,
elementwise, nonlinear, optimizer and context costs prevent full FLOP parity.
Actual CPU spend remains independently reported. No full architecture-superiority
claim unless every standing resource gate passes; failure cannot be waived.

EWMA features: Laplace prior32, Laplace all-prefix prior, seven normalized
exponential means half-lives2/4/8/16/32/64/128, R-only item popularity, known
offered duration and mean recent32 historical gap. Their normalization is R-only.
Current-only uses current features from the same R context but excludes personal
history and its prior; this is a trained no-history reference, not zero masking.
The common head already sees pooled embedding history and five EWMAs. Determinism
of an extra representation given C differs from determinism of all states given
the full raw history. Order-invariance of pooled predictors cannot prove that
the data have no temporal signal.

## Fixed optimization and resource accounting

63 neural fits=7arms*3seeds*3rotations, each exactly300updates, batch128,
four actual targets after32/64/96/128 events, weights1/6,1/6,1/6,1/2.
All neural arms receive identical sampled S indices and153600 response
presentations /38400 histories. Seed=14001/2/3+rotation*1000+68000;
sampler=seed+300; correction seed separatelyseed+1. Rotate execution order.
AdamW memory.001, heads.003, decay1e-4, active-gradient clip1. No search,
early stopping or repeated attempts. Frozen tensors absent from optimizer.
Save step100/200/300 latest partial checkpoints, Adam/RNG, every step loss,
CPU, memory gradient and embedding gradient; original final scripted runtimes.

9 trees: sklearn1.8.0 HistGradientBoostingClassifier,100 iterations, lr.05,
15max leaves, min20 samples/leaf, L2=1, early_stopping=False, same cell seed.
Fit all S four-prefix rows with the registered prefix weights. Tree exposure
and parameters are unmatched, explicitly separate practical evidence.

One CPU thread, hardware/software captured,180CPU sec S-stage and300charged
pipeline CPU sec ceilings for every arm. Record actual spend, shared original
R300+teacher500 charges, new context/feature preparation, peak RSS, inference
latency and deployed tensor bytes. Serialization excluded from charged training
steps; new physical whole-study CPU/wall separately measured. Equal caps are
not equal spend. Source verification is a shared audit cost, reported separately.
No paid/external training, automatic CI fitting, synthetic primary evidence,
causal action utility, maintenance ROI or independent physical outcome claim.

## One final decision and registered stop

Primary native_frozen versus common_adapt, final-user cohort only. Report Brier,
logloss,AUC,ECE10 for all arms, per seed and per model. Paired user-clustered
bootstrap3000draws, seed34034. Compute Brier loss differences averaged across
the three fitted seeds and mean seed-level AUC differences. Resample users,
keep all their windows and all fitted seeds together. Two Bonferroni97.5%
two-sided intervals jointly control familywise error<=.05 across these two
registered endpoints, conditional on the fixed fits/population sample.

PASS iff upper Brier interval<-0.001, lower AUC interval>0, both differences
have favorable direction in all three seeds, and native ECE does not worsen
more than0.003. This measures incremental predictive utility against a smaller
practical control; it is not a parameter-matched superiority claim. Other
contrasts are descriptive exploratory comparisons, without unadjusted
significance declarations. Do not report any CI as uncertainty over retraining.

If ANY primary gate fails, close this frozen-input native KuaiRec recipe and
retire positive native-memory claims on this task. Preserve all results and
weights. Do not add gates/hierarchy/latent computation or retune on remainder6.
Further work requires a genuinely new task/objective/registered dataset rationale.
Failure is not proof that every future recurrent architecture must fail.
If gates pass, keep research-only status pending complete parity and independent
corroboration. No architecture promotion or merge is authorized by this protocol.

After fitting, record sequence-only shuffle diagnostics with common context held
fixed, fixed permutation seed34034 within each observed32-event block, and native
state-zero substitution. These are distribution-shift diagnostics, not calibrated
tests of whether order contains information. Report a pooled-shuffle invariance
check as architectural identity only. The separately trained current_only model
is the primary descriptive history-deletion control. No diagnostic selects models.

## Reproduction and failure preservation

Run `python -m scripts.me034_stable_research --stage fit`, then audit frozen
payloads before `--stage final`. Preserve failed/partial attempts with receipts;
never overwrite their outputs. Verify original source/roles, active parameters,
gradients/freeze behavior, optimizer clocks, sampling RNG, final checkpoint
and scripted predictions. Recompute final statistics from archived arrays.
Include fresh Torch-only raw-input export checks, corrupted-payload rejection
before source access, full regression suite and independent archive restoration.
