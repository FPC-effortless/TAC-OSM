# ME027 — learned-state value and two resource lanes

Register source/contract before fitting. Twelve fits, four fixed models × three
original seeds. No pilots, tuning, early stopping, best-checkpoint selection,
extra seeds, retraining or new test access. This is real historical development
research, not independent validation. Native fast/slow recurrence stays intact.

Restore the authenticated original ME022 snapshot after verifying its twelve
checkpoint hashes, exact snapshot hash and canonical tensor/entity/time
fingerprints. The official publisher KuaiRec archive (Zenodo 18164998, CC BY
4.0, Gao et al. CIKM 2022, DOI 10.1145/3511808.3557220) has SHA-256 in the
contract and MD5 `261550d472c48eff4990fb13c0e5bcf7`. Exactly 10,776 FIT windows
before 2020-08-01 and 5,952 already-exposed DEV windows on August 1–13 from
user_id%16=0; each has 128 genuine prior events. Original missing/invalid
exclusions and ordering persist. Other cohorts and August 14+ outcomes remain
closed. No source download, simulated training or independent evidence claim.

All models receive the same raw channels and the same explicit causal features
at every prediction boundary: current item embedding (8), FIT-standardized
current duration (1), normalized feedback EWMAs at half-lives 2/8/16/32/64 (5),
FIT-only smoothed current-item popularity (1), mean prior-item embeddings (8)
and mean observed metadata (3): 26 features. Prior32 is the identical additive
logit offset, not an answer-bearing current response. Population/preprocessing
fit uses FIT only; training item popularity is in-sample and may overfit.
All models use each of the four real next-response targets with identical
weights. MLP has a learned embedding and 252-unit tanh feature head; it receives
all raw channels through these summaries but has no learned order-sensitive
state. TAC, a standard 34-unit GRU+projection, and the fixed two-layer/two-head
16-wide causal Transformer add learned 64-wide state features to the same 26.
Trainable counts 135067/135125/134907/135057 are within 1% of 135050. Publish
nonembedding counts too; the shared 128000 embedding coefficients dominate.
This common feature interface is a diagnostic setting, not promotion of ME025's
failed kernel head or replacement of the established ME024 RMS candidate.

## Resource accounting and fixed selection

Same AdamW body/head rates .001/.006, decay .0001, clip 1, batch 128, one CPU
thread, sampler seed+300 and sample-stream prefix per training seed. Allocate
the SAME 50,000-step and 180 charged-process-CPU-second ceilings to every fit.
Stop once BOTH step 300 and CPU 20 have been reached. Do not stop based on DEV.
Record exposure checkpoints at exactly 50/150/300 steps and CPU checkpoints at
the first completed step reaching 5/10/20 charged CPU seconds. Keep all six
budget checkpoints (aliases may share a step), optimizer/RNG and original
probabilities. Trace 0/1/25/50/100/150/200/250/300 and CPU-budget crossings.

Charged CPU includes identical common FIT preprocessing cost, model creation,
script compilation and optimizer setup, then sampling, feature computation,
forward/backward, clipping and optimizer steps. Common source authentication
and data restoration are reported separately. DEV monitoring/checkpoint I/O,
post-fit probes and analysis are excluded from training CPU but included in
whole-study CPU/wall. All conventions and times are reported. Evaluation may
warm shared graphs; all arms have the same registered monitoring privileges.

The exposure lane has equal steps/examples/targets but different compute.
The CPU lane has matched charged CPU (primary 20s, overshoot at most 1%) but
different examples/updates. Mark data/exposure parity FAIL in that lane, and
actual compute parity FAIL/UNESTABLISHED in the exposure lane. Full FLOP/energy
and end-to-end study-cost equivalence remain unestablished. No result earns
strict architecture superiority. Post-fit profiler passes at the step-300
checkpoint, first 8 real FIT windows, seed 14001 only, record matrix FLOPs and
unsupported/opaque operations as a LOWER BOUND, never full compute parity.
No optimizer step occurs in profiling and no weights are changed.

## Primary hypothesis and falsifiers

Primary is TAC−feature-MLP Brier at the registered first 20s checkpoint.
Engineering candidate gate: every paired seed improves; mean delta <=−.0005;
TAC post-RMS FIT-feature-mean replacement worsens Brier by >=.0002 each seed;
mean ECE no more than .002 worse; all primary arms' CPU spends in [20,20.2].
This is a learned-state development diagnostic, not a causal mediation test.
Do not rename a secondary GRU/Transformer or step-300 contrast as primary.

For three learned-state arms at 20s only, replace their final 64 post-RMS
features by the mean of the first 1024 FIT windows, preserving common features
and prior. It preserves FIT-average linear latent logit, not an in-distribution
state or causal identification. MLP has no such state; report not applicable.
Report Brier/logloss/AUC/ECE and paired seed distributions at every budget;
1000 user-cluster bootstrap draws, seed27027, for the primary loss contrast,
conditional on these three training seeds and reused DEV. Confidence intervals
cannot undo development reuse. Include prior32 and FIT prevalence as unmatched
nontrained reality checks. Report per-checkpoint batch128 inference latency,
serialized bytes, buffers and recurrent/working storage separately.

Nonzero-head causality, deliberate unmasked-attention failure, exact original
TAC equations, standard GRU equality, actual parameter counts, shared features,
active finite gradients, mean-feature logit preservation and serialization are
software gates before fitting. Retain negatives, partial runs and all failures;
never adjust tolerances or compute gates after seeing scores. Draft PR only.
