# ME022 — optimize the native prediction interface before adding memory

This is an exposure-matched engineering experiment on **already inspected real
development data**, not a new architecture contest or independent validation.
Registered with executable source before fitting. The machine-readable contract
is `research/mathematical-engineering/me022/contract.json`.

L0: inherit native fast/slow equations, ME018 scripted execution, authenticated
KuaiRec parsing and ME019 deterministic windows. ME020 does not establish that
forgetting causes the deficit; ME021 identifies a large practical feature/
training gap. L1: check unchanged active capacity, forward reparameterization,
nonzero finite gradients, exact matched initial state and causal source bounds.
New L2 question: does train-only normalization of the **current offered video's
duration**, and/or sixfold head learning rate, improve native learning on real
observations without changing its recurrent state equations? This effect has
not been measured in the inherited studies. No extra memory or summary module.

Use original KuaiRec Zenodo 18164998, publisher `big_matrix.csv`, measured Kuaishou
video impressions/full-watch outcomes (watch_ratio >= 1), 2020. Published archive
MD5 `261550d472c48eff4990fb13c0e5bcf7`; SHA256 pinned in the contract. Publisher metadata reports CC BY 4.0 (Zenodo API checked before fitting). Cite Gao et al., CIKM 2022, DOI 10.1145/3511808.3557220. Data reuse
remains subject to publisher terms; no redistribution of raw records. Preserve
original parser exclusions: invalid timestamp/item/duration/ratio excluded,
nonnegative user %16=0 only; later timestamps skipped before label interpretation.
No imputation, synthetic history, user reassignment or other cohort access.
Same 128 genuine preceding observations; target exposure is known before its
response. Temporal-gap reset and source-only stride inherit ME019 exactly.
Expected FIT/DEV window counts 10,776 / 5,952. All normalization fits on FIT only.
All users1–5 and other cohorts, labels from Aug14 onward, MetroPT and all sealed
tests remain closed. DEV is historical and has been inspected repeatedly.

Fixed 2×2 native matrix: original, duration normalization only, head LR×6 only,
both. All four have exactly 135,050 trainables and identical initialized weights,
same batch-index draws, 300 AdamW updates, batch128, LR .001, weight decay .0001,
global gradient clip1. Normalize only current duration by FIT mean/std (floor
1e-5); no head or state normalization, no extra features. LR×6 applies only to
existing `head.weight` and `head.bias`. All other trainables keep .001. Zero
initial head means initial logits remain the original causal prior32 for every
arm. Same model and analytic anchor in every cell. Three seeds14001–3; rotate
arm execution order across seeds to reduce systematic timing drift.

Record the declared dense step checkpoints' FIT/DEV losses, total training CPU
and wall seconds excluding evaluation/checkpoint I/O, gradients, head weights,
peak process RSS, inference latency and analytical forward affine MACs. Same
180 charged CPU-second maximum per fit and 300 update maximum: abort and retain
partial state if exceeded; no padding. Report actual compute and exposure parity
separately; equal ceilings do not prove equal spend. End-to-end elapsed includes
source building and diagnostics separately. Forward MACs exclude elementwise,
backward, embedding and optimizer operations and are not a complete FLOP audit. This lane is deliberately
exposure matched; no fully compute-matched superiority claim. No search or
score-driven refit. Preserve every result, trace, final/partial optimizer/RNG
checkpoint and event prediction, including negatives. Checkpoint final step only.

Primary contrast: combined arm minus original mean paired final DEV Brier.
Engineering candidate iff all3seed differences <0 and mean reduction >=.005.
Report separate factorial effects, logloss, AUC and descriptive ten-bin ECE.
Also show fixed causal prior32 and the ME021 225-parameter standardized causal
summary MLP (same FIT/DEV rows, 300 updates/3seeds) as **unmatched practical
references**, not an architecture ranking. Report their actual capacity/compute.
This reference's summaries are richer explicit features; disclose that difference.

Final frozen internal-state knockout zeros the existing head's fast/slow state
coefficients without retraining; report paired Brier degradation as an OOD
readout-reliance diagnostic, not a causal learned-memory contribution. No future
cohort opening or qualification follows a passing development gate. No native
architecture promotion, whole-system reliability, action causality, maintenance
utility, deployment or merge. Standing real-world/parity rules from PR237 as
reconciled in PR282 apply; this task intentionally narrows to TAC optimization.
