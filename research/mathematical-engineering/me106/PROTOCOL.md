# ME106 — complete point-priority TAC training

Prospective capability-development panel motivated by exposed ME105 results.
The hypothesis is that distinct uncertainty/median learning roles plus an
asymmetric point-priority gradient rule can retain source-gap coverage while
improving ordinary and gap point accuracy with sharp intervals. No attribution
campaign, new memory-capacity claim or novel projection-theory claim.

Exactly15 fits: three fixed complete objectives ×allfive ME103 strong parents
102001..102005, new sampling seeds106001..106005. No ME105 checkpoint or favorable
seed selected. New class PointPrioritySensorWorld inherits the full ME105 forward
network,100,349 trainable/deployed parameters and33 active tensors. Allweights
train; inherited29 tensors lr3e-6, four new-head tensors lr3e-4. FreshAdamW wd.001,
clip1,1,536 updates/model, batch16, contexts8/64/256. Parent weight lineage3,072
updates plus1,536; width/median heads1,536. Two one-thread CPU workers maximum.
No retry, early stop, post-score selection or further ME106 fits. Allmodel,
optimizer/RNG/order/mask/source/gradient/resource artifacts Git-frozen BEFORE REPORT.

For P=c_point*physicalmedianMAE/FITrawstd and
U=properNLL+.1standardizedloglocationMSE+.3maskedcurrentGaussianNLL
+.25proper90intervalscore/FITrawstd, compute gP and gU independently.
When dot(gP,gU)<0 and ||gP||²>0, remove gU's component against gP:
  gU'=gU-dot(gP,gU)/||gP||²*gP; otherwise gU'=gU.
Combined gradient gP+gU' goes to the same guarded AdamW. Fixed recipes:
balanced c_point=.5; point_guard1; point_strong2. All compute same graph.
NLL/interval location view mu-(correction-correction.detach()) preserves EVERY
primal forecast value while removing the direct new median-head contribution.
The point and log-location losses retain that head. See full MATHEMATICS.md.

Projection guarantees concern exact-real raw gradient inner products ONLY.
AdamW group rates, preconditioning, clipping, decay, finite precision and finite
steps do NOT imply per-step point loss decreases. No 90% coverage theorem.
Prior art: Yu et al., Gradient Surgery for Multi-Task Learning,NeurIPS2020;
Howard/Ruder discriminative fine-tuning; Gneiting/Raftery proper scoring. This
asymmetric priority rule is not verbatim randomized symmetric PCGrad.

Authentic UCI501 CC-BY-4.0 archive SHA256
 d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8.
FIT2013–2014 only for training/statistics/augmentation; hiding actual recorded
current values does not invent labels. Same per-seed plans/order/masks/initial
weights/exposure and profiled graph/dense FLOP estimates across allrecipes.
Unsupported pointwise/optimizer/hardware FLOPs are partial. CPU is an outcome,
not a parity gate. Old/new exposure and gradient workloads differ; no controlled
architectural superiority claim. This is exposed-data capability development.

Original2015 REPORT unchanged:143 gap queries/21 calendar-week clusters,
4,288 ordinary queries/53 overlapping clusters;6 pollutants,six-hour horizon,
256-hour observed history. Seeds/repeated queries do not add independent cases.
Protected2016–2017 numerical records remain unparsed/unscored. No test tuning.

Unchanged ME105 gates: EVERYseed must pass ALL for an allfive recipe pass:
- Primary gap PM2.5 nominal90%coverage in[.90,.98].
- Gap PM2.5 MAE <=min(1.05mappedME101full256,1.05ownME103strongparent).
- Ordinary PM2.5 MAE <=min(1.05LOCF,1.05ownME103strongparent).
- Gap six-channel NLL <=min(6.234504261016846,1.05ownME103strongparent).
- Gap PM2.5 interval score <=min(1.05mappedME101, .90ownME103strongparent).
Do not round into a pass or choose favorable seeds. Allchannels/width tails,
paired conditional week bootstrap and allfailures published. Allfive required.
No independent confirmation, commercial qualification, promotion or merge.

Before optimization: live100k checks, all33 finite nonzero combined gradients,
actual no-step graph/FLOP profiles, source+registration Git freeze, reciprocal
PNDS Lean/Python/source links and numerical source/reference gradient tests.
After fitting: all15 source-vs-independent-serial FP32/FP64 outputs/state/input+
weight gradients/chunk/distributions AND new loss/projected-gradient audits.
Immutable tolerances: FP32rtol2e-5/atol2e-6, FP64rtol1e-9/atol1e-10. Raw-unit
absolute discrepancies reported separately. Original checkpoint/scoring and
clean-checkout/GitHub replay remain compulsory. Whole-neural/IEEE/calibration
and Gaussian-law refinement remain OPEN. Exactlysix families retained.
