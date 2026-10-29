# LANGMEMORY042 — positional extrapolation contributes to the long-prefix loss

**Complete internal diagnostic, no new training or protected TEST.**
[Original CPU run40802141789](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40802141789)
PASS at7936e6dccf5e00be1115485a49561a8c24d91f1a. [Full receipt12865849418](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40802141789/artifacts/12865849418),
ZIP SHA256 d2f6dc3c5ceb3df73ad447a0f2a2c4eb99bae0e5b52d4999db53e2c06831f9da,
also downloaded and hash-checked locally. Original five040 checkpoint archives,
receipts, model/optimizer/RNG/source schedules and all041 native results replayed;
unchanged weights/files and bitwise128-position identity passed. Source bytes,
ordinal memory identities and basis-context denominators were not altered.

## Frozen two-trial loop

Preregistration17eb4ec794f60fc72ca035f198fbfe8aec54f555 preceded implementation
and evaluation. Freeze evaluator/source/checkpoints; edit only candidate maps;
log every trial; choose on FIT before confirmation. This is a bounded adaptation
of Karpathy autoresearch, not unlimited GPU search or deletion of failed trials.
The report-closure workflow guard prevents accidental repeated batch execution;
explicit workflow_dispatch retains exact-protocol replay. Each later hypothesis
needs its own preregistration, not tuning this benchmark after confirmation.

All14 authenticated PSF CPython FIT owners ×4 original target offsets =56 targets
per seed. Same five original trained499623p graph-free48D models, no optimizer.
512-byte FIT selection means:

| Position map | FIT NLL ↓ | Outcome |
|---|---:|---|
| Native positional features | 2.400549 | Original reference |
| Cap only log feature at log128 | 2.350724 | Preserved, not selected |
| Cap all seven features to phi127 beyond trained clock | **2.244639** | Selected by frozen FIT rule |

The selection JSON was written/hashed before DEV inference. Both candidates had
fixed definitions and the threshold/seed-sign rule was preregistered. FIT is
training-exposed data, not an independently held-out calibration cohort.

## Already-exposed DEV confirmation

Exactly the original04132 target offsets (four documents ×8 targets), same
weights/seeds and genuine/donor histories with last64 observed bytes unchanged.
These are32 distinct target events, not160 independent data examples.

| Observed prefix | Native NLL | FIT-selected NLL | Native BPB | Selected BPB |
|---|---:|---:|---:|---:|
| 128 | 2.007742 | 2.007742 | 2.896560 | 2.896560 |
| 256 | 2.227038 | 2.031763 | 3.212936 | 2.931214 |
| 512 | 2.204320 | 2.031324 | 3.180162 | 2.930582 |

512-byte NLL improved in5/5 seeds, mean−0.172996 nats (about15.9% lower
perplexity on this small exposed cohort). Top1 stayed45.625% averaged across
seeds at512; individual seeds changed, but the aggregate did not. At256,4/5
seed NLLs improved; mean top1 rose41.875%→46.25%. These are internal exploratory
probability/accuracy measurements, not fresh-generalization or intelligence claims.

**Useful long-history information is still not established.** Selected512 NLL
is still worse than its128 counterpart. Selected donor-minus-genuine NLL is
+0.003909 at512, with genuine history winning only1/5 seeds; four seeds favor
the donor. The mean is dominated by one seed. Better out-of-range positional
behavior does not establish reliable old-fact retrieval. New dual-memory,
algebraic verification/reconstruction is NOT implemented by this experiment.
No P3/SSM/Transformer quality-cost benchmark was run on this32-target cohort.
Earlier P3 superiority belongs to its original paired4096-target protocol.

## Costs, proof and qualification

Measured model weight tensors:1998492 bytes =1.906MiB FP32. Observed peak RSS of
the ENTIRE benchmark process:503263232 bytes ≈479.95MiB, including PyTorch,
all five loaded models, original artifacts and scan workspaces. This is not
single-model/device peak RAM, and does not show a memory advantage. CPU process
times are descriptive, ordered native-before-selected, not a controlled latency
or mobile energy comparison. Retained raw source/cache/workspace costs cannot
be removed from future dual-memory accounting.

[PNDS65 run40801483401](https://github.com/FPC-effortless/pnds-lean/actions/runs/40801483401)
PASS: five scoped feature-map identities/idempotence/prefix properties, no
sorryAx, independent Python reference comparison atFP64<=1e-12. [Proof12865777665](https://github.com/FPC-effortless/pnds-lean/actions/runs/40801483401/artifacts/12865777665),
SHA256e554fc69e94a38ef600a20ee932fbd291d1686578c63ecd062c53066afa9a3f4.
Complete frozen-model Lean/source/IEEE/AD bridge remains OPEN. Local suite
1342PASS/4SKIP; dedicated21 software tests PASS; all original-head TAC checks
completed successfully (three unrelated historical jobs skipped).

No new weights, native memory capability, independent qualification, GPU,
checkpoint promotion, merge/deployment or customer-hosted inference. Original
025/035–041/H64 negative results remain. The next capability experiment should
retain this controlled positional factor and demonstrate real retrieval beyond
simple equal-information baselines with complete data/memory/cost accounting.
