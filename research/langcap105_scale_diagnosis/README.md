# LANGCAP105 scaling diagnosis — 2 November 2026

**Finding:** the published TAC–GRU gap is already present on seen FIT data. The source has a demonstrable temporal-capacity scaling mismatch: TAC widens its tokenwise computation without widening its contextual or associative state. This is the leading engineering hypothesis, not an experimentally established cause of the language-loss gap. Read-only source/checkpoint checks did not identify a scan, carry or disconnected-gradient defect in the tested cases. No corrected architecture was trained and no original DEV evaluation was repeated.

## Evidence identity and recovery

Original training: `ba9220163c777a7d9b6f765378810408baf00b1f`, [run 41229365510](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41229365510), [PR 685](https://github.com/FPC-effortless/TAC-OSM/pull/685). Source and frozen DEV publication recovered at `0651f2f4f186b63a164d44bb0fb7bec7053b13e4`, [PR 687](https://github.com/FPC-effortless/TAC-OSM/pull/687).

Recovered original trainer, registration, comparator factory, vendored TAC graph, complete published 18-model checkpoint audit and DEV summary. Additionally downloaded and SHA256-verified original ZIPs and final checkpoints for seed 10301: small TAC, medium TAC and medium GRU. Full 2,560-step loss receipts and optimizer payloads for those three were recovered. The other 15 raw checkpoints/full loss receipts were not downloaded in this investigation; their published audit and scores were inspected, not independently reexecuted. Original artifacts expire in January 2027. Originals remain unchanged and are not duplicated in this PR.

| Downloaded artifact ID | Configuration | ZIP SHA256 |
| --- | --- | --- |
| 13053720086 | small TAC, 10301 | 3407ad6ac892394ec1725ceff403ce2f899bc92d02c08752d47baebd82d5c216 |
| 13053797509 | medium TAC, 10301 | 0a056bf06e2453ea231c2ea392e30dbaca8be307e06ad590aa2fd038c4a05e2c |
| 13054828702 | medium GRU, 10301 | fff255e3f812985531c2ff0e82a4258ab66c36725e1216245fbb60370a55a063 |

## What actually scaled

| Model | ~1M | ~10M |
| --- | --- | --- |
| TAC | 8 layers, width 256, 1,000,028 parameters | 8 layers, width 992, 9,999,922 parameters |
| GRU | 4 layers, width 193, 997,680 parameters | 4 layers, width 634, 9,987,024 parameters |
| Transformer | 4 layers, width 128, FFN 649, 999,972 parameters | 8 layers, width 256, FFN 1889, 9,999,880 parameters |

TAC is the nonhierarchical `StackedCandidate`: byte embedding + three learned lexical lags; eight residual associative blocks; a gated BudgetHead. It is **not** a later deep-cache/SSA/BDH variant. Every block retains c=16, k=4, v=8 with two banks, at both scales. No hierarchy update gates are active. The Transformer uses eight heads, pre-norm blocks, zero dropout, sinusoidal positions, embedding scaling by sqrt(width), and a linear byte head. GRU is PyTorch's four-layer dense gated recurrent module with a linear byte head.

Both scales use AdamW, lr=0.0003, betas=(0.9,0.95), eps=1e-8, weight decay=0.1 applied to all parameters, FP32, batch 4, 128 input bytes, 2,560 updates, seeds 10301/10302/10303, two CPU threads, deterministic algorithms. There is no warmup, LR schedule or gradient clipping. Every batch starts with fresh state, including GRU; there is no cross-window persistent training or truncated-BPTT carry. Each fit sees 1,310,720 target presentations from authentic CPython FIT files; this is not a claim of distinct corpus size. Source-order coordinates match by seed. Checkpoints retained are final-only: LATEST was repeatedly overwritten and then deleted. Intermediate optimizer trajectories cannot now be inspected from milestone snapshots.

## Separating fitting from generalization

Published means across three seeds; no new predictions:

| Scale/model | Seen-FIT reference NLL | DEV NLL | DEV minus FIT |
| --- | ---: | ---: | ---: |
| 1M TAC | 1.629568 | 1.741724 | 0.112155 |
| 1M GRU | 1.634320 | 1.727076 | 0.092756 |
| 10M TAC | 1.503027 | 1.617619 | 0.114592 |
| 10M GRU | 1.235382 | 1.346757 | 0.111375 |

At 10M the FIT difference is 0.267645 nats/byte and DEV difference is 0.270862. Only about 0.003217 of that difference is the difference in DEV-minus-FIT gaps. Therefore a claim that TAC primarily suffers *more overfitting* is unsupported. Optimization, representational restriction and the chosen data regime remain confounded. Both models improve with scale; GRU improves much more. All medium seed scores favor GRU. Same-project DEV and changing file composition do not prove generalization on new projects.

The original audit's average CPU worker times were TAC 164.50s vs GRU 350.68s at 1M, and TAC 721.40s vs GRU 1626.35s at 10M (about 281.8 vs 635.3 ms/update). These are original job observations, **not** a new controlled speed benchmark: runner identities/concurrency and full FLOPs were not qualified. They show why equal updates must not be mistaken for equal compute. More TAC updates under an equal-compute budget might help; the amount is unmeasured.

## Demonstrated structural constraints

1. **Temporal state does not scale.** Each TAC block stores context 16, previous keys 8, matrices 64 and mass 8: 96 floats. Eight blocks retain 768 nonlexical floats at either parameter scale. The three-byte lexical buffer grows, so total state DOES grow (1,536 to 3,744 floats); it is wrong to say total TAC state is constant. GRU's entire recurrent state grows from 772 to 2,536 floats. State size is not equivalent to information capacity, but TAC's far-history state dimension is fixed. With the recent three byte embeddings fixed, the differentiable influence of older history must pass through those 768 coordinates; this is a state-dimension bound, not a proof of a 768-item limit.

2. **Most added weights transform the current token representation.** TAC's block `out.weight` has d*(d+42) entries. Its h-to-output d*d submatrix alone occupies 8*d*d weights: 524,288 (52.4% of total) at 1M versus 7,872,512 (78.7%) at 10M. The 10M model has 8,698,224 block parameters and 1,044,770 head parameters, but only 128 contextual coordinates and 512 matrix coordinates across layers. These dense maps can improve per-token representation and spatial interaction between layers, so they are useful rather than dummy padding. Their usefulness does not remove the memory bottleneck.

3. **Associative address/value ranks remain tiny.** Each bank is 8x4. Positive softmax keys/queries and EMA writes yield a normalized mixture of previous values; writes pair the current value with the previous key. This is an intentional transition association, not evidence of an off-by-one bug. Rank <=4 is not a hard four-association storage theorem. However, fixed address/value dimensions cannot be assumed to inherit the expressivity of the widened 992-dimensional stream.

4. **Memory writes are affine averaging, not error-correcting replacement.** Per bank, M_t=alpha_t M_(t-1)+(1-alpha_t) v_t k_(t-1)^T and z_t=alpha_t z_(t-1)+(1-alpha_t) k_(t-1); read is M_t q_t/(z_t.q_t+1e-4). Mixing and interference are plausible limitations. Their contribution to this corpus's loss has not been measured.

5. **Temporal feedback differs from a GRU.** TAC's scan coefficients/offers are computed from the current layer input and low-dimensional past context, not from a dense nonlinear transformation of that block's entire previous matrix. GRU's full recurrent vector influences reset/update/candidate computation. Width-matching or count-matching does not make those temporal function classes equivalent.

6. **Short reset training underexercises long horizons.** Bank retention is bounded to [0.97,0.995] and [0.995,0.99995], corresponding to approximate e-folding lengths 33–199 and 199–19,999 bytes. Context initialization is exactly exp(-1/H): sigmoid(-log(expm1(1/H))) is correct, not a decay bug. All training examples reset at 128 bytes; the experiment cannot validate long-horizon state use. The first write is deliberately suppressed at clock zero. The fixed 1e-4 read epsilon attenuates low-mass early reads; its real-data impact is unknown.

## Read-only engineering checks actually executed

`audit.py` loads exact original weights under torch 2.15.0+cpu, verifies five original source hashes, checks archive-derived checkpoint hashes against original receipts, checks allocations and original loss curves, and performs constructed 2x128 byte graph tests. No real corpus was accessed; there were zero optimizer updates, no labels/rewards and no DEV/TEST rescoring. These are local engineering checks, not a GitHub Actions execution or independent scientific review. The phone planner chose GitHub Actions CPU/READY_NO_FIT; that planning receipt is not an execution receipt or approval to train.

| Check | 1M TAC, 10301 | 10M TAC, 10301 |
| --- | ---: | ---: |
| Parallel/serial logit max absolute difference | 1.53e-5 | 9.54e-6 |
| Full/chunked carry max absolute difference | 1.14e-5 | 9.54e-6 |
| Parallel/serial parameter-gradient relative L2 | 3.72e-7 | 2.10e-7 |
| Future-poison change to earlier logits | 0 | 0 |
| Every fixture parameter gradient finite | PASS | PASS |
| Source and original checkpoint unchanged | PASS | PASS |

The medium GRU also passed the fixture future-prefix/entity checks and unchanged-checkpoint check. This investigation did not independently repeat all 18 checkpoint audits or all original equation proofs.

A scalar final-logit probe has lag-127/current embedding gradient-norm ratios about 0.00219 for small TAC, 0.00191 for medium TAC and 0.147 for medium GRU. This is **one artificial input and one arbitrary output projection** with different parameterizations. It motivates real-FIT lagwise Jacobian/credit assignment measurements; it does not prove vanishing training gradients or explain any fraction of the DEV gap. Existing finite-gradient checks similarly cannot rule out weak/poorly conditioned gradients or saturation. No full training gradient norms, clipping frequencies or activation distributions were recorded originally, so historical gradient instability remains unproven.

Original seed-10301 256-step loss block averages are retained in RESULTS.json. They show TAC keeps learning over much of the run but fits worse than medium GRU. They do not establish a converged asymptotic floor. No stopped-early optimizer defect was found: the published audit verifies 2,560 updates for every optimizer state in all 18 models.

## Mathematically motivated engineering changes and falsifiers

**First priority: scale temporal capacity while keeping total parameters matched.** Allocate budget to c, number of heads, k and v rather than predominantly to the dense d*d maps. Factorizing/tokenwise projection rank and moderating d can pay for larger memory without dummy padding. Freeze candidates using parameter counts only, before new validation. Hold depth fixed at eight within the TAC comparison. Use at least original TAC and a wider-state TAC, each at 1M and 10M, alongside original GRU. The diagnosis is strengthened only if matched wider-state TAC improves real-FIT learning and fresh, separately registered validation consistently across seeds, and intervention tests show the added state is used.

**Second priority: optimize the actual training regime.** Equal-budget per-family LR/warmup/decay searches; parameter-group decay for raw decay/bias/norm parameters; log preclip and postclip gradient norms, update-to-weight ratios, tanh saturation, address entropy, effective bank rank and read epsilon/mass ratios on FIT. Test shorter horizon initialization appropriate to 128-byte resets separately from longer-context carried training. Changing carry requires new contiguous file batches with correct detach and file-boundary isolation, applied fairly to GRU. Do not continue old optimizer states and call the modified recipe an original LANGCAP105 run.

**Third priority: corrective memory, informed by prior art rather than a novelty claim.** A candidate law is Mbar=alpha M_(t-1), then M_t=Mbar+beta*(v-Mbar*k)*k^T. With ||k||_2=1 and beta in [0,1], the memory-state perturbation multiplier is alpha*(I-beta*k*k^T), whose spectral norm is <=alpha. This gives a nonexpansive direct state update, NOT a guarantee about full input-dependent Jacobians, trainability or competitive language performance. Larger heads and correction are separate factors; combine only after isolated tests. This law changes the graph and scan algorithm: the existing scalar affine scan cannot be reused unchanged. It requires a source-matched PNDS Lean/Python record and verified chunkwise forward/backward implementation. Matrix-state DeltaNet and Gated DeltaNet establish relevant prior art, not evidence that this TAC modification works.

- [DeltaNet, NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/file/d13a3eae72366e61dfdc7eea82eeb685-Paper-Conference.pdf)
- [Gated DeltaNet, ICLR 2025](https://openreview.net/pdf?id=r8H7xhYPwz)
- [Mamba-2/SSD, ICML 2024](https://proceedings.mlr.press/v235/dao24a.html)

A competitive scaling claim needs: measured training FLOPs and same-exposure curves; equal compute curves as a separate analysis; <=1% count parity; matched tuning budgets and original-source access; at least three paired seeds; a credible tuned attention/SSM reference; independent source/evaluator qualification; and a fresh sealed test reserved until selection ends. The original exposed DEV remains historical evidence, not a new tuning target. Later long-context and executable-code evaluations are additional capability studies, not inferred from byte NLL.

## Status

Demonstrated: original architecture/recipe recovery, fixed temporal-state allocation, FIT-localized gap, read-only checks on three original checkpoints, reproducible source/gradient/carry tests. Not demonstrated: a unique causal attribution, a corrected model beating GRU, gradient pathology on real FIT data, saturation/interference frequency, FLOP parity, all-seed new probe replication, or competitive long-context/code capability. No historical sources, checkpoints, evaluation registration or scores were edited. New fitting remains subject to a new frozen protocol, PNDS obligations and the repository's independent-source/reward/ACL execution gates; this audit does not fabricate them.

## Reproduction

On the publication checkout, extract the three original Actions ZIPs into `originals/small-tac-10301`, `originals/medium-tac-10301`, `originals/medium-gru-10301` (each containing TRAINED.pt and FIT_RECEIPT.json). Verify the ZIP SHA256 values above before extraction, and preserve originals read-only. Run:

```bash
python research/langcap105_scale_diagnosis/audit.py --originals /absolute/path/originals --out /absolute/path/RESULTS.json
```

The script requires original source pins and refuses mismatched checkpoint hashes. No model scoring on language labels occurs. `RESULTS.json` records checks and original receipt-derived losses. Run the phone compute planner on `research/phone_queue/langcap105-scale-diagnosis.json` separately; it does not launch this script.
