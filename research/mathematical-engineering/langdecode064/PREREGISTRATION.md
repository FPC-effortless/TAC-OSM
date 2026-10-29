# LANGDECODE064 — original-trained capped positional feature cache, no-fit source engineering

## Hypothesis, fixed before original CPU result

Original `IncrementalTAC043._advance` executes `transform(phi_position(tt,dtype),position_mode)` for every token. Original `langmemory042/candidate.py` uses `cap_all`: if `log1p(t) > log1p(127)`, it substitutes an exact, source-computed seven-coordinate vector evaluated at clock127. For every integer clock >=128, the same seven values are recomputed with trigonometric operations, tensor stacks and conditional dispatch; no trained model input changes.

The experimental subclass `CachedCapPositionTAC064` precomputes exactly `transform(phi_position(torch.tensor([128]),original_dtype),"cap_all")` **once during construction**. Only for `position_mode=="cap_all" and t>=128` it expands this source-computed, detached 1x7 tensor across stream owners, then calls the identical original `m.pos` and subsequent arithmetic. For native mode and t<=127, unchanged original path is preserved. Engine cache is 7 FP32 (28 byte) values per process, **not persistent per-owner state**, and does not alter snapshot, weights, parameters, clocks, masks or causal output.

This is NOT a change to the pretrained position law or a novel architecture; it requires no model fitting and no protected TEST.

## Falsifiable acceptance, before benchmarking

1. **Constructed invariants:** original `phi_position` and `transform` bitwise identical cached features at clocks128/129/512/8192/65536 in FP32 and FP64; boundary127 unchanged; both native/cap_all modes, two masked owner streams and exact snapshots. No model weight, owner, mask, clock or public method alteration.
2. **Original checkpoint replay:** unchanged seed1701 499623p FIT320 checkpoint, source/optimizer/RNG SHA, licensed original CPython3.13 FIT owner prefixes128 and512, same original greedy16 tokens. Compare candidate/original *every* incremental logit and all nine persistent state fields bitwise; original full-prefix logits <=3e-5 absolute and frozen tolerance, matching argmax; original 3904B FP32 per owner, snapshots both directions.
3. **CPU speed:** original validated `BatchedValidationTAC057` versus independent `CachedCapPositionTAC064`, 2 warmups + 7 alternating paired measured repeat arms, FP32 eager batch1 Torch intraop2/interop1, growing-sequence 16 token decode with full original guards. Require **>=5% median decode reduction AND p95 ratio <=1.10 on BOTH 128B and512B**. Report prefill, validation and unfiltered raw samples; **no adoption** if either speed gate fails, and no retuning on the same FIT source.
4. If numerically or performance negative: save all evidence and keep research PR draft/unmerged; never claim model language/energy/latency superiority from isolated math, one CPU host or successful Actions status. Any positive remains internally measured and still requires independent host repeatability, PNDS Lean/Python source-pinned proof/IEEE/AD mapping before release.

Previous negatives [LANGDECODE062 PR #601](https://github.com/FPC-effortless/TAC-OSM/pull/601) (median 4.49/4.35%, tail failure) and [LANGDECODE063 PR #602](https://github.com/FPC-effortless/TAC-OSM/pull/602) (median 1.51/0.74%) are immutable historical results. They are NOT mixed with this candidate: compare only the original unchanged 057 decoder baseline.

**Permitted compute:** GitHub Actions CPU 0 training / 0 Kaggle / 0 protected TEST / 0 new data. Source-CI tests and preflight required. PNDS reciprocal Lean + Python full formalization and IEEE/AD remain OPEN until separate review, not a claim of proven full graph.
