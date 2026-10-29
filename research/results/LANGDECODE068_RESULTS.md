# LANGDECODE066–068: capped positional-feature source, integrity and CPU repeatability

## LANGDECODE066 — actual original source numerical audit
- [TAC draft PR #605](https://github.com/FPC-effortless/TAC-OSM/pull/605) / [Actions #40867905441](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40867905441).
- 11 focused tests passed; **65,553 checked positions in FP32 and another 65,553 in FP64** (dense t0..65535 plus 17 extreme source clocks) exactly matched cached/original positional source bitwise. 16 high-clock original-vs-064 constructed masked two-owner recurrent state/logit transitions per precision bitwise. Local positional projection weight gradient wrt original and cached features bitwise.
- **Negative integrity finding:** original `_check_model` failed to detect a deliberate ordinary `add_` to cached 7D tensor (both dtypes). Original trained checkpoint was NOT loaded, no real long-history claim. Complete IEEE/AD/source theorem OPEN.

## LANGDECODE067 — guarded original-checkpoint trained decoder CPU result
- [TAC draft PR #606](https://github.com/FPC-effortless/TAC-OSM/pull/606) / [Actions #40868438062](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40868438062).
- Separate `VersionGuardedCapTAC067` inherits identical original 064 `_advance` (source math unchanged). Only an O(1) identity/tensor-version/shape/dtype/device guard on cached position tensor was added. It detects normal PyTorch in-place edit and cache replacement but NOT adversarial .data/raw-storage bypass.
- Original seed1701/FIT320 499623p checkpoint, licensed PSF CPython3.13 original FIT owner prefixes128/512, same greedy16, bitwise all nine persistent state tensors and incremental logits, original 3904 FP32 bytes/owner, full prefix max abs errors **6.67572021484375e-6 / 5.7220458984375e-6** within frozen tolerance. 56 targeted tests passed, snapshots bidirectional.
- Single-host original-vs-guarded median decode gain **6.5023% / 6.0027%**, p95 candidate/original **0.94938 / 0.92862**. Preregistered gate PASS on first host only.

## LANGDECODE068 — three clean runner repeatability: PASS
- [TAC draft PR #607](https://github.com/FPC-effortless/TAC-OSM/pull/607); [three independent GitHub CPU jobs #40868726001](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40868726001), complete aggregate `REPLICATION_PASS_ON_GITHUB_CPU_ONLY`. These three are **guarded 067**, NOT unguarded 064. Each ran the unmodified original 067 benchmark, identical source/checkpoint/greedy16, two warmups/seven paired alternating repeats, FP32 eager batch1, 2 intraop and 1 interop.

| Fresh hosted CPU job | 128B median improvement | 128B p95 ratio | 512B median improvement | 512B p95 ratio |
|---|---:|---:|---:|---:|
| 1 | 6.3405% | 0.92260 | 6.5495% | 0.91425 |
| 2 | 6.2515% | 0.94716 | 5.7707% | 0.91571 |
| 3 | 6.2702% | 0.93864 | 6.7454% | 0.85806 |

All **three** separate clean jobs individually passed BOTH 5% median and 1.10 p95 gates. 56 focused tests each passed, bitwise source original-vs-guarded logits and 9 state tensors, 3904B FP32 state, original source full prefix within tolerance, unchanged optimizer/source/RNG checkpoint, 0 training/GPU/TEST. No cherry-picked host/prompt.

**Boundaries:** Hosted x64 CPU source evidence only. Complete source-to-PNDS IEEE/AD and cross-repo full variant contract remain OPEN. `Tensor._version` is not cryptographic tamper protection. No actual ARM/Android/mobile energy or deployment result yet; original model quality untouched. All PRs draft/unmerged. Prior LANGDECODE062 and 063 speed failures retained. No trained checkpoint promoted or model weights altered.
