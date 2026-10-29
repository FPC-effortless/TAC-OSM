# LANGSTREAM043 — measured same-model incremental execution

[Draft572](https://github.com/FPC-effortless/TAC-OSM/pull/572),
[original CPU run40818307547](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40818307547)
PASS at0145381abb5743538540dc35754c970f466c890a. [Raw evidence12873875810](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40818307547/artifacts/12873875810),
ZIP SHA25622eecaf43aeeca9b9772b789581cfa73289b29472fafde241f92ce0284753304,
downloaded/hash-audited locally. Original040 graph-free48D499623p weights,
optimizer/RNG/source schedules and source file identities checked; zero fitting.

## Parity gate before performance claims

300 authentic FIT prefix cases:5 original seeds ×2 precisions ×2 clock modes
×(14 original publisher owners at33 bytes +first owner at513 bytes). All logits
match the original full-prefix graph and all nine state tensors match a separately
written PNDS functional token step within the preregistered tolerances. Independent
reference imports no TAC code, helpers or traces; shared PyTorch primitives remain.

| Precision | Max original-prefix logit error | Max independent-step logit/state error |
|---|---:|---:|
|FP32|1.0491e-5|6.6758e-6|
|FP64|4.441e-14|1.599e-14|

Frozen FP32 atol3e-5/rtol1e-4 and FP64 atol2e-10/rtol2e-8 passed without
relaxation. Bulk prefill/state reconstruction, unchanged caller state/weights,
source/owner rejection and bitwise snapshot continuation were checked. Exact
original-vs-stream greedy sequences passed for all measured repetitions.
This is bounded numerical evidence, not universal IEEE or ranking equivalence.

## Actual CPU timings

AMD EPYC7763 runner exposing4 logical CPUs, PyTorch2.10.0+cpu,2 threads,
FP32,batch1. Original trained seed1701; first genuine FIT document. Prompt lengths
32/128/512,16 greedy feedback/update bytes,2 warmups+7 measured repeats per
profile/path, alternating path order. Both paths use IDENTICAL growing history;
these are not the cropped/reset128 fallback semantics of the separate571 pilot.

Native positional profile median per-update latency:

| Prompt bytes | Full-prefix recomputation ms | Incremental ms | Decode speedup |
|---|---:|---:|---:|
|32|4.5530|2.1878|2.08×|
|128|7.2457|2.1946|3.30×|
|512|17.2753|2.2195|7.78×|

042 cap_all profile, preserving its corrected positional semantics:

| Prompt bytes | Full-prefix ms | Incremental ms | Decode speedup |
|---|---:|---:|---:|
|32|4.7517|2.2809|2.08×|
|128|7.4082|2.2763|3.25×|
|512|17.1370|2.2760|7.53×|

Cap_all512 per-update p95:17.6472→2.3402ms. Raw per-update samples are preserved
in the JSON; repetitions on one hosted CPU are not device/population qualification.

**Prefill is slower:** native512 original16.94ms vs incremental bulk21.26ms;
cap_all51216.86→21.10ms. Extra state recovery costs work. Nevertheless measured
prefill+16 NN-update totals improve: native512294.34→57.08ms,cap_all512290.75→
57.47ms. Engine verification/initialization is separately measured at about2.3ms
and excluded from those totals; adding it still leaves the measured benefit.
Greedy argmax/caller I/O are outside individual NN-update timing windows.
No warmed KV-cached Transformer, BDH, SSM, P3 or mobile benchmark was run here;
do not use old571 numbers to claim a cross-architecture winner.

## Memory, mathematics and scope

Retained per-owner tensor state is976FP values:720 recurrent/context coordinates
plus256 cached logits. FP32 payload3904bytes; source/version/owner/clock metadata,
model weights, caller input/output, source cache, transient workspace and runtime
are separate. Original weight tensors1998492bytes≈1.906MiB. Whole-job peak RSS
≈383.02MiB includes runtime, source/checkpoint loading, parity and benchmark work;
not per-state/device RAM. Bulk prefill transient memory still scales with prompt.
There is no stored entire token history in the returned inference state.

[PNDS66 run40817966756](https://github.com/FPC-effortless/pnds-lean/actions/runs/40817966756)
PASS atcf14fea287c949aa5aea52f2123d65cce87e0aa4,6 full-graph scoped axiom checks
without sorryAx. [Proof12872947970](https://github.com/FPC-effortless/pnds-lean/actions/runs/40817966756/artifacts/12872947970),
SHA2569c1b322a3cc2d54e30809ab6504e160b7c8d856c11baa11dc3a7d2fbf4f06ce2.
General Lean also PASS. Exact-real affine composition is not an IEEE
reassociation theorem. Complete source-extracted proof, CED uniqueness bridge,
formal AD and independent qualification remain OPEN.

Local1349PASS/4SKIP; dedicated25 software checks PASS; initial import-name
collision retained/fixed before trained scoring, no equation/tolerance relaxation.
A completed-run guard prevents accidental repeat execution on report changes;
workflow_dispatch retains exact-protocol replay. Source evaluation/read-only
runtime stays separate from independent qualification. No new weights, NLL or
history-recovery improvement, GPU, protected TEST, promotion, merge/deployment.
Original025/035–042 and PR483/H64 failures preserved. This closes the missing
incremental inference API for this source-specific model; it does not establish
useful long-history intelligence or competitive mobile performance.
