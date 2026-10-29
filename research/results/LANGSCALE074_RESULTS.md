# LANGSCALE074 — parameter-matched 500K language TAC upper recurrent state reallocation: wider upper memory loses to h48

**Source evidence:** [draft TAC PR #613](https://github.com/FPC-effortless/TAC-OSM/pull/613); [original nine new CPU fit + source/optimizer replay run #40878240759](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40878240759); [full verified per-seed original aggregate artifact #12899027025](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40878240759/artifacts/12899027025). Nine new source-authentic original checkpoint jobs and final nine-checkpoint/three-h48 historical aggregate **PASS**, meaning the experimental protocol completed and replayable evidence was retained, **NOT** that wider recurrence quality improved. [PNDS companion draft #72](https://github.com/FPC-effortless/pnds-lean/pull/72), [Lean CI #40878192425](https://github.com/FPC-effortless/pnds-lean/actions/runs/40878192425) **SUCCESS**, limited pure parameter arithmetic only.

## Control and original sources

- Original source PSF CPython 3.13.0 licensed 14 FIT document owners (13+1 UTF8 supplement) and four original previously exposed DEV owners. 160 fixed AdamW steps per model, 8 identical source-authenticated per-seed windows × 128 real next-byte targets = **163840 source FIT labels/fit**, seeds 1701/1702/1703, exposed DEV 4096 genuine byte targets/model. No hyperparameter selection.
- Three newly fitted variants upper state h24, h96, h128, complete model/optimizer/Python NumPy PyTorch RNG source order SHA checkpoints, cold DEV replay, actual parameter counts verified. Original source-matched h48 graph-free upper checkpoint/seed evidence from [LANGSCALE070 run #40871491842](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40871491842) [source-verified original 9-model aggregate #12896077841](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40871491842/artifacts/12896077841), **reused**, not retrained. Exact original sample order SHA must match old h48 for all new arms; aggregator verified every SHA and counts.
- The first-stage TAC native fast/slow and **8×8 associative state**, aligned, wave and CED concepts are **unchanged**. The graph-free *upper recurrent* state is reallocated with the final SwiGLU FFN to preserve useful learned parameter count: source-derived model `P(h,w)=91145+4h²+564h+818w`. No unused/dummy weight padding; all new learned upper gradients verified in tests. Live count gap across four arms <0.02%, but exact training FLOPs differ and were **not matched**.

## Actual observed mean source exposed DEV at 160 updates (three seeds)

| Upper h | SwiGLU FFN w | Active trainable params | Upper FP32 state per owner | DEV mean NLL ↓ | DEV top1 ↑ | Measured mean CPU process fit sec |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 480 | 499625 | 96B | 2.319428790 | 38.0371% | 21.0107 |
| **48** (original h48 070) | **455** | **499623** | **192B** | **2.317014445** | **38.1348%** | 21.5024 |
| 96 | 388 | 499537 | 384B | 2.333355140 | 37.8174% | 24.2016 |
| 128 | 331 | 499631 | 512B | 2.352750889 | 37.2884% | 22.2158 |

All three newly trained variant arms had original checkpoints and full score cold replay, with original h48 reference source parity.

Paired NLL deltas **new h minus original h48** (positive = worse):
- h24: mean **+0.002414345**, seed1701 **+0.000608800**, seed1702 **+0.014509737**, seed1703 **−0.007875502**. h24 better 1/3 seeds only; mixed.
- h96: mean **+0.016340695**, seed1701 **+0.025271024**, seed1702 **+0.005455823**, seed1703 **+0.018295236**. **Worse all three seeds**.
- h128: mean **+0.035736444**, seed1701 **+0.035210351**, seed1702 **+0.029908797**, seed1703 **+0.042090184**. **Worse all three seeds**.

## Interpretation and open gates

This is a **reproducible negative finding for this specific parameter reallocation**: increasing upper graph-free recurrence width from48 to96/128, while shrinking FFN to maintain ~500K learned parameters, does **not improve** the genuine publisher exposed DEV next-byte loss in a 160-update pilot; may consume FFN capacity for unproductive wider state. It does **not prove that upper recurrence is intrinsically ineffective**, nor does it isolate width because FFN changes too. It says nothing definite about the original native 8×8 associative memory or long-lived streaming retention, both unchanged/not exercised as carried state. This is NOT a model scaling law, architecture superiority comparison or fresh independently held-out quality claim. DEV was already exposed. Training FLOPs not matched; CPU process time is an efficiency observation and not a parity gate. Three seeds and a short 160-step run limit inferential strength. No protected TEST, no new GPU/Kaggle minutes, no promotion, no merge.

Lean/Python exact Nat/source parameter math in PNDS [#72](https://github.com/FPC-effortless/pnds-lean/pull/72) CI PASS only for `P(h,w)`, state counts and native8 unchanged; entire original PyTorch computation graph, floating-point/AD, serial scan, source/semantics correspondence and reciprocal version registry remain **OPEN**, not an end-to-end mathematical proof.

**Decision:** preserve the actual negative result and original 070–073 experiments; do not prematurely promote h96 or h128. The next primary scientific question is whether expanding the **native first-stage fast/slow/associative memory** on genuinely unexposed independent source gives measurable useful delayed-history retrieval at matched total weights and resource budgets. Require source/license/owner/time boundary, long-history and truly carried state tests, explicit position policy trained for L256, matched SSM/GRU and Transformer only if model-param/data/FLOP parity can be verified; no further blind hidden-width increases.
