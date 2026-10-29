# LANGSCALE074 — ~500K parameter-matched upper recurrent-state reallocation

The current known problem is that model FFN scaling (070) improves authentic exposed DEV NLL whereas reset-window history above 32 bytes (071) degrades, largely due to positional extrapolation at unseen 128..255 (073 inference-only source diagnostic). **This study does not re-run those metrics for promotion.** It asks whether a larger learned upper recurrence is worth more than a larger final readout at a fixed ~500K total live model budget.

Exact original-source derived parameter law (not a whole-model proof):
`P(h,w) = 91145 + 4h^2 + 564h + 818w` where h is upper recurrent dimension and w is FFN hidden dimension. The first-stage native fast/slow/8x8 associative/2 aligned/wave/6 concept memory is held *entirely unchanged* from original CONTEXT026. Only graph-free upper recurrence and downstream FFN are reallocated. Every new learned weight must be active and receive finite gradients. The maximum count gap across new trained arms and original h48 is (499631-499537)/499631 <0.02%, well below the 1% parameter match.

| Arm | Upper h | FFN w | Original source trained params | Upper FP32 recurrent bytes/owner | Source |
|---|---:|---:|---:|---:|---|
| h24 | 24 | 480 | 499625 | 96 | new 074 fit |
| h48 | 48 | 455 | 499623 | 192 | original *unmodified* 070 p500k checkpoint, re-use |
| h96 | 96 | 388 | 499537 | 384 | new 074 fit |
| h128 | 128 | 331 | 499631 | 512 | new 074 fit |

Original 070 model uses the same upper h48 recurrence/FFN455 and first-stage base. Reuse all 3 original trained 070 h48 weights via immutable source & checkpoint SHA instead of spending extra compute to retrain, and compare source exposure and seed sampling SHA exactly. New h24/h96/h128 x seeds 1701/1702/1703 = **nine** genuine CPU fits.

## Frozen protocol before first fit

- Original PSF CPython3.13.0 publisher-source licensed 14 FIT original owners and four previously exposed DEV owners, with original source receipt/hash, 129-byte same-document next-byte windows. Same seed-index sampler from original 070 `iter_batches`, 160 steps, batch8, 128 authentic next-byte targets per step = 163840 FIT targets per run. Reset per owner; no future label leakage.
- Every new 074 arm must live count model/optimizer parameters and prove the full first-stage base unchanged before training. Epoch/train NLL, exact source sample and byte hashes, optimizer, RNG, checkpoints and postfit cold reload verification required; original exposed DEV score has 4096 targets and is visited once after frozen original fit. No seed/hyperparameter postselection, same settings as 070. Save per-run real CPU process/wall/RSS.
- Null and negative controls: original h48 trained 070 full source-validated five-context checkpoint reference; source-matched genuine 070 evaluation; constructed unit tests verify h48 new *untrained constructor* has matching original 070 state_dict source and identical forward calculation with shared initialization; h24/h96/h128 need causal prefix and finite gradients through trained upper read/observe and FFN. Compare paired seed NLL and accuracy, not only averages. Preserve any nonmonotone state reallocation losses.
- Source scope no protected TEST, no Kaggle/GPU, no protected stage, no automatic model promotion/merging. CPU GitHub bounded. The 500K live count training waiver is restricted by hard validator to exact LANGSCALE074 study/contract and exact three new counts; all other 100K/previous narrow 500K policies remain unchanged.

## Interpretation

Equal approximately500K parameter counts, original publisher data, source sample order, optimizer steps and evaluation conditions, but **NOT equal exact training FLOPs**; wider h increases quadratic upper recurrence compute and may move training cost significantly. This is a **parameter-reallocation comparison**, because FFN width decreases as h rises; it cannot isolate a pure causal upper-state gain. Further, this is not scaling the original first-stage TAC native fast/slow/8×8 associative memory or verifying carried persistent information. A significant NLL gain from h128 would justify a next-source memory-capacity study but not itself prove long-term recall. A loss or null result is equally informative.

Primary reported quality: mean and paired 3 seed exposed DEV NLL (nats/byte), top1, Brier, CPU process training time, model parameter count and upper-state FP32 bytes, plus original source/optimizer checkpoint integrity. Source DEV already exposed, so no independent qualification/general scaling law or world-model conclusion. PNDS Lean/Python operator count math required; full original graph IEEE/AD source proof OPEN, and reciprocal version registry integration OPEN.
