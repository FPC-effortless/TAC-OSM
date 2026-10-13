# DEV-008 — measured synthetic hardened-loop outcome (NEGATIVE / DEVELOPMENT)

**Status: REPLAY-VALIDATED-DEVELOPMENT-ONLY; NO POSITIVE CAPABILITY CLAIM.**

GitHub Actions development run: https://github.com/FPC-effortless/TAC-OSM/actions/runs/38347629079 . The job completed with nine focused mechanism tests passing, smoke validation and full-grid evaluation replay. Source files in this run include `scripts/run_ced_hardened_dev.py` and `src/tac_osm/development/ced_hardened_router.py`; independent evaluation replay was executed by `scripts/validate_ced_hardened_dev.py`. Raw archive: GitHub Actions artifact named `DEV-008-RAW-NOT-CONFIRMATORY` (6 checkpoints, 24 evaluation cells and 576 episodes). Previous initial run https://github.com/FPC-effortless/TAC-OSM/actions/runs/38347510446 had matching output but did not yet enforce checkpoint evaluation replay.

## Development protocol

- Three independently initialized seeds: 43, 59, 71. `shared` and `sparse` start byte-identically for each seed and see identical generated training tasks.
- 120 supervised steps per seed per arm, on 8 candidate descriptors. The training supervisor is evaluator gold, not an inference input.
- Four test conditions: N=8/32 retained candidates, temporal read delay D=0/4, 24 separate synthetic cases per condition and seed. D=4 executes all four intervening real hardened-loop steps. Repair disabled to expose raw routing quality. Staged temporal state, actual executor and verifier remain operational.
- Task is deliberately simple unique equality over two marked bits; **a deterministic matching control can select the unique valid candidate from query-visible or state-readable reference bits**. Model quality must therefore be benchmarked against this near-trivial 100%-solvable synthetic reference; good scores would not demonstrate open-world reasoning.
- No full model FLOP/memory-match, no new PLM learning or scalable learned semantic indexing. The CED encoder still runs on retained candidates only.

## Raw pre-repair exact decision accuracy

| Seed | Shared CED | Sparse CED |
|---|---:|---:|
| 43 | 27.08% | 12.50% |
| 59 | 21.88% | 13.54% |
| 71 | 81.25% | 13.54% |
| **Mean** | **43.40%** | **13.19%** |

Each seed is the mean of four equal-sized evaluation cells (96 episodes). Sparse performed poorly and shared had high initialization sensitivity. Shared-minus-sparse paired seed difference: +30.21 percentage points, exploratory 95% Student-t interval **[-50.84, +111.26] points**. With three seeds and major optimization failures, this is not a credible general architecture comparison.

### Severe training failure, not evidence that all sparse attention is ineffective

The sparse mode implements hard first-stage top-K and masks excluded candidates with `-1e6` logits. If gold is not in top-K, the loss grows dramatically and **the masked gold logit is a constant disconnected from the indexer**. It cannot directly teach the first indexer to retrieve the gold candidate. The recorded last-step losses expose instability:

| Seed | Shared last CE | Sparse last CE |
|---|---:|---:|
| 43 | 3.083 | 118.406 |
| 59 | 2.886 | 667,885.875 |
| 71 | 0.065 | 20.383 |

Thus the failed sparse arm is partly a **training-objective/information-flow defect**, not an interpretable controlled scientific verdict about a properly trained CED/CSA2 selector. Do not treat sparse's 13.19% as a valid generalized sparse-attention penalty.

## Reference attention operations

In the implemented 2-encoder-block and 2-query-only-decoder design with two heads, when N=32:
- Both encoder modes: 4,096 full causal score positions.
- Shared decoder: first 64, subsequent 64, final 64 = 192; **total 4,288**.
- Sparse decoder: first 64, subsequent 8, final 8 = 80; **total 4,176**.
- Difference 112 logical score pairs = **2.61%** of shared model total. Both use 2,048 decoder-global KV scalars at N=32, width 32; saving is in selected score pairs, **not** in global KV storage. These counts omit many other operations; they are not timings/FLOPs.
- Initial candidate encoder is O(N^2) in this small reference, and the first sparse scoring is O(N). Neither method demonstrates sublinear end-to-end addressing or meaningful hardware speedup.

## Validation and next milestone

The Actions validator reloaded all six checkpoints, checked file/source/tensor hashes and exactly regenerated 576 saved evaluation rows spanning 24 cells. This is **evaluation replay only**, not re-training or an untouched external holdout. No existing earlier experimental results were overwritten.

Next create a separate development successor that gives the sparse indexer's **first-stage selection** a valid training gradient (e.g. supervised first-stage index retrieval CE), and compare to a shared reader with matched auxiliary supervision. Report actual first-stage top-K recall, pre-repair decision quality, all loss terms, and measured cost. Only after that should future-state prediction or real bounded finite memory be integrated.
