# LANGDECODE057 original trained TAC CPU result

Completed 2026-10-29. **Internally measured, not independently qualified.**

**Preregistered engineering acceptance: PASS.** Batching finite-state checks reduced
per-byte median decode latency by >=5% at both original prompt lengths and p95
also improved. This changes validation execution only; original neural graph,
checkpoint and 3904-byte per-owner FP32 persistent-state payload are unchanged.

[Draft PR #596](https://github.com/FPC-effortless/TAC-OSM/pull/596) ·
[successful original CPU run 40855792865](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40855792865) ·
[raw artifact 12889745522](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40855792865/artifacts/12889745522)

Frozen implementation/preregistration commit: `e53767b9427910c8dfe9e581dc56b8c928bddd4f`.
GitHub tested its merge snapshot `99f65d09ab1fea052a08fa82b078e0454f7f26ca` against
unchanged parent052 `07617627906d51000b1d9a0932f6b598fcb81554`.

## Same-host measurements

| Prompt bytes | Original median ms/byte | Batched median ms/byte | Reduction | Original p95 ms | Batched p95 ms |
|---|---:|---:|---:|---:|---:|
| 128 | 1.338244 | 1.252200 | 6.43% | 1.428414 | 1.334513 |
| 512 | 1.359946 | 1.258224 | 7.48% | 1.428435 | 1.357187 |

FP32 eager, batch1, two PyTorch CPU threads and one interop thread, same host,
original seed1701 checkpoint, same observed FIT prefix and 16 original-greedy
bytes. Two warmups plus seven measured repeats, alternating arm order; raw
latencies and setup/prefill/validation/whole-repeat measurements preserved in
`research/evidence/langdecode057/RESULTS.json`.

Validation-only median cost fell from about 0.158–0.159ms to 0.116ms per call.
Both input and output validations remain inside the per-byte timing.
Median prefill+16-byte latency fell from 28.327 to 26.920ms at 128 and
37.405 to 35.778ms at 512. These totals are medians of whole repeats.

## Replay and guards

- GitHub and local constructed regression suite: **58 passed**. Includes the
  original incremental suite and architecture-agnostic research probe checks.
- All nine state tensors and every generated-step logit: **bitwise identical**
  to original incremental execution, including bulk-prefill state.
- Original full growing-prefix logits: maximum absolute difference 6.675720e-6
  at 128 and 5.722046e-6 at 512, within frozen 3e-5/1e-4 tolerance and identical
  argmax predictions.
- Original-format snapshot transfer in both directions: next transition
  **bitwise identical**. Persistent state stays 3904 tensor bytes per owner.
- Original checkpoint SHA256 `0cbc1b89f53ea2a0fe1371a179a9ff14cc35ae84d09f76bdab1dffeac4ac86cb`
  verified with original sample-order/optimizer/RNG metadata, strict state_dict
  loading and 499623 historical parameters. Weight tensor hash unchanged after
  replay/timing. All 11 original LANGSTREAM043 SOURCE_PINS hashes also match.
- Phone planner: READY_NO_FIT / github_actions. Scientific preflight:
  PASS_DIAGNOSTIC_ONLY. Separate compulsory research parity CI passed.

## Interpretation and remaining limits

A small measured CPU engineering improvement with preserved trained predictions;
no new language capability or architecture superiority result. Only one hosted
CPU, one historical checkpoint and previously exposed authentic CPython FIT
owner `Doc/howto/unicode.rst` were used. No new fit, GPU minutes, protected TEST or
checkpoint promotion. Inherited source loader reads already-exposed DEV files
only to verify historical receipts; DEV bytes do not reach this inference run.

The new check creates a temporary 3904-byte concatenation per FP32 owner; this
is released each call. Persistent-state bytes exclude weights, Python/tensor
objects, transient activations and process memory. No RSS, CUDA VRAM or mobile
performance measurement. Uninstrumented anti-cheat/gradient dimensions remain
INCONCLUSIVE; universal IEEE/AD/source proof obligations inherited from
LANGSTREAM043 / PNDS #66 remain OPEN. No new computational graph or formulation
claim is made. Keep draft for independent review; do not merge automatically.

LANGMEMORY056 remains a preserved negative. This result does not reopen keyed
cache tuning or justify new training.
