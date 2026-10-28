# LANGLAYER037 — FFN-only vs recurrent TAC upper depth (authentic CPU 320×5)

**Evidence status:** COMPLETE internal exploratory **previously exposed DEV**;
no protected TEST, no FLOP-matched architectural-superiority certification.
**All ten new original FFN CPU fits and full source/seed/checkpoint aggregate
passed** in [GitHub Actions #40788869823](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40788869823).
Reference recurrent scores and original 15 checkpoints are frozen under
[#40785939365](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40785939365)
and audit [#40786118546](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40786118546).

| Architecture | Trainable/deployed params | Exposed DEV mean NLL ↓ | PPL ↓ | Top-1 ↑ | Mean CPU process seconds ↓ |
|---|---:|---:|---:|---:|---:|
| CONTEXT026 original | 500,122 | 2.099715 | 8.165 | 43.81% | **41.65** |
| Recurrent TAC depth 2 | 499,725 | **2.098918** | **8.158** | 43.81% | 57.32 |
| FFN-only TAC depth 2 | 499,661 | 2.108155 | 8.234 | 43.90% | 44.38 |
| Recurrent TAC depth 3 | 499,934 | 2.105924 | 8.215 | 43.80% | 59.82 |
| FFN-only TAC depth 3 | 499,684 | 2.116557 | 8.304 | 43.38% | 46.69 |

Original count-based P3 practical unmatched control NLL **1.877306**.
**Neither new FFN-only candidate beats the original neural model or P3.**

## Original five-seed NLL pairing — new FFN minus old recurrent

| Seed | FFN depth 2 − recurrent depth 2 | FFN depth 3 − recurrent depth 3 |
|---|---:|---:|
| 1701 | +0.007699 | +0.001216 |
| 1702 | +0.000791 | −0.000380 |
| 1703 | +0.026819 | +0.004638 |
| 1704 | +0.010519 | +0.003562 |
| 1705 | +0.000362 | +0.044125 |
| **Mean** | **+0.009238** | **+0.010632** |

FFN-only won **0/5** paired seeds versus recurrent depth 2 and **1/5**
versus recurrent depth 3. Descriptive 95% paired-*t* intervals for
FFN minus recurrent NLL are **[−0.004125,+0.022600]** and
**[−0.012743,+0.034008]**, both including zero. These are small,
previously exposed development cohorts and must not be generalized
or interpreted as confirmed superiority. The recurrent models cost
~29% and ~28% more mean CPU process time than corresponding
feedforward controls under these GitHub runner jobs.

## Registration, actual training and proof of execution

- Two independently trained candidate depths × five seeds 1701–1705.
- Identical original 14 authentic CPython 3.13.0 source FIT owners,
  source version/license receipts, seed/owner/offset order as original
  CONTEXT026/recurrent LANGLAYER035.
- 320 AdamW updates per model, B8 and 128 original next-byte targets
  per element: 327,680 original FIT targets each. 4,096 original
  **previously exposed** DEV targets scored per seed.
- FFN-only upper blocks have no additional temporal mixing or persistent
  concept seeds. They match recurrent upper hidden widths (48 for
  depth 2, 24 for depth 3); original source native/aligned/wave/concept
  core and masking are unchanged.
- Within 1% active and deployed learned parameter parity. Final FFN
  widths redistributed because recurrent/graph weights carry genuine
  parameter cost: not an identical intermediate parameter allocation.
- Strict source prefix/causality tests, live parameter counts, finite
  gradient checks, exact source-hash preflight, actual CPU fit, all
  model+optimizer+Torch/NumPy/Python RNG checkpoints, SHA-256
  checkpoint integrity, cold reload and identical DEV scoring.
- Strict *independent* completed ten-checkpoint aggregator checks every
  run receipt, actual model size, original source owner/sample order,
  original frozen recurrence reference and complete seed matrix.
- GPU minutes = 0; protected TEST access = NONE; training FLOPs not
  independently matched; no merge, model promotion or external
  qualification.

## Engineering conclusion

Upper recurrence has value *relative to added tokenwise feedforward
depth* when both take from the same ~500K model size budget. This
converges with LANGLAYER036's frozen upper-memory-read lesions:
recurrence removal changed logits/useful NLL, whereas extra concept
graph specificity and slow concept history were close to inert.
Nevertheless the original single-stage TAC is still much more
time-efficient at these data/step budgets.

**Next falsifiable experiment:** replace only the **extra** layer
concept graph with a simpler recurrent-only read/write path; reallocate
saved active parameters to recurrent state/readout, keep ~500K budget,
match original source/seed/data/update/compute conditions, and then
measure long-context authentic-source information retention. Do not
assert a memory advantage from this 128-byte, 320-update DEV experiment.

**Original source and result IDs:** PR #565; training run #40788869823;
original runs #40785939365, #40786118546; PNDS full composed graph,
IEEE and differentiation equivalence remain OPEN.
