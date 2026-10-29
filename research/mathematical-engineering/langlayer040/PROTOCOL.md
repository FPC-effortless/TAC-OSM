# LANGLAYER040 — upper graph × state-width factorial reallocation

Registered **before new fits**. User requested isolating the two
changes that jointly improved the LANGLAYER039 genuine-source native
language benchmark. Fixed ~500K actual-active budget; no TEST and no
training-FLOP equivalence or independent generalization claim.

## Frozen 2×2 matrix

| upper recurrence | upper graph ON | upper graph OFF |
|---|---|---|
| **48 state** | Original recurrent/concept depth2 from LANGLAYER035; 499725p, FFN442 | **NEW** no-graph 48: 499623p, FFN455 |
| **59 state** | **NEW** full graph 59: 499848p, FFN426 | Original LANGLAYER039 recurrence-only 59: 499901p, FFN442 |

Original base CONTEXT026 native/aligned/wave/six BASE concepts remain
present in **all** four models. Only the *additional upper* concept
module is toggled. The new no-graph uses exactly the verified
RecurrenceOnlyUpperBlock; full graph uses exactly the old verified
CausalResidualTACBlock, at specified upper width.

**Critical confound:** decoder widths vary (426, 442, 455) to keep
active parameter totals within 0.06% of each other. Thus this is a
**parameter-budgeted allocation factorial** (graph/state vs available
decoder capacity), not a pure fixed-decoder or identical-initialization
causal intervention. Unlike a naive 48-vs-59/head-fixed experiment,
it holds TOTAL active parameter counts near 500K. Complementary fixed
decoder and frozen-weight graph lesions are separate investigations,
not results of this study.

## Training/provenance

- Two new upper configurations × fixed original five seeds
  1701–1705 × 320 updates, AdamW LR .001, clip1, batch8,
  original owner-safe 129-byte authentic CPython FIT windows.
  327680 genuine next-byte FIT targets/seed.
- Four original previously exposed DEV publisher docs, 4096 byte
  targets/seed, same metric code as frozen LANGLAYER035/039.
- Original frozen model checkpoints and scores from
  #40786118546 (h48 graph ON) and #40795790808 (h59 graph OFF)
  reused without refit. The new source independently verifies
  original publisher source, model/parent SHA/blob, actual counts,
  training sample order, checkpoint/optimizer/RNG, cold DEV replay.
- All 10 new fits must complete and pass original checkpoint SHA256;
  any missing/duplicate seed fails closed. CPU cost/RSS measured.
- Report mean and per-seed NLL/top1/Brier/PPL, four paired simple
  differences and the 2×2 interaction. Original single-stage TAC,
  P3 language n-gram still external reference; their parameter/FLOP
  baselines not assumed interchangeable with these exact factors.
- No GPU, TEST, DEV hyperparameter tuning, automatic merge,
  promotion, publication claim or hidden external authorization.
  PNDS whole composed graph/IEEE/formal AD still OPEN.
- Data and parameter matched by source/seed/exposure and <1%
  active param gap; **training FLOPs remain NOT MATCHED**. All
  outcomes are exploratory short-window engineering evidence.

## Longer-context follow-on

Prepare a separately frozen **no-training** authentic-language
context sensitivity protocol using same targets at 32/64/128/256
observed prefix lengths. Longer prefixes must be true within-document
original bytes. Score identical exact target offsets and record
document/source/window hash. Variable reset-window context is NOT
proof of a true carried-state streaming model or long-term memory;
no protected TEST. Report all arms' performance and compute, and
distinguish positional extrapolation from useful recovered history.
