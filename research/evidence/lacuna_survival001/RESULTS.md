# Lacuna P0 execution results — 30 October 2026

The four generated proposals were added to issue #633 and actual MCP calls
succeeded. All eligible no-fit screens ran. The registered utility gate failed,
so the staged program stopped before expensive neural candidate fits. This is
not completion of the later trained comparison program or proof that these
architectures cannot work.

## Evidence survival

Three exact original640-update checkpoints; two original PSF FIT owners;
1,792 distinct next-byte positions reused across seeds. There were210 event
measurements over70 source/query/age cells and54 distinct historical positions,
not210 independent historical bytes. Each age has14 source/query cells measured
under3 checkpoints. Inputs are FIT only, not unseen generalization evidence.

| Age | Isolated event norm retained | Whole-matrix byte accuracy at supplied historical key | Isolated event oracle accuracy | ΔNLL removing event | Norm-matched wrong-event ΔNLL |
| --- | --- | --- | --- | --- | --- |
| 8 | 96.43% | 2.38% | 100% | -0.0000383 | +0.0000434 |
| 32 | 87.21% | 7.14% | 100% | -0.0000329 | -0.0000840 |
| 128 | 57.06% | 2.38% | 100% | -0.0000974 | -0.0000976 |
| 256 | 33.33% | 7.14% | 100% | -0.0000228 | -0.0000062 |
| 512 | 11.92% | 9.52% | 100% | -0.0000182 | -0.0000024 |

The percentages measure contribution norms, not retained information bits.
Isolated event recovery uses evaluator bookkeeping and is structurally easy:
a rank-one contribution preserves value direction. It is not content recovered
from the actual compressed matrix. The historical-key probe is also supplied
by the evaluator and never enters normal model predictions. Its low accuracy
is consistent with interference under this fixed read/decoder, not a proof
that no trained readout could recover anything.

Old event removal changed logits; the actual decoder is not disconnected from
slow state. Six source local AD/finite-difference checks passed (maximum error
0.00038945, preregistered tolerance 0.002, one predicted-class direction). This
is a partial local check, not full neural AD verification. Events remain
numerically present, but none of the age bins passed the preregistered positive
utility threshold or consistent seed/owner/control criterion. Complete event
rows and all subgroups remain saved. Attribution holds other original prior
state/query fixed and discards counterfactual state; it is not full causal
replay of all downstream reads. The sample does not require recalling a
particular older fact, and does not uniquely identify a querying versus decoder
failure. Numerical survival alone therefore cannot justify a memory-capability
claim or expensive fusion under this program's current gate.

## Bounded exact lookback and retention

| Policy | FIT next-byte NLL | Queries per owner/checkpoint |
| --- | --- | --- |
| never | 2.274959 | 0 |
| always | 2.229398 | 896 |
| random | 2.260461 | 224 |
| entropy | 2.263825 | 224 |
| survival_proxy | 2.264031 | 224 |

Always looking back improved this FIT loss by 0.04556 at 4× the selective lookup
budget. Random lookup beat entropy by 0.00336 and the overlap/survival proxy by
0.00357 with EXACTLY equal query counts. Each selective policy causally chooses
one lookup per four-byte block; no future labels or future uncertainty determine
its budget. The tested cheap trigger choices are unsupported. This does not
rule out a separately trained router. Full method uses3,858 logical payload
bytes per owner (native 1,928  + ring 1,928  + flags 2), not the original1,928-byte tier.
Shared FIT prior 2,048 B and allocator/workspace/RSS are separate. No production
speed or whole-model memory advantage is claimed.

Each standalone context cache uses exactly 256 payload bytes including metadata:
24 B header +25 slots × 9 B +7 unused charged bytes. Results:

| Retention policy | NLL | Lookup hit coverage | Mean retained age |
| --- | --- | --- | --- |
| fifo | 3.230185 | 2.344% | 12.00 |
| reservoir | 3.226343 | 2.846% | 300.68 |
| diversity | 3.230185 | 2.344% | 12.17 |

Reservoir retention preserved much older contexts with a small coverage/loss
change. These count predictors do not replace the neural model and are not
learned utility selection, swap rounding or eviction-to-summary conservation.
Those proposed mechanisms remain unimplemented rather than being counted as
successful by these simpler component screens.

## Remaining seven-mechanism P0 coverage

- Targeted erasure:462 observed continuation-disagreement cases. Context-directed
  erasure changed mean NLL by −0.000222; wrong-key erasure by −0.000819. Multiple
  continuations in language are not stale facts. Actual unlearning efficacy is
  BLOCKED without source-authenticated supersession/versioned truth.
- Kalman math on actual native keys: minimum covariance eigenvalue 0.003931,
  minimum innovation denominator 0.1. Discarded off-diagonal norm reached 1.156824,
  confirming diagonal projection is an approximation. Reference covariance is
  Float64 / 512 B and its reference memory 512 B; neither is an integrated trained
  model or an uncounted addition to native state.
- Independent erase, Möbius and triangular-affine identities/counterexamples:
  source-pinned original seven mathematical checks passed. ARCHFUSE089’s 12
  operator/source checks passed. Combined targeted tests/probe suite: 36 passed.
  These include KDA, rotations, Titans-style update, xLSTM, Lightning Attention,
  exact retrieval, verifier and ternary kernels; no trained integration occurs.
- Cross-layer retrieval: source has one native fast/slow pair, not a second
  native layer bank. Aligned heads are not independent layers. Additional
  layer state needs explicit reallocation before a matched candidate exists.
- Whole-model scan: native slow transport has an affine input-conditioned law;
  fast clipping and upper/query/read coupling prevent treating the entire
  original neural graph as one token-local affine scan. No speedup was measured.
- Structural stack: all 14 FIT owners audited for literal brackets; maximum depth 5,
  zero overflow at capacity 8. This ignores syntax/strings/prose context and is
  not a parser or a real-code prediction benefit.
- Historical-credit and other neural fits: NOT RUN. They remain gated by useful
  old-content evidence, independent per-variant formulations, matched active
  95k–105k/within1% weights/data/training compute/state and fresh unseen DEV.
  Original LANGMEM087/088 findings and checkpoints remain untouched.

## Reproducibility and scope

Source c5df300cf0d565ba7f9c9622a31821ba812333df. Original screen run:
https://github.com/FPC-effortless/TAC-OSM/actions/runs/40997601362

Artifact 12952152198, SHA256
`e64641737ccb3d4f6b14d6bcc8835d5c7c092a060457346636222be31b002a30`.
The exact original artifact and JSON are Git-owned here. Independent saved-row
recomputation passed every summary, control exclusion, query budget and gate;
this required no new model inference or source scoring.

Actual screen CPU: 67.364 seconds, peak RSS: 364,404,736 bytes. Conditional contribution
sum matched original S to 1.13503e-7; independent PNDS087 step reference error
6.88833e-9. Numerical/gradient checks remain scoped. PNDS draft #88 adds typed
conditional transport, probability mixture and budget formulations; complete
neural IEEE/AD, learned policy correctness, global semantic goals and independent
qualification are OPEN. Archive replay independently passed all nine checkpoints from
Git (run 40996629169 and direct remote clean clone); new anchors are labeled new,
and original LANGMEM086 incomplete NumPy metadata remains documented.

No neural optimizer step, GPU quota, DEV scoring, protected TEST, model promotion
or merge. This is internal FIT mechanism evidence. Original issue #633 stop rule:
"If no attributable, nontrivial older signal, stop before expensive candidate
fits." The preregistered useful-content screen did not pass; the program does
not launch the later trained sweep or claim that all proposed research is done.

Current scoped PNDS Lean/Python CI40998663258 PASSED. Full local TAC suite:
1,620passed/4skipped; no new fits or DEV scores were needed for verification.

Cross-host replay against the NEW anchors was within the preregistered tolerance,
with maximum absolute error 8.583069e-6; none of the nine complete outputs/states
were bitwise identical across hosts. Within-host cold reload was bitwise for
all nine. Direct remote clone on the anchor host matched bitwise. Original-job
FIT logits were never saved, so historical-output equality stays INCONCLUSIVE.
