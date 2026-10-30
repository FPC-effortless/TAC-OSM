# LACUNA-ADDRESS005 — prospective address-specific prediction benefit

Registered before acquiring/scoring these source documents. Parent TAC
e41069d62f2b92fb3d4a8a64385ccbe9a366bedb; PNDS a2c9bd0453030f0431b59b8aaf5013528cf90c5c.
Three original frozen LANGMEM087640checkpoints1701/1702/1703. Internal research,
no fits/tuning/GPU/protected TEST/promotion/merge. Previous failed gates remain
immutable; even a positive005result does not automatically reopen training.

## New cohort, frozen before acquisition

Only Git tree metadata has been inspected. Paths are absent from source/data
receipts and research text in the inspected lineage. This is first005evaluation
of these documents, not a claim of no phrase overlap/pretraining exposure.

| Publisher/repository, pinned commit | Path | Git blob | Full bytes |
| --- | --- | --- | ---: |
| PSF python/cpython60403a5409ff2c3f3b07dd2ca91a7a3e096839c7 | Doc/library/decimal.rst | 916f17cadfaa7ed560f76789fb049aa966d03d57 | 81974 |
| same | Doc/library/itertools.rst | 43e665c3f0d5e377c70b32642f1a2a4e23e39053 | 64405 |
| same | Doc/library/statistics.rst | 614f5b905a4a2ebfb70a71958de1e908eb7e5780 | 43667 |
| Rust Project rust-lang/book2771df165f092a4fbb9f46b5b2e7450f19010adb | src/ch13-01-closures.md | ecda1acff5cfecdcc955d1256c9f185e6b99dea4 | 20936 |
| same | src/ch15-05-interior-mutability.md | 2bc04a770cbe1f4c5102dc1777c8064d5d5aa704 | 18548 |
| same | src/ch17-01-futures-and-syntax.md | 4ac55c467a6174d2564abd5fd2e51423de65ef19 | 19958 |

PSF Licensev2/code examples0BSD; Rust MIT/Apache2.0, retain publisher licenses.
Acquire original blobs after registration, verify Git hash/length and record
SHA256 before any inference. Source groups are6documents within2publishers;
seeds never increase independent source support. Require all6owners; no
adaptive replacement/lengthening/query changes after acquisition or scoring.

Observe first4096bytes perowner, score next bytes through offset4096; queryevery4
clocks t256..4095. Context cohort/age bins unchanged from003: [8,32),[32,128),
[128,240); most recent j with source[j-4:j]==source[t-3:t+1]. Never select by
past continuation agreeing with future label. Main endpoint age8–31 is fixed
from prior exploratory evidence; all other ages/arms are secondary.

## Fixed storage, addresses and controls

Same source model/trajectory, conditional slow decomposition H+C, same256Braw
ring including8Bclock, reconstructed source keys/values, fixed temperature8raw
attention and byte-code readout. Original model and native query are retained.
Read directions A=native Hq, B=H*supplied historical key, C=raw attention at q,
D=raw attention at supplied key. Historical supplied keys are evaluator-only.

Use003hard wrong key: cosine-nearest retained different-context key, absindex
distance>=5, intended historical target excluded from wrong4bytecontext. Keep
native/global source clocks, weights and candidate data fixed. Cyclic value
shuffles preserve respective native/full-history or raw/retained multisets.

Additional irrelevant-history key: uniformly select by fixed SHA256prefix seed
from qualifying wrong records in the SAME age bin; if absent, mark unavailable
and make that bin/arm inconclusive, never silently substitute another age.
Same irrelevant record for native and raw reads. Two generic random-direction
controls from SHA256(study,owner,clock,draw) map to8signed numerical components,
with no historical key/value/target access. Project every direction identically.

## Geometry matched between all compared conditions

Let h=Hq,c=Cq,f=original fast read, gates alpha_f/alpha_s fixed. Build a float64
orthonormal basis Q for span(c,f) using two-pass Gram–Schmidt, absolute epsilon
1e-12. P=QQ^T; b=Ph,r=||(I-P)h||,vperp=(I-P)v:

T(v)=b+r*vperp/||vperp||.

This fixes historical norm, dot with current write, and dot with fast read;
therefore it preserves historical, total slow, and final mixed fast/slow norms.
Test the ACTUAL float32 mixed vector passed to all3source consumers, not merely
the mathematical read. Store original/candidate totals and mixed vectors for
read-only independent checks. Original trajectory advances; patched states
are discarded. It is a conditional intervention, not a retrained rollout.

Exact zero axes are omitted. Nonzero axis norm<=epsilon => original h fallback,
INCONCLUSIVE_NEAR_ZERO_AXIS. Residual axis norm<=epsilon treated rank dependent,
with rank recorded and explicit norm checks. Radius<=epsilon or candidate
orthogonal norm<=epsilon => original h fallback and NO_DIRECTION status.
Rank0 uses norm-matched candidate, zero candidate falls back. Reject nonfinite,
nonfloating or non-[8] inputs. Retain every fallback/unavailable row; no useful
direction gate may pass with any such row in its compared arm/control set.

The remaining decoder nonlinearities and directional/content shift are not
controlled by these norm equalities. Projections can remove useful content.

## Primary endpoint, support and precommitted thresholds

Primary pair: Dcorrect versus raw hard-wrong, age8–31,
delta_address=NLL_wrong-NLL_correct (positive favors intended address).
Average perowner over its source cells and3seeds, then equally over6owners;
also report equal-weight publisher means and seed macro-means. No token-count
weighted global score may replace these group-balanced decisions.

Address-specificity gate requires ALL:
- At least20distinct source/query cells PER EACH of6owners in the bin, with
  valid hard-wrong/irrelevant/shuffle/random controls, no fallback or missing rows.
- Macro correct-minus-wrong margin>=0.001NLL (about14times004's observed margin),
  and publisher-specific mean margins>=0.001 for BOTH publishers.
- Positive correct-minus-wrong means in EACH document and EACH checkpoint seed.
- Correct advantage>=0.001 over shuffled values, irrelevant history, and the
  mean of two random-direction controls at macro level; positive for each
  publisher for all three comparisons.

Separately report practical prediction utility: original-minus-correct>=0.001
macro and positive in every document/seed. Specificity without practical utility
does not establish a useful component. Other bins/native matrix are secondary,
with the same gates and no picking a new primary after viewing labels.

## Autonomous query and all-query denominator

At EVERY scheduled query, before any evaluator history j is selected: record
native q's highest-cosine retained writer key (oldest index tie) and a uniformly
hash-selected retained key. Count whether each selected key's4bytecontext equals
the current context. These selection metrics use observed prefix only.
Measure original A, native-query/raw C and both random-direction controls on
ALL scheduled clocks, including clocks without a matching history. No oracle
match information is passed to those predictions. Selection gate requires at
least0.01absolute macro context-hit advantage over random, positive in every
document/seed. Native-query conditional utility additionally requires>=0.001NLL
advantage over original and mean random direction, positive in each document,
publisher and seed on all-query denominator. A supplied-key success cannot pass
this autonomous-query gate. An actual autonomous full-state architecture still
requires its own budgeted rollout and fresh registration.

## Integrity, scope and execution

Source hashes/licenses/owner resets; original9checkpoint replay; model100521p,
1920Btensors+logicalclock; frozen hash; existing086 ResearchProbeAdapter and
unchanged003three-consumer patch. Conditional sum<1e-5, key/value parity<3e-5,
Aidentity/slow_zero<3e-5, all three norm errors<=1e-10float64 and<=3e-6native.
Independent list projection<=1e-10; local projection AD/FD; actual composed
projection→decoder VJP/FD perseed/owner h=.001,error<=.002. Prefix-future poison
at first eligible query perseed/owner <=3e-5. No future label input to prepare.
Unimplemented full IEEE/AD/qualification checks stayOPEN/INCONCLUSIVE.

CPU phone plan/parity and scoped PNDS Lean/Python checks must pass before run.
Bound GitHub CPU job to15minutes. Reuse equal replacement reads within a clock
only; count actual decoder calls and attempts, all raw queries and source steps.
All14candidate/control arms and all-query scores retained in original artifact;
record CPU/RSS/logical banks/workspace. Native256Bbank matching is not equal-total
autonomous storage: evaluator ring/reconstructed keys/event coefficients are
charged separately. No training or latency superiority claim.

Preserve failures and exact original outputs; report-only recomputation never
reruns new-source inference. These documents become exposed after first scoring.
Report all6document/2publisher/3seed means, full availability and minority support;
no iid seed/token p-values. Six authored documents from two publishers do not
establish arbitrary-domain or factual-reasoning generalization.
