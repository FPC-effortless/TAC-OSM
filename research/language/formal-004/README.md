# TAC-LANG-FORMAL-004: Language source math and guard contracts

Date: 2026-10-27. Status: **exploratory source restoration and proof development, no model promotion**.

This stacked draft preserves original five-seed TAC-LANG-GATE-003 source as `parallel_tac_language_gate_original.py`. The original human-authored Vim manual and binary checkpoints are intentionally not republished here. The source is pinned to the experiments documented in [TAC Issue #525](https://github.com/FPC-effortless/TAC-OSM/issues/525). An unmodified architecture is required for frozen checkpoint replay, so strict checks are in a **separate adapter** and the bounded-mean idea is an untrained mathematical alternative, not a silently changed model.

## Six confirmed weaknesses and fixes

1. **Constructor/parameter mismatch.** The source defaults (`decoder=92,post=10`) instantiate 107,089 learned coordinates. The reported experiment's 100,680 require `decoder=75,post=10`, which the wrapper now enforces. Unqualified default instances cannot be mistaken for the originally reported model.
2. **Checkpoint positional-mode ambiguity.** The actual `train_gate.py` constructs its 100,680-parameter SiLU/linear arms with `position_mode='multiscale'` (single-lag read), NOT the class default `multiscale_relative` (four-lag read). Both have identical parameter counts. The frozen-session validator now rejects the misleading four-lag default; that mode remains an exploratory separate architecture, not verified trained gate-study evidence. The arm named `tanh` in source actually uses the raw linear modulation (not hyperbolic tangent).\n3. **Guaranteed ordinal-key alias.** Fourier periods (7,13,29,59) repeat after **155,701** steps. The checked streaming adapter rejects steps past the first period. This excludes one *guaranteed* alias source in the restricted runtime; it is **not** a proof of collision-free or accurate retrieval within that period.
4. **Unbounded stored source sum.** The actual 88-dimension `state['sum']` accumulates bounded per-token `tanh` features but can grow linearly with stream length. The decoder consumes `sum/max(n,1)` instead. We supply an exact-real mean recurrence `m_(n+1)=m_n+(basis_n-m_n)/(n+1)` and local numerical test. No changes to checkpoint computation, trained results, or 408-float count are made; replacement would need source-specific full replay, numerical bounds and independent evaluation.
5. **Stateless forward nonzero-offset misuse.** `forward(tokens,offset>0)` changes positional and ordinal codes without accepting previous state, so it does not implement session continuation. `CheckedLanguageSession` enforces the true owner-locked streaming path, exact counter step and valid 128-character IDs. Unchecked original forward stays untouched.
6. **Missing language semantics.** Source-aligned PNDS model proofs apply to continuous R14 output, not this 128-character categorical emission, SiLU modulation, 64-lag reconstruction or language tokenization. Companion PNDS draft #51 includes a source-shaped *complete typed exact-real language forward graph*, its source-to-source chronological continuation/owner/prefix and distribution lemmas, and further bounded-memory categorical/affine theorems. This is **not** yet a full Torch/IEEE/autograd implementation-refinement proof.

## Evidence, tests and open obligations

`python -m unittest -v test_language_contract.py test_fullgraph_correspondence.py` runs the original seven numerical contract test groups plus the independent source-to-reference full-graph and derivative tests, including the actual t-64 historical read. Tested locally on CPU with **7/7 passing**, including model parameter identity, forced-ordinal cycle, batch/stream future-causal prefix, owner and invalid token refusals, corrective write equivalence, AD gradient agreement between serial and rectangular scans, and the alternative bounded mean. These are finite numerical tests and are **not** IEEE or all-weights proofs. The PR workflow re-runs them with clean CPU dependencies.

Companion `FPC-effortless/pnds-lean` draft `research/tac-lang-formal-004-2026-10-27` adds `PNDS/LanguageCategorical.lean`, `PNDS/LanguageFullGraph.lean`, and a dedicated full-graph Lean/Audit build. An initial six-lemma full-graph build [#40573976380](https://github.com/FPC-effortless/pnds-lean/actions/runs/40573976380) passed after a diagnosed and corrected state-constructor parse error. Subsequent categorical logit-shift and root-integration CI is independently tracked; only completed green runs may be cited as verified. No full-model Torch refinement proof is claimed.

Original experimental binary archive was located in the user's library. **Local safe replay PASS**: 31/31 original archive member hashes, 36/36 Vim 9.1 source chapter hashes, and 10/10 model-checkpoint scores with original 1e-6 tolerances and no optimizer updates. SHA-pinned source and archive details are in `LOCAL_REPLAY_RECEIPT.json`. `verify_original_archive.py --bundle /path/to/tac_language_gate003_complete.zip --source /usr/share/vim/vim91/doc` reproduces the source-hash and results check with restricted Torch `weights_only=True` loading. GitHub cannot currently cold replay the uncommitted binary ZIP; **independent CI checkpoint qualification stays OPEN**.

Still **OPEN**: **formal correspondence** between the source-shaped typed full 100,680-parameter mathematical graph and the actual PyTorch source (the graph is specified, not proven a compiler refinement); float32 tensor kernel, exp/log/tanh/sin/cos and normalization refinement; complete backward/AdamW derivatives; trained checkpoint cold replay from GitHub; source ownership across document boundaries; no-leak historical loss derivation; stochastic/learned probability calibration; long-distance fact fidelity; held-out independent real-text evaluation. Synthetic tests do not validate user-level language intelligence.

No new seventh model family, no changes to the original six-family register or historical gates. No training or protected source TEST occurred here.

## Continued finite-precision and audit work (2026-10-27)

The following changes are **additive to the checked inference adapter**, never changes to frozen checkpoint weights or the recorded SiLU/linear experiments:

- **Safe log probability**: `CheckedLanguageSession.log_categorical` uses `torch.log_softmax` rather than `log(softmax())`. A constructed FP32 example at logit -1000 proves the latter produces zero mass and `-inf` logs even though all logits are finite. The stable API returns finite log probabilities in that diagnostic and remains differentiable in the source session. This does not prove calibrated language probabilities or eliminate all FP32 overflow paths.
- **Positive normalization stabilizer**: the model contract checks that `model.norm.eps` equals its frozen value `1e-5`. PNDS `LanguageLayerNorm.lean` proves the 248-dim variance is nonnegative and the exact-real square-root denominator is strictly positive under a positive epsilon. This is a mathematical precondition, NOT a Torch `rsqrt` kernel refinement.
- **Chronological additive error accounting**: `observe_fp32_sum_error.py` compares constructed FP32 addition traces to an FP64 source and accounts separately for basis-cast and addition-rounding errors at every step. It tests 88 channels across several stream lengths without training or protected source access.
- **Lean finite-precision accumulator contract**: PNDS `LanguageFPContracts.lean` proves a source-prefix absolute-error envelope `initial_error + N * per-add-error`, a composition theorem including upstream basis quantization, a bound on count-normalized errors and an L1 envelope for bounded inputs. These are **conditional on per-operation trace receipts**; they do not prove all FP32 kernels satisfy one global error constant.
- **Fail-closed proof audits**: the original workflow had a stale theorem-count comparison and could permit unknown axioms via over-broad grep. The new `scripts/audit_language_lean.py` compares every declaration requested in `AuditLanguage.lean` against Lean output and checks the exact admitted axiom set `{propext, Quot.sound, Classical.choice}`. Adversarial parser tests reject missing/duplicate declarations, unknown axioms and `sorryAx`.

This is a strict iterative proof project. All new theorems require green Lean build **and** the corrected axiom-audit job before they may be counted as verified. Even after these tests, the global Torch/IEEE, full AD, optimizer and real-world retrieval claims remain open.


## Language AD and memory-stability continuation (TAC-LANG-AD-009/010/011)

The new PNDS proof files are `LanguageLossAD.lean`, `LanguageMemoryAD.lean`, and `LanguageMemoryStability.lean`. Their source version is the five-seed 100,680-parameter gate experiment, **not a new model family**.

- **128-way loss terminal cotangent**. The exact-real directional derivative of the full categorical negative-log-likelihood is `sum_i softmax(logits)[i] * dlogit[i] - dlogit[target]`. The theorem covers an arbitrary differentiable curve of all 128 logits; it does not claim every upstream Torch operation is automatically verified.
- **Two memory derivative refinements**. Actual 8×8 native corrective writes and both 16×8 aligned banks are definitionally mapped to the earlier kernel-proven PNDS `writeRow`. Each coordinate's source JVP includes variations of memory, address, value, retention and write strength, including all 16 rows and both aligned heads.
- **Prior-memory sensitivity**. A new language-specific stability theorem applies the source's epsilon-normalized address, sigmoid retention and two-head routed write strength to prove nonexpansive squared row norms for **two states receiving the same key/value/gate input**. This is not a bound on total memory values or a guarantee of reliable historical recall.
- **Independent numerical falsifiers**. `test_language_finite_difference.py` independently checks central function-value differences against PyTorch AD across all 36 trainable tensor groups, including a 64-character historical loss. This is a constructed diagnostic only, not a claim of real-text quality. An additional test perturbs native/aligned memory across two streams and compares row energy before/after the same source input.
- **Strict limitation**. Mathematical `HasDerivAt` proofs are exact real; branch points of source `F.normalize` and FP32 numerics, layernorm/softmax backward kernels, entire training optimizer, and whole-graph source-to-Lean lowering all still require separate proofs and/or restricted numerical checks.

The dedicated PNDS CI **successfully built and axiom-audited 67 scoped theorem declarations** at [run #40594586314](https://github.com/FPC-effortless/pnds-lean/actions/runs/40594586314). This adds analytic SiLU/sigmoid and all-248-channel mean/variance LayerNorm derivatives beyond the verified 44-theorem baseline. This result is exact-real source-math verification, not full Torch/IEEE refinement. Historical checkpoints have not been changed or retrained.


## Further formal proof milestone: activation and complete LayerNorm JVP

The companion PNDS files `LanguageActivationAD.lean` and `LanguageLayerNormAD.lean`
now kernel-check exact-real derivatives for the source's sigmoid, SiLU,
2×SiLU source gate, a normalized feature and all 248-source mean/variance
couplings. The **52/52** source-scoped theorem/axiom audit passed at the
recorded PNDS SHA `7a4b43d4cefbcf73545f33bd5819a2b3369690bc`.

`test_language_nonlinear_derivatives.py` separately uses source-shaped
Torch double-precision operations to compare those slopes with analytical
formulas and central differences across gate saturation, nearly-constant
LayerNorm inputs and multiple scales. The test suite passes clean CPU CI
at TAC SHA `359c53149e75bdba2973f0281a55532bdc66270d`.

Important: nontrivial finite-difference truncation near a positive-but-small
variance motivated a documented *finite-difference-only* tolerance correction;
the independent analytical-vs-autograd agreement is stricter. None of these
source-scoped proofs claims an IEEE-754, GPU, optimizer or checkpoint semantics
refinement. The original trained model stays frozen.


## TAC-LANG-FP-014/015: local tensor rounding receipts and key sensitivity

**Added mathematics in PNDS #51.** `LanguageTensorFP.lean` proves, for
arbitrary finite reductions, how coordinatewise input/product errors add
and how **one separately accounted** tensor reduction error composes.
Its eight-term source memory read law permits distinct quantization
errors in the address and value row, distinct rounded multiplications
and one rounded reduction. Every error term is a premise backed by a
backend receipt; none is asserted for all CPU/GPU operations.

`LanguageKeySensitivity.lean` proves that the actual epsilon-floor
normalization branch can amplify small errors by `1/eps`. With
`eps=1e-6`, an input coordinate change of 0.0000005 changes the
normalized coordinate by 0.5 in the scalar witness. The source's
`F.normalize` also has a clamp-boundary derivative kink, shown by
the dedicated Python test. This is a **research weakness to measure
on authentic records**, not grounds to alter the frozen checkpoints
or claim a observed retrieval failure.

**Replayable finite-precision diagnostic.**
`audit_tensor_rounding.py` compares FP32 CPU tensor operation results
against the exact binary-rational sum of the **actual FP32 operands**,
not a potentially rounded FP64 surrogate. It records a candidate
`gamma_(2n)*sum(abs(ai*bi))+8*n*2^-149` envelope and reports whether
a given operation satisfies it. Shape coverage includes 8,16,48,88,128
and 248; adversarial cancellation and subnormal cases are included
as explicitly limited diagnostics. The suite and source-hash-pinned
`generate_tensor_rounding_receipt.py` CI artifact pass in [TAC
CI #40591297284](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40591297284).

The observed envelope **is NOT a formally discharged general
IEEE-754 or Torch-kernel axiom**. FP environment, FMA/reassociation,
flush-to-zero, transcendental accuracy, optimizer state and all backprop
tensor operations still need explicit per-device/backend refinement.
The original ten binary checkpoints are not yet downloadable
from a cold GitHub workflow, despite complete verified local replay.

## TAC-LANG-REPAIR-016/017: Research-only alternatives, no frozen model change

See [REPAIR_016_RESEARCH.md](REPAIR_016_RESEARCH.md) for the mathematical
derivations, failure modes, source-verified risk ranking and the preregistered
real-data test gates.

- `candidate_stable_address.py`: smooth 8D key normalization, 2×8×8
  parameter-count-neutral bounded rotation and a position-era code. The
  conditioned rotation prevents an aligned source key from collapsing to
  zero in exact real arithmetic, while the era tag removes the **forced
  exact 155,701-step address repeat** at the pre-projection level. Finite
  FP32 era resolution, query interference and retraining remain open.
- `test_candidate_stable_address.py`: no new dataset or training, only
  operator diagnostics. Tests source identity in the first era, source
  repeated-cycle aliasing, the smooth 8D Jacobian spectral bound, a
  lower singular-value floor, and deliberate eventual FP32 era collision.
- `candidate_bounded_mean.py` and `test_candidate_bounded_mean.py`:
  an 88-value alternative cumulative mean update with exact-real
  equality to source context, but *no guaranteed floating superiority*.
  Includes a case where it improves accuracy and a counterexample
  where the original source sum is more accurate.
- `audit_fit_key_conditioning.py`: executable, source- and
  checkpoint-hash-gated diagnostic of the original 10 frozen models
  on actual Vim 9.1 FIT chapters; requires the original extracted
  checkpoint archive and matching Vim manual. No TEST or training.
  Initial diagnostic observed **native min norm 0.52088**, aligned
  min norm **1.71583** across 2,048 positions per original checkpoint;
  none reached the F.normalize epsilon floor. This rules out
  *asserting an observed near-zero-key failure* in that sample,
  not a general stability flaw.

Companion PNDS `LanguageRepairCandidates.lean` has nine additional
exact-real theorems for smooth-key denominator positivity, input relative
amplitude, exact ordinal era/residue reconstruction and era monotonicity,
raising the scoped audit to 67 results (compilation and axiom audit must
remain green). Other math derivations (full 8D Jacobian/operator norm,
conditioned learned rotation floor) are documented and numerically tested
but are **not yet directly kernel-proven in Lean**.

No branch changes the original PyTorch model, checkpoints, training budgets
or historical negative experimental results. Real-text matched-parameter,
matched-budget evidence is required before promoting any candidate.

## TAC-LANG-REPAIR-018: empirical untrained-delay retrieval safety

The original model's historical loss was trained for **exactly lag 64**.
Earlier treating all lags <=64 as qualified was UNSUPPORTED and has been
corrected. The source-exact adapter `selective_history.py` trusts
**only lag=64**, abstains for every other delay, and if a categorical
result is requested returns a strictly positive 128-token prior derived
from the **FIT documents only** (Dirichlet-style add-0.5 smoothing).

`audit_selective_history_dev.py` verifies all 36 genuine Vim9.1 chapter
SHA256 digests, the original model source, 10 original checkpoint byte
hashes, parameters and sample histories; it replays the **exact exposed
DEV sampler** without retraining or accessing independent TEST.
`REPAIR_018_DEV_REPLAY.md` contains the 10-pair results:
mean original lag64 3.460926 nats vs FIT prior 3.531907;
mean original lag96 5.219261 vs FIT prior 3.542343,
with the FIT-only fallback ahead at lag96 for all 10 original checkpoints.
The 96 improvement is 1.676918 nats on the *reused DEV*.

The fallback is a source-derived **uncertainty safeguard**, not a new
historical memory or evidence of long-context retrieval. Its 128
nonlearned float32 probability values (512 bytes) are accounted for
in addition to original 100,680 learned parameters and 408 persistent
floating values. It cannot use the historical target, future tokens
or DEV scores in its routing decision.

The companion PNDS mathematical contract
`LanguageSelectiveRetrieval.lean` proves exact trained-delay selection,
positive and normalized fallback and convex mixture mass/floor; it is
an exact-real source contract, not model calibration or IEEE validation.
The expanded 73-theorem audit must be green before reporting it
as verified. Real-text independently reserved TEST and matched training
for genuine 96-lag retrieval remain open.


## MLAG001: real source-matched dual-horizon training (new experimental evidence)

The original 100,680-active-parameter, 408-float-core language architecture
was trained afresh for five paired seeds × two existing gate modes using **the
same authentic Vim9.1 FIT chapters and byte-identical per-seed source sample
order**, still **120 optimizer updates**. We replaced the auxiliary objective
`0.30 * L_history64` with `0.15 * L_history64 +
0.15 * L_history96`; main next-character loss, architecture and optimizer
configuration were unchanged. The original binary checkpoints were frozen.

On the **already-exposed original six DEV chapters** the new 96-character
historical NLL was **3.127606** versus original **5.219261** across ten paired
checkpoints, and exact character recall was **24.740%** vs **5.514%**.
Every one of the ten comparisons improved. However next-character NLL
**worsened** from **2.824216** to **2.827443**, and this is an objective
ablation rather than demonstrated superiority to GRU, Transformer, RWKV,
Mamba or external RAG. 96 is inside a 128-character observation window,
**NOT** large-context or lifelong memory.

Source control checks at lag96: new correct-key NLL **3.127606**; FIT-only
unigram **3.542343**; history read removed **4.852031**; adjacent wrong-key
**3.222506**, 8-step wrong-key **7.629065** (wrong-key evaluations use
shorter common-target intersections). Evidence for use of the address,
not proof of robust semantic/temporal world memory.

Source and checkpoint hashes, ten complete new model+optimizer+RNG
checkpoints and numerical cold reloads, full individual results,
per-ordinal controls and exact training/replay source are bundled in
the research conversation export, sha256
`b19e026e49c06093850fafa6ca63bffd84f1153aa947f8cae67938d66a4c5bfd`.
The binary files are **not yet committed into GitHub** and the byte-identical
Vim manual is not mounted in GitHub CI: **independent cold repo replay is still
OPEN and this cannot be promoted**. The GitHub branch includes the exact
`train_multilag_real_fit.py`, `audit_multilag_memory_controls.py` and
`test_multilag_protocol.py` code, and CI asserts source geometry,
parameter budget, target alignment and source-causal differentiability
on labeled diagnostics only.

See `MLAG001_REAL_DEV_REPORT.md` for methodology, seed-by-seed
measures and the five-independent-seed-block uncertainty. The companion
PNDS module `LanguageMultiLagObjective.lean` formalizes 0.30 weight
parity, exact-real objective derivative and past-only query ordinals.
The source-specific Lean audit targets 78 declarations (check latest CI).

See `LANGUAGE_COMPETITIVE_ADVANTAGES_PROTOCOL.md` for the research
benchmark ladder and near-term memory SDK evidence requirements, including
LongMemEval-V2, NoLiMa and the actual Mamba-3/RWKV-7 competition.
