# SEQ-TRANSFER-006C — held-out long-history state-update results

**Status: COMPLETE synthetic confirmation; mechanism attribution NOT established.**
All 60 fixed-budget training fits and the only registered held-out test
exposure completed on GitHub Actions 2026-10-19. This is a synthetic sequence
study, **not model promotion for real repair decision-making**.
[Original successful workflow](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39302028231)
with artifact `seq006c-original-60-fit-evidence`.

## Main result

For **both** scratch entity-local learners — TAC-inspired gated recurrence and
a simpler routed GRU — every one of **20 new training seeds** achieved **100%
on all three tests** in all three fresh held-out strata:

| Model | L256/E4: overwrite / toggle / pair-XOR | L512/E4: same | L512/E6: same | Individual 18/20 qualification gate |
|---|---|---|---|---|
| TAC-inspired gated | 100 / 100 / 100% | 100 / 100 / 100% | 100 / 100 / 100% | **PASS, 20/20 in all 9 cells** |
| Entity-routed GRU | 100 / 100 / 100% | 100 / 100 / 100% | 100 / 100 / 100% | **PASS, 20/20 in all 9 cells** |
| Global GRU | 52.80 / 51.33 / 47.73% | 48.97 / 48.47 / 52.30% | 50.30 / 46.92 / 50.30% | **FAIL, 0/20 in all 9 cells** |

The global-GRU percentages are arithmetic means over 20 seeds, on
shared 200-example/task/stratum datasets. They are **not** separate
20-fold samples of independent physical environments. Model-seed confidence
intervals and example-level uncertainty are different estimands. For a
single perfect 200/200 held-out task cell, a two-sided 95% Wilson interval
has a lower bound around 98.1% under independent-example sampling;
this should not be misread as a universal reliability guarantee.

**Interpretation:** The result supports *structured entity-local neural
update learning and length/slot generalization on the fixed synthetic
operations*, but it emphatically **does not establish a CASM-specific or
TAC-specific advantage**: the independently trained routed GRU exactly
matches all nine test cells and is structurally simpler.
The global GRU fails to preserve/query multiple independent entity states.
TAC operator gating may be convenient, but no necessity has been shown.

## Data and exposure audit

All exact discrete event inputs were observable SET0/SET1/TOGGLE/NOOP
with public entity IDs and queries supplied only after the complete
stream. Training used 3,000 sequences of L32/L64/E4, with eight
intermediate exact symbolic-state readouts supplied during training.
Validation used separate L64/E4 and L128/E4 sources, 600 each.
The 20 fresh training seeds were 690701–690720 for *every* model.

Only after all **60 completed checkpoint records** were written to
`all-trained.json` did the runner write `TEST-EXPOSURE.json`, including
its SHA256 digest, then generate the locked tests. Data and labels
were replayed from scratch by a second independent symbolic transition
loop during evaluation. Every held-out task and stratum contains
200 examples with exactly 100 positive and 100 negative labels.

| Registered held-out stratum | Namespace | Minimum queried fact age | Median age | Count |
|---|---|---:|---:|---:|
| L256 / E4 | 690603 | 110 events | 112 | 600 |
| L512 / E4 | 690613 | 220 events | 225 | 600 |
| L512 / E6 | 690623 | 220 events | 225 | 600 |

Original TEST public-source digest, as archived in `test-results.json`:
- L256/E4: `6ca67cc1476dedfb1aebcb8211161cbe480bdfdcfedeeb93908118be5d108235`.
- L512/E4: `6185060d1e39cfc117855c9d46e7666c2669b40fbc4e860ecb68d053a228a247`.
- L512/E6: `2ab8aae73d8701590df6f1c539ff6313cdfae4a525c250f1e67fb29d71f7d427`.

Model archive verification performed separately against the downloaded
original ZIP: all **60 checkpoint SHA256 digests matched** the training
ledger, all three arm-specific seed sets contained exactly seeds
690701–690720, all reported model evaluations refer to the matched
checkpoint digests, and the exposure marker's training-manifest digest
matches the exact original `all-trained.json` bytes.

**Provenance caveat:** `source_sha` in the result manifest is
`0bd6b369378cd35242b25eb057ff14f37269f9a0`, the GitHub Actions
pull-request **synthetic merge ref** available in `GITHUB_SHA`. The
checkout action explicitly uses the PR HEAD SHA
`a34f03450d8b42c8aa3d86e82e052b8fdf0763b4`; the latter is the
actual frozen source revision for this run. Both identifiers are
preserved instead of retroactively rewriting evidence.

Single-thread job-reported elapsed time was 890.73 s for fitting,
preparation, validation, and held-out evaluation on that worker.
This is **not** a production throughput guarantee or an energy benchmark.

## Scientific limits and unresolved questions

1. These were newly trained **scratch** models using TAC-*inspired*
   gating. The original DEV061.5 pretrained weights and native
   runtime were NOT loaded, so this is no direct attribution of
   original TAC-OSM learned parameters.
2. Public identity routing is programmed and supplied. Performance
   does not demonstrate emergent entity discovery or alias resolution.
3. Training uses exact intermediate-state labels. The separate
   **SEQ007 exploratory ablation** showed that such teacher labels
   were *not necessary for the same L64/L128 validation tasks*, but
   no independent 20-seed teacher-free L512/E6 confirmation exists.
4. Pair-XOR uses two cancelling toggles for each queried entity; it
   does not require arbitrary parity from both independently evolving
   entities. Update times have fixed generator structure.
5. No missing/contradictory evidence, noisy receipts, stochastic
   physical repairs, variable learned entity discovery, continuous
   sensor forecasting or real intervention economics are tested.
6. The global GRU is a bounded benchmark control; broader modern
   full-context Transformer, selective SSM and tuned explicit-state
   baselines were not tested here. Parameter counts differ by arm.
7. Perfect sample accuracy is not a guarantee beyond this closed
   synthetic distribution and must not be marketed as commercial
   model reliability.

## Decision

**Promote the scientific finding of useful entity-local state-update
structure, not a particular TAC/CASM mechanism.** Keep both neural
models as research controls. A simple routed GRU is an equally strong
candidate on this evidence. Do not merge an architecture promotion
into the product or announce general world modeling.

Next new registration: independently vary event timing, noncancelling
pair updates, public ID aliases, uncertain observations and unseen
compositions; compare final-only supervision, tuned routed GRU,
TAC-inspired recurrence and modern SSM/attention with matched budgets.
Also prioritize the real action-outcome/inspection-cost validation
dependency rather than another saturated fixed-rule classification.
