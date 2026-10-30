# LANGPERSIST096: lexical reuse and broader capabilities

Prospective internal development research using nine original trained models.
Two separate batteries answer different questions: causal use of acquired code
history, and broader learning/reasoning/executable-action capability. Neither is
a certificate of general intelligence. See `PREREGISTRATION.md` for all fixed
endpoints, controls, thresholds, budgets and limitations.

- Lexical:300real-code episodes, five publishers, held-out from model FIT but
  historically exposed DEV. Unknown FIT identifiers, matched recent queries,
  complete reset/donor/corruption and an explicitly metered raw-history ring.
- General:24fixed public development tasks, eight each from ARC-AGI, GSM8K and
  HumanEval. Same models, carry/reset, fixed128-byte generation. ARC demonstrates
  unknown transformations; mathematics requires numeric reasoning; programs are
  checked against executable assertions in an isolated restricted interpreter.
- HumanEval/160 has an unsupported canonical eval reference. It remains
  inconclusive, without replacement or a fabricated model failure.

No new fitting, GPU quota, protected repository TEST, checkpoint promotion or
architecture-superiority claim is involved. The existing training FLOPs are
unmatched, and no qualified frozen Transformer control was available. Report
mechanism effects within each checkpoint, not an architecture leaderboard.

## Data and process separation

The corpus builders authenticate original publisher commits, SHA256, byte sizes,
Git object identities and licenses. `FROZEN_INPUTS.json` binds the selected
arrays, task identities and checkpoints before first model predictions.
Original files and all outputs are create-once. A failed attempt is preserved.

Acquisition and query workers are separate Python processes. Neural query input
contains only Q and complete source-bound state, never raw acquisition, future
labels, filenames, identifiers or evaluator code. General judges live in a
separate file from actor inputs. HumanEval code has no unrestricted host-exec
fallback; missing confinement is inconclusive and blocks this preregistered run.

Fixed-episode future poisoning mutates only future source bytes in private
copies, re-extracts A/Q without reselecting episodes, then checks numerical
prediction replay in fresh workers. Original sources are never modified. This
is scoped producer/worker evidence, not arbitrary-code information-flow proof.

## Run

Use a clean checkout of the frozen prospective commit, Python3.12, NumPy2.3.5,
CPU PyTorch2.5.1, pytest and system bubblewrap. The reviewed phone planner routes
the bounded job to GitHub Actions CPU; the workflow retains state/prediction/
resource files and failed outputs in a downloadable artifact.

```sh
python scripts/phone_lab.py plan research/phone_queue/langpersist096-real-state-utility.json
python scripts/validate_research_parity.py --manifest research/mathematical-engineering/langpersist096/FAIR_COMPARISON.json --phase preflight
python -m pytest -q tests/test_langpersist096_*.py tests/test_research_probe_agnostic.py
python research/mathematical-engineering/langpersist096/run.py --cache /tmp/langpersist096-cache --output ci-artifacts/langpersist096/experiment --preregistration-commit FROZEN_COMMIT_SHA
```

Use a new cache/output directory for an independent end-to-end replay; existing
sealed data manifests are not overwritten. Query replay within each run reuses
the same immutable acquired state and checks byte-identical numeric outputs.

Resource receipts distinguish shared setup/acquisition, per-condition query
and generation, full experimental totals, state/metadata/model bytes and peak
RSS. Four neural interventions are not compared against one raw-control policy
as though they were serving latency. Real deployment energy/latency remain open.

## Interpretation and next decisions

NLL and exact-four completion have separate gates; an NLL gain alone does not
establish useful completion. General domains are reported separately, including
all unsupported and failed cases. No single AGI/IQ score is produced. Frozen
byte-prediction models have not been instruction-tuned, so a null result is a
measured present capability limit, not a universal architecture impossibility.

Acquired lexical information is weaker evidence than a reusable procedure.
ARC tests unknown-rule transfer; external program verification tests executable
outcomes. Autonomous repair, learned verification, continual consolidation,
open-ended structure discovery and general intelligence remain separate
unproved hypotheses. Changes to the learning objective or architecture require
a new source-authentic preregistered study after these results, not retroactive
selection on this trial.
