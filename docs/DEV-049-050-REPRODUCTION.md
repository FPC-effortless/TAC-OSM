# DEV-049–050 reproduction and evidence

## Frozen sources

* DEV-049A training and original protocol: `84db3f1` (30 fitted configurations; seeds 4901–4910).
* DEV-049A strengthened replay/aggregation code: branch `research/dev049-reliability-veritas-2026-10-17`. The recipe/model were not changed after fitting.
* DEV-050A initial engineering execution: `af3cacb`; attempt-order correction: `c8e3ec8`; amended public generator: `01ab1d1` (the final result cohort).
* Veritas runtime: `b21d829b62d3cab2811e746267ca305e110ed004` in `FPC-effortless/veritas`; the repository was not modified.

Public evidence is archived on [the dedicated evidence branch](https://github.com/FPC-effortless/TAC-OSM/tree/evidence/dev049-050-development-2026-10-17). It contains compressed raw reports, checkpoints, native synthetic traces, all three DEV-050 revisions, source bundles, environment versions and SHA-256 manifests. The native traces are harness evidence for these newly authored public worlds, not learner input or private Veritas panel exports.

## Repeating training

Create an isolated clean checkout at `84db3f1`, install Python 3.12, Torch 2.14.1+cpu and NumPy 2.3.5, then run each seed 4901–4910:

```bash
python scripts/run_reliability_dev049.py --seed 4901 --output /absolute/path/evidence/dev049/4901
```

Use a new output directory. No source edits or untracked files are allowed. `--smoke` executes only two updates and is rejected by the scientific audit. Fit, calibration, certification and evaluation namespaces are disjoint; final evaluation has 288 paired worlds per seed (144 independent generated pairs, not 2,304 independent entities).

## Replaying checkpoints and simulator outcomes

Use the DEV-049 branch containing the strengthened audit, with all source frozen, and the archived output tree:

```bash
python scripts/audit_reliability_dev049.py --root /absolute/path/evidence/dev049 --output /absolute/path/evidence/dev049/analysis.json
```

This regenerates public histories and labels, verifies the source protocol identity, seed/path uniqueness and split isolation, checks checkpoint hashes, replays checkpoint predictions exactly, independently recomputes threshold selection, executes certification and evaluation actions through the simulator, and aggregates all fixed policies. It does not independently replay optimization updates. Per-seed `independent-replay.json` files record the already completed audit; the summary aggregates these verified rows after checking their report hashes.

## Repeating the public Veritas comparison

Clone Veritas at its exact source pin and install it into the research environment (`pip install -e /absolute/path/veritas`). Its GIS and spreadsheet dependencies are optional for TAC-OSM itself but currently needed by Veritas's package imports. In a clean TAC checkout at `01ab1d1`:

```bash
python scripts/run_veritas_public_dev050.py --veritas-root /absolute/path/veritas --output /absolute/path/evidence/dev050-v3
```

The runner verifies the installed runtime path, source pin and clean checkouts. It executes 64 worlds × 10 seeds × 6 fixed controls, then independently reconstructs and re-executes each policy episode. Full public observations, native traces and verifier breakdowns must match. The declared sensor adapter charges a native inspection action and projects only the inspected entity's current fault; it does not pass the hidden oracle or global state into the policy.

## Limits

Hashes bind file identity, not scientific correctness. Public engineering development is not sealed confirmation. Policies share paired simulator worlds, and entities share a maintenance resource; report seed variability and whole-world/pair reliability, not inflated independent-entity sample sizes. The nominal one-sided 95% zero-event bound for 144 independent pairs is about 2.06% per seed under the declared sampling assumptions. It is not a deployment safety certificate or a simultaneous confidence statement for a post-hoc selected winner.
