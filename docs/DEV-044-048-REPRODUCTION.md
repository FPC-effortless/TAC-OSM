# DEV-044–048 development program

These are bounded development studies, not sealed confirmation. The protocols
were committed before their respective training runs. Each result identifies its
source commit; reproduce from that commit, not an arbitrarily newer branch tip.
Use a detached checkout per study so concurrent branch editing cannot change its
source. Evidence directories must be outside the checkout.

## Reproduce

Install Python 3.12, the editable project, torch 2.14.1 CPU, pytest 9.1.1 and
NumPy. The local continuation used NumPy 2.3.5; the original DEV-044A workflow
pinned 2.3.3. Preserve the actual package versions with the artifact bundle.

Download the three raw artifacts from Actions run `39011950314`, with directories
normalized to `/path/to/dev044/4401`, `4402`, `4403`. Retain their original bytes
and SHA256 values. The original workflow's complete audit is a separate artifact.
From the integration branch, a fresh full development rerun is:

```sh
python scripts/run_research_program_dev044_048.py \
  --parent-root /path/to/dev044 --output /path/to/new-evidence
```

Each study also has an individual `run_*devNNN.py` entry point with `--help`,
`--seed`, `--output` and `--smoke`. Smoke is a two-update engineering check and
is rejected by every scientific audit. The orchestrator never resumes or
overwrites partial runs; investigate failures and preserve them separately.

## Independent checks

* `audit_reservoir_scaling_dev044.py` checks original DEV-044A provenance,
  frozen writers, paired data, reported probabilities and simulator costs.
* The separate DEV-044 replay branch adds
  `audit_reservoir_checkpoint_replay_dev044.py`, linking checkpoint tensors
  back to calibration/evaluation probabilities and decisions. Batched float
  tolerances and exact paired rechecks are explicit in its source.
* `audit_frozen_policy_dev045.py` reconstructs frozen features, head forecasts,
  calibration selection and physical interventions.
* `audit_joint_training_dev046.py` reconstructs forecasts, frozen-writer
  interventions, training cohort identities, calibration and physical costs.
* `audit_local_generalization_dev047.py` reconstructs each checkpoint at
  8/16/32 entities, transfers the calibration threshold, and checks the public
  event-queue reference and physical outcomes.
* `audit_verified_routing_dev048.py` replays every candidate in both splits,
  verifies the selection lock, and independently recomputes sensing and
  intervention costs. Hardware timings are measured observations, not values
  expected to reproduce bit for bit.

These audits do not independently repeat optimizer steps. A checkpoint replay
is distinct from an independently trained replication. Three seeds support
descriptive comparisons only; worlds within a seed are not independent training
replications. Do not select the best evaluation arm and call it confirmatory.

## Remaining qualification gates

The added DEV-044B0 independently freezes transition parameters and internal
query projections because DEV-044A froze both together. That pilot uses three
new seeds, 108 updates and 128-event training histories; it is not the full
DEV-044B study, which still needs matched baseline tuning, longer optimization and at least
ten new independent seeds. The continuation adds bounded training and integration
evidence, not that missing attribution study. Learned memory-item retention,
robust uncertainty/value-of-information learning, learned identity discovery,
full original CASM/CDL equivalence, external digital twins and untouched domain
confirmation remain unestablished. The testable retention probe removes a
historical event; it does not train the proposed retention objective.

No code in this program invokes GEPA's LLM reflection optimizer. DEV-048's
development-only candidate archive is a deterministic control. No key or service
is required and no GEPA effectiveness claim follows from it. The selective
diagonal-state control is an equation-level hypothesis, not a Mamba reproduction.

DEV-047's variants share observation/prediction initialization and training
examples. The fast-GRU replacement creates a fresh writer before masking the
other read channels; it is an architecture comparator, not a byte-matched
single-bank ablation of the multiscale writer. The temporal-order replacement
also initializes its own writer. Their effects cannot be attributed solely to
removing or adding one memory channel. The temporal arm must additionally be
compared under matched parameter, tuning and compute budgets before interpreting
an advantage. The retention probe removes the earliest target observation,
which need not be the decisive historical event in a cancellation sequence.
