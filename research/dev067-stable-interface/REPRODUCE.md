# Replay DEV067 without refitting

Frozen fitting registration/source: `c39fc6c2110e63e01314d6409ae52f5eb245eb04`.
Main initialization seeds96701–96720; pilot96700 at both registered learning rates.
Six candidate heads share three encoders per initialization. Four TAC heads have
identical final encoders; do not count them as four independent encoder fits.

Python3.12, CPU PyTorch2.14.1 and NumPy2.5.4. Install TAC-OSM editable. Native replay
requires the separate Veritas checkout at
`b21d829b62d3cab2811e746267ca305e110ed004`, installed editable. The exported model
does not import Veritas, the training harness, the simulator or NumPy.

After the evidence archive is published, restore and replay:

```sh
python scripts/restore_stable_semantic_dev067.py
python scripts/audit_stable_semantic_dev067.py --directory results/dev067-main --output results/dev067-replay.json --exports results/dev067-restored-agents --veritas /path/to/veritas
python scripts/analyze_stable_semantic_dev067.py --directory results/dev067-main --output results/dev067-replayed-statistics.json
python scripts/intervene_stable_semantic_dev067.py --directory results/dev067-main --output results/dev067-replayed-interventions.json --details results/dev067-replayed-interventions.jsonl.gz
python scripts/shuffle_stable_semantic_dev067.py --directory results/dev067-main --output results/dev067-replayed-shuffle.json
```

The audit checks every manifest hash, registered fitting source, final EMA
checkpoint, encoder equality, posterior metrics, validation-selected temperature,
recorded policy decisions, public execution receipts/costs, clock invariants and
the semantic-only actor interface. Raw weights remain hashed and preserved;
registered results use final EMA, with no intermediate or test-based selection.
Both sequential and batched ingestion adapters must reproduce the same state and
outputs. Exports use the predeclared first fresh seed 96701 for every arm, rather
than the best seed. Read the final results/model card before treating an export
as a qualified candidate. An artifact alone is not qualification.

Each closed-loop intervention follows its own evolving public history. Exact
posterior substitution is an evaluation-only known-law public reference, never a
deployment correction or input from hidden physical state. Blurred-belief failures
belong to intentionally damaged variants. The fixed-query cyclic shuffle stays
within the public graph class; its action flips are sensitivity, not executed regret.

Refitting is separate: use a clean checkout of the fitting commit, restore the
completed DEV065 evidence to `results/dev065-main`, then run
`python scripts/run_stable_semantic_dev067.py --directory results/new-dev067-run`.
Legacy imports are limited to hash-verified training-only DAgger snapshots and
preserve file/row provenance. This reproduces an already-open benchmark, not a
fresh confirmation. Never drop failed original seeds, extend this budget, change
the gates or substitute a test-winning control. A new hypothesis needs a new
registration and untouched evaluation. CI restores evidence without fitting.

`benchmark_stable_semantic_dev067.py` reports narrow first-seed, one-host CPU/wall
time and tensor storage. It does not measure physical DRAM traffic, energy or a
production SLA. `research_process_usage.py REPORT COMMAND ...` records OS resource
use without loading weights or changing any research result.

Real-source inventory is independent: `audit_ornl_source_inventory.py` validates
the acquired OEDI909 zip/PDF and reads CSV first lines only. It fits no model and
does not establish actual repair/inspection costs or field-repair success.
MetroPT July B remains unopened. Prior evidence and exports stay intact.
