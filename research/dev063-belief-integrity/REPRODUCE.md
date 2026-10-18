# Recorded semantic-belief and temporal-integrity experiment

Use Python3.12, CPU PyTorch2.14.1, NumPy2.5.4, this source commit and Veritas
`b21d829b62d3cab2811e746267ca305e110ed004`. No fitting is required for replay.
The evidence restorer checks every archive and derived-evidence hash.

```sh
python scripts/restore_belief_dev063.py
python scripts/audit_belief_dev063.py --directory results/dev063-main --output results/dev063-replay.json --exports results/dev063-replay-export --veritas /path/to/pinned/veritas
python scripts/intervene_belief_dev063.py --directory results/dev063-main --output results/dev063-interventions.json --details results/dev063-queries.jsonl.gz
python scripts/analyze_belief_dev063.py --directory results/dev063-main --output results/dev063-statistics.json
```

All raw and EMA checkpoints, pilots, shared DAgger collections, curves and failed
outcomes are preserved. Smoke data use a disjoint namespace and remain excluded.
The main test opens only after the complete matrix is fitted; opening it does
not authorize another fit on that test. See the source-frozen protocol.

Extract `learned-agent.tar.gz`, enter an exported arm's directory, then run:

```sh
python runtime.py < demo-requests.jsonl
```

Runtime imports only PyTorch and the standard library after normal Torch startup.
It receives typed public records and candidates, and returns learned action
scores, health/relation probabilities, action-conditioned outcomes and normalized
cost predictions. It does not execute hardware actions or verify physical success.
The exports use predeclared seed93401 for every arm; they are not chosen for a
favorable test outcome. `tac_rel_belief` is the registered primary, not an assumed
winner. Fresh-seed distribution and failed gates govern qualification.

The model has supplied public cosine identity binding, a learned source-type
admission gate, FIFO64 and a supplied health4 xgraph3 vocabulary. Each decision
reconstructs learned state from the bounded admitted history. Neural CASM here
evaluates a learned mixture of three operators; this is neither symbolic program
discovery nor demonstrated sparse indexed retrieval. The belief-only actor has
no parallel latent-history input. Relative-event arms discard physical durations,
an explicit prior suitable only for this time-independent synthetic task.

Posterior, policy and action-consequence supervision use the training simulator's
public filter/teacher. No filter, repair law, analytical planner or scripted STOP
is included in the export. This remains a bounded teacher-assisted model, not
autonomous discovery of hidden dynamics or a commercial maintenance product.

Local hardware measurements can be reproduced with
`scripts/benchmark_belief_dev063.py`; optional figures require Matplotlib and
`scripts/plot_belief_dev063.py`. No physical DRAM traffic/FLOP/energy claim follows
from timings or retained tensor bytes. Strict cross-host replay checks recorded
float64 decisions on tested hosts, not universal determinism.

History-length strata include explicitly typed noise records, many rejected by
the admission gate. H1024 therefore tests this clock/input-length distribution,
not storage of1024relevant facts, growing entity counts or C5 retrieval scaling.
The gate still examines raw records; rejected writes do not remove that scan.
The task has two entities, three supplied static dependency structures, a fixed
five-action vocabulary and deterministic repairs conditional on prerequisites.
New identities and clock transformations are not new causal laws or real sensors.
