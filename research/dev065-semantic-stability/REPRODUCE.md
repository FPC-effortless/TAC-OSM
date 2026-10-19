# Reproduce DEV065 without refitting in CI

Frozen fitting registration/source: `1510f7a` (parent PR222 remains intact).
Main run: `results/dev065-main`; main initialization seeds96501–96520; pilots96500.
Smoke: `results/dev065-smoke`, +10000000 namespace, excluded from performance.

Python3.12, CPU PyTorch2.14.1, NumPy2.5.4. Install TAC-OSM editable. Native replay
also requires the separate Veritas checkout at
`b21d829b62d3cab2811e746267ca305e110ed004`, installed editable. Do not merge these
projects or supply the simulator to a deployed checkpoint.

To replay published evidence after packaging (no training, no July B access):

```sh
python scripts/restore_semantic_dev065.py
python scripts/audit_semantic_dev065.py --directory results/dev065-main --output results/dev065-replay.json --exports results/dev065-restored-agents --veritas /path/to/veritas
python scripts/analyze_semantic_dev065.py --directory results/dev065-main --output results/dev065-replayed-statistics.json
python scripts/intervene_semantic_dev065.py --directory results/dev065-main --output results/dev065-replayed-interventions.json --details results/dev065-replayed-interventions.jsonl.gz
```

The independent audit verifies all manifest SHA256 values, frozen fitting source,
each final EMA checkpoint, fixed predictions, recorded decisions, public simulator
receipts/costs, trained clock invariants and the semantic-only actor interface.
Raw weights are retained and hashed, not substituted for registered EMA results.
The predeclared first seed96501 exports every arm; no best-seed selection.
The portable JSON-lines runtime imports only its bundled model files and Torch.
The isolation/mixing switches are training procedures, not deployed oracles.

Refitting is a separate expensive action: clean checkout of the fitting commit,
then `python scripts/run_semantic_dev065.py --directory results/new-dev065-run`.
It reproduces the fixed recipe and opened scientific benchmark, not a fresh
confirmation study. Never use a rerun to discard bad original seeds or change
promotion thresholds. A new candidate needs a new registration and untouched
evaluation. CI restores recorded evidence only.

Source coverage can be checked independently with
`python scripts/audit_ltpp_followup.py --directory /path/to/acquired-ltpp --output results/ltpp-followup.json`.
This reads previously acquired public tables and adds rehabilitation dates via
the normal public preview API. It never compares condition improvements or fits
a model. It is unrelated to synthetic training and cannot qualify a factory agent.
