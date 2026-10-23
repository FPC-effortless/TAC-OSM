# ME073 original artifact and replay contract

Registration: FAIR_COMPARISON.json, FIT_PREPROCESSING.json, SOURCE.json and
original code frozen in Git before fitting. Five original seeds73001–73005;
99,781 trainable parameters each. No supplied old checkpoint is modified.
Bundle: research/replay-bundles/ME073/; native model/optimizer/Torch RNG,
preprocessing, deterministic schedules and trace for each seed, original log,
CHECKPOINTS.json, FINAL_REGISTRATION.json, environment, original distribution/
representative path/suffix-likelihood arrays, proper scores and exact SHA index.

Commands from repository root, dependencies in requirements.txt:

```sh
PYTHONPATH=src:. python scripts/me073_autoregressive_world.py --phase register --cache /tmp/me073-source
python scripts/validate_research_parity.py --manifest research/mathematical-engineering/me073/FAIR_COMPARISON.json --phase preflight
PYTHONPATH=src:. python scripts/me073_autoregressive_world.py --phase train --cache /tmp/me073-source
# Commit original weights before original REPORT scoring.
PYTHONPATH=src:. python scripts/me073_autoregressive_world.py --phase score --cache /tmp/me073-source
# Commit original predictions/scores before independent checkout replay.
PYTHONPATH=src:. python scripts/me073_autoregressive_world.py --phase replay --cache /tmp/me073-source
```

Registration/training commands are historical reproduction, **not** replay.
Never rerun them over original files. Replay restores Git-owned original
weights and optimizer state without fitting; promotes weights to float64 for
stable deterministic QMC integration; verifies recorded source/preprocessing
identity, array/score identities and prefit numeric tolerances. It restores
original output bytes even on failure and retains failed replay outputs.
Recoverable failed fit state is saved and must not be erased.

Raw external source retrieval and publisher terms are compulsory; unavailable
source means full score replay is BLOCKED. A clean checkout with indexed model
bytes and clean GitHub scoring replay is required before REPO_REPLAY=PASS.
Training continuation/full deterministic training replay is not independently
claimed merely because state/order is preserved. Scoring evidence is exposed
DEV only. EOR is a retrospective record proxy; no sealed TEST is scored.
