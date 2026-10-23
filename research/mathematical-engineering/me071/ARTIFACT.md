# Original native models and independent real scoring replay

Use Python 3.12 with `requirements.txt`. Source retrieval and publisher hashes
are in the inherited `scripts/me069_workflow_source.py`; raw BPI2012 records
are not redistributed. The inherited FIT-only five-event count model is
tracked at `research/replay-bundles/ME070/practical-ngrams/FIT_NGRAM_STATE.json`.
Its exact bytes and source identity are charged in FAIR_COMPARISON.json.

The original model/optimizer/Torch and NumPy RNG files for every native seed
are under `research/replay-bundles/ME071/checkpoints/`. CHECKPOINTS.json records
actual exposure, optimizer updates, process CPU and exact SHA-256. Training
source was preregistered at commit `1d0c37c2`, without held-out weight selection.
No architecture-comparison or baseline-training fit is part of this cycle.

The training script first fits all five originals and never invokes DEV
calibration or scoring. Checkpoint bytes must be Git-owned before the
`calibrate` phase. Calibration uses disjoint TEMP/CQR cases and must be committed
before `score` is allowed. ORIGINAL_FORECASTS.npz stores original class
log-probabilities, positive-delay logits and log1p waiting quantiles; original
scores are in ORIGINAL_SCORES.json. No retrained substitutes count as replay.

After checking out the final pinned research commit in a new directory:

```bash
pip install numpy==2.3.5 pytest==9.1.1
pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
export PYTHONPATH=src:.
python scripts/validate_research_parity.py --manifest research/replay-bundles/ME071/FINAL_REGISTRATION.json --phase final
python scripts/me071_persistent_workflow.py --phase replay --cache /tmp/me071-source
python scripts/me071_streaming_probes.py --cache /tmp/me071-source
```

Replay verifies exact original checkpoint bytes against Git, strictly restores
every model and optimizer, loads the frozen calibration, and recomputes actual
REPORT forecast arrays and scores without training/calibration refitting. CI
prints the fresh replay receipt. Different CPUs are checked under the
preregistered log-output and aggregate-metric numerical contract; failures
remain failures and original arrays/scores must not be replaced.

Generated masked/interleaved fixtures and real-record streaming probes verify
execution behavior. They do not establish long-memory advantage, GRU collapse,
causal intervention utility or independent clinical/enterprise reliability.
Raw next events and timestamps cannot certify SLA or financial outcomes.
