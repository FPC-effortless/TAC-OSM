# ME024 reproduction and artifact contract

The fitting source was frozen and pushed at `969a92e` before the first fit.
The four variants share 135,050 parameters, identical sample draws per seed and
four causal response targets per window. Read PROTOCOL.md and MATHEMATICS.md
before interpreting results. This is a historical development experiment.

Environment: Python 3.12, Torch 2.14.1+cpu, NumPy 2.3.5, pytest 9.1.1;
Matplotlib 3.10.8 for figures. Install the repository with `pip install -e .`.
The library remains importable without Torch; the research core is optional.

From the repository root, recorded-only replay requires the ME022 authenticated
snapshot and its twelve original checkpoints under `results/me022`, the twelve
ME023 original checkpoints and manifest under `results/me023`, and the delivered
ME024 archive extracted into a new directory. Never substitute another cohort
or fetch reserved outcomes to make replay pass.

```bash
python -m scripts.me024_audit --root /workspace/me024-restored
python -m scripts.me024_summarize --root /workspace/me024-restored --out /tmp/me024-summary
python -m pytest -q
```

The audit validates all ME024 checkpoint/export hashes before loading any data;
then all 24 parent checkpoint hashes, the original dataset hash and canonical
fingerprints. It reconstructs all sampler draws, verifies exposure, budgets and
trace steps, replays all 71,424 forecasts through checkpoint, standalone,
feature-mean and feedback-disabled paths, and checks streaming/window parity
on 32 genuine windows per model. It computes float64 full chronological
Jacobian products on exactly the first eight FIT windows per model. No training
or new test access occurs. Output filenames are `replay.json` and
`jacobian-probes.json`; preexisting copies may be overwritten by the replay.

To repeat training deliberately, use a fresh output directory. These are
additional fits and must not be represented as the original registered run:

```bash
python -m scripts.me024_coupled_research --out results/me024-independent-rerun
```

The recorded archive contains original final weights, Adam moments, Torch and
sampler RNG, normalization, executable standalone models, feature means,
probabilities, learning traces, manifests and post-fit diagnostics. It excludes
raw source records. SHA-256 indices and the archive checksum are in ARTIFACT.md.
All negative results remain in the artifact; none are selected away.

The streaming API takes one actual event plus a caller-owned `[batch,64]` state:

```python
state = torch.zeros(batch, 64)
state = model.stream_step(item, metadata, state)
logit = model.readout(state, next_item, next_duration, causal_prior32)
```

Maintain the last 32 observed responses separately for the analytic prior.
The 64 float32 state coordinates occupy 256 bytes per entity, excluding that
buffer, parameters and runtime workspaces. State must remain with its entity.
Do not feed a target response before forecasting it. Streaming equivalence is
verified for recorded 128-event windows, not predictive quality on longer
streams. No associative scan or constant-memory training is claimed.

TorchScript provides the tested standalone runtime in this pinned environment.
Upstream marks that API deprecated; migration to another exporter is separate
engineering work and has not been validated by this experiment.
