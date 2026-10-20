# Executed checks and evidence

- **889 repository tests pass**, including 8 new mathematical/causal/streaming
  tests; 12 subtests pass. The source test-count document is synchronized.
- Seeded native initializer and initial predictions match the established
  native model exactly. Both query variants have 135,562 trainable parameters.
- Analytical state/action mixed derivatives match autograd, the actual module
  matches the stated local partials, and the four-point representability
  falsifier distinguishes conditional and additive log-odds readouts.
- Feedback at and after a prediction prefix cannot affect that prefix forecast.
  Prefix final output and ordinary forward agree exactly. Parameter tensors
  have finite active gradient paths in the constructed numerical check.
- All 12 checkpoints, original standalone exports and FIT-mean-state intervention
  paths reproduce **71,424 native development forecasts per path with zero
  local difference**. Sampler states, mean states, metrics, budget checks and
  checkpoint hashes pass. The recorded-only audit makes zero fits and opens no
  new outcomes. See replay.json for full spectra and numerical fields.
- Each model's 32 real probe windows give zero maximum state/logit error between
  incremental and full-window execution. Irregular chunk boundaries also pass
  the mathematical test. All 12 separate streaming exports are retained alongside
  the original fitting-source exports.
- A fresh process importing only Torch loads the first declared action-prefix
  streaming export and reproduces 4 genuine state trajectories after 128 events.
  Neither `tac_osm` nor `scripts` is imported by that process.
- A corruption falsifier changes one checkpoint in an isolated linked copy;
  the audit rejects its SHA before loading outcome data. Original files intact.
- The 75-file portable archive is independently extracted, each indexed file
  SHA is verified, and the full recorded audit passes again against the original
  authenticated parent source. No raw event snapshot is redistributed.
- Lightweight `import tac_osm` still does not import Torch. Python compilation
  and `git diff --check` pass. New CI runs regression/math checks only, without
  fitting, research-source download or reserved-outcome access.

Checkpoint archive: `me023-checkpoints-and-records.tar.gz`,22,776,795 bytes,
SHA256 `ea10e72cf522e8e24d8a798c43c329d55311deb1fe5032d5219164f5c1f43eb5`.
Its index and receipt are committed; weights remain outside tracked source in
the task workspace under the repository artifact convention. Reproduction and
official-source reconstruction instructions are in REPRODUCE.md.

The mathematical bounds and numerical checks do not turn the failed primary
query gate into a pass. Prefix-objective findings remain secondary historical
development evidence. Full compute parity, independent generalization, reliable
actions, SSM/LLM-scale capability and commercial readiness remain unqualified.
