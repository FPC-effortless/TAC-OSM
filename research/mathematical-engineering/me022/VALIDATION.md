# Executed validation

- Native algebra/capacity/gradient preflight: PASS. Includes nonzero initialized
  head reparameterization, script/eager equality, matched seeded initial weights
  and finite nonzero gradients for every parameter tensor.
- `python -m pytest -q`: **881 passed**, 12 subtests passed; one inherited Torch
  scalar-conversion warning. The baseline test-count documentation was stale
  (859) before this work; synchronized to the actual 881 without changing tests.
- `python -m scripts.me022_replay`: all 12 native and 3 reference checkpoints replay;
  **71,424 native predictions match exactly**, along with reference scores and
  predictions, source bounds, sampler states, parameter/update/exposure budgets.
- All 12 exact standalone exports match all 5,952 development forecasts each with
  **zero local difference**. A separate fresh process using only Torch loads the
  combined first-seed export and reproduces 16 genuine forecasts; no `tac_osm` or
  `scripts` imports are loaded.
- Float32 folded exports: **11/12** satisfy the unchanged 1e-6 probability guard.
  Maximum error 1.1324882507324219e-6. A float64-coefficient-construction diagnostic
  with float32 deployment did not repair the failing seed. Fold qualification
  remains **FAIL**; accepted exports keep the exact normalization operation.
- Portable archive independently extracted into `/workspace/me022-restored`;
  all 69 indexed input/output file hash matches. Original local source snapshot
  is linked only after verifying its stored hash; all recorded replay checks
  pass again. No additional fitting or reserved outcomes.
- Corruption falsifier: replace one checkpoint in an isolated linked copy;
  replay rejects its SHA before reading outcome tensors. Original files intact.
- Python compilation and `git diff --check`: PASS.

The 22,435,762-byte archive SHA256 is
`e64dc94945835f4d39300e809725043216f38758b992d2518eb9bcef26fb1c2a`.
It is delivered as `me022-checkpoints-and-records.tar.gz` in the task workspace,
with 70 files including its index. Weight artifacts remain outside tracked source
as required by the repository's artifact convention. It excludes the derived
event snapshot; official early-source reconstruction is documented. The source
manifest, fingerprints, raw metrics, dense curves, archive index and replay
receipt are committed for review. See ARTIFACT.json and artifact-index.json.

New CI is preflight/regression-only. It downloads no research data, fits no
models and opens no tests. Remote check status is separate from these completed
local checks. This validation is reproducibility evidence, not a scientific
architecture superiority or reliability qualification.
