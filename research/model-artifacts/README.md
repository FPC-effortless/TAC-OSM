# Original trained native TAC models

`me038-native-inference-models.tar.gz` contains the original 60 validation-selected
native ME038 physical-input models and their replay probes. The bytes are identical
to the independently restored local delivery; no retraining or changed forecasts.
48 models are from the exposure lane and 12 from the separate CPU-budget lane.
This explicit repository artifact is intentionally tracked rather than treated as
regenerable seed-only output: timed-training weights cannot be reproduced from
seeds alone. No Git LFS is required for this 30.2 MB file.

SHA-256: `3bb753071dcc544919437aecf54eedd015da13e89aa61c6e15ae508d1e8c6d04`.

From the repository root:

```sh
python scripts/verify_research_archive.py research/model-artifacts/me038-native-inference-models.tar.gz 3bb753071dcc544919437aecf54eedd015da13e89aa61c6e15ae508d1e8c6d04
```

The archive is also uploaded by the ME039 CI workflow as the
`original-native-models-and-me039-records` Actions artifact after hash checks. Git retains
the original file; the Actions copy has 30-day retention. See the actual CI receipt
in ME039 for artifact ID and download verification, rather than assuming a run.

Extract with tar's safe data filter, then run `python verify.py` with the recorded
Torch/NumPy environment for same-CPU forecast replay. Serving requires raw Bx96xC
input in data.json channel order; output is BxHxC in original publisher units.
No TAC package is needed to load these TorchScript models. Keep the original
serialized serving format: ME039's alternate scripted-core format was slower.

[Scores, gates and limitations](../mathematical-engineering/me038/RESULTS.md) ·
[Native equations](../mathematical-engineering/me038/MATHEMATICS.md) ·
[Current execution research](../mathematical-engineering/me039/PROTOCOL.md)

The complete 1.05 GB ME038 optimizer/checkpoint/source archive remains a verified
local workspace delivery. Release metadata writes succeeded, but GitHub's upload
host rejected configured credentials (HTTP401). The empty draft release was
removed. No full-release-asset download is claimed; its checksum/location are in
me038-repository-delivery.json. No credentials were extracted or interactive login
started. These are unmerged research artifacts, not promoted commercial models.

The additional ME039 archives contain every original numerical continuation:

- `me039-ETTh1-numerical-records.tar.gz`: 24 final optimizer/RNG/model states,
  their traces and all 1/16/64 gradient/parameter snapshots.
- `me039-electricity-numerical-records.tar.gz`: the corresponding 24 records.
- `me039-experimental-serving-exports.tar.gz`: 60 alternate scripted-core exports
  and probes. These are experimental and slower than the original serialized
  models; they are preserved for reproducibility, not recommended as replacements.

All sizes/SHA-256/payload counts are in me039-archives.json. Files are kept below
GitHub's 100 MB single-file limit; no LFS or private external storage is required.
The numerical continuations are engineering probes, not new capability models or
new accuracy training results. A separate freeze candidate failed exact replay
and is not included as a recommended serving model.
