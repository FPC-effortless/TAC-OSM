# ME034 original checkpoints and self-contained offline replay archive

Local delivery: `/workspace/me034-stable-input-checkpoints-and-evidence.tar.gz`

- Bytes: **145074479**.
- SHA256: **797b90a6bffe0294b0e38fdf4c8f465b5b012087e545e7ec26dd0e3654505839**.
- Indexed payloads: **1081**, plus internal `ARCHIVE-INDEX.json`.
- Source snapshot: `aaede6f4e8e21387abb61e62e82145bf85bd6aea`.
- Original fit/protocol freeze: `260fd62b6ea60cec9adefd3410e8d2160a2817e8`.

This is an actual local workspace archive, not a GitHub Actions training artifact.
The repository contains reports, hashes and audit receipts; original model bytes
are delivered in this archive. Its final digest is recorded independently in
[archive.json](archive.json); every payload is listed in [archive-index.json](archive-index.json).
Circular archive metadata and later restoration receipts are published outside
the bundle and excluded from its source snapshot.

## Included original evidence

All63neural checkpoint files with Adam states and sampler/model RNG, all63original
scripted students,63portable raw-input pipeline exports,9serialized original
boosted models,9inherited R-context caches and9context exports. All losses,
gradient/freeze receipts, CPU traces, original CV/final prediction arrays,
final ensemble arrays, final source-derived snapshot, access receipt and audit
records remain intact. No weights are reconstructed by retraining.

Offline support includes the original authenticated ME022 train/DEV snapshot,
its12original checkpoint files and manifest required by source verification,
9original R-only ME031 encoder checkpoints and9original ME032 R-common teacher
checkpoints. The original parent result manifests and repository source are
included. This suffices for ME034 replay without the originating workspace or
network. It does not include every unused artifact of every ancestral study.
Full historical-study refitting is separate and is not required for this replay.

The431964858-byte publisher ZIP itself is not bundled; authenticated source
arrays and publisher provenance are. KuaiRec observations retain the source
CC BY4.0 attribution to Gao et al., CIKM2022 DOI10.1145/3511808.3557220,
Zenodo18164998. Source ZIP SHA256
c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c.

## Offline restore

Verify the archive digest, extract into a fresh directory, then verify every
listed file against `ARCHIVE-INDEX.json`. Use the pinned environment in
[environment.json](environment.json): Python3.12,Torch2.14.1+cpu,NumPy2.3.5,
scikit-learn1.8.0,SciPy1.17.0,joblib1.5.3,threadpoolctl3.6.0.

From the extracted root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD/src" \
  python -m scripts.me034_audit --out "$PWD/results/me034" --quick
```

This verifies original hashes BEFORE source loading; reconstructs all63neural
models from checkpoint state; checks optimizer clocks, sampler RNG, frozen
tables and original R-only teacher/encoder hashes; rebuilds all R/S/T features;
loads all63exported raw pipelines; replays CV/final forecasts and9trees;
reconstructs final ensembles and both adjusted bootstrap intervals. Full audit
without `--quick` additionally executes every raw-input prediction path, fresh
Torch-only processes, operator profiling and latency measurements.

Never rerun `--stage final` as a new test: remainder6 has been consumed. No fitting
or label selection is performed by replay. Tree pickle files are original,
hash-verified local artifacts loaded under their pinned library environment.

Independent extraction and replay receipts appear in
[restoration.json](restoration.json) and [restoration.log](restoration.log).
Restoration is a separate filesystem/source copy on the same CPU environment,
not independent hardware or a new statistical replication. All original indexed
payloads are checked again after replay to detect accidental mutation.
