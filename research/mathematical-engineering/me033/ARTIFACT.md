# ME033 original fitted models and exact-storage engineering evidence

Local archive `/workspace/me033-semantic-research-checkpoints-and-evidence.tar.gz`:
**98,078,224 bytes**; SHA256
`203133e0ff77c343b63bc83fe144417ab399b597ed1d7187309540a3bf7dd62c`.
289 indexed payloads plus archive-index.json. See archive.json and per-file index.

Includes all54original S-stage checkpoints/Adam/RNG and exports,54step traces
(16200updates),9inherited R teacher/context records,original predictions/roles,
source/frozen hashes,9byte-identical inherited context exports,36streaming and
18frozen-feature-reconstruction receipts,fresh-process checks,profiler/timings,
code,protocol/math,results,953test log and18optional shared-table runtime exports.
The shared exports are post-fit exact-execution artifacts, not new fitted models.
Original unshared weights are preserved alongside them.

Real-derived feature tensors and public pseudonymous user IDs are from KuaiRec
CC BY4.0,Gao et al CIKM2022 DOI10.1145/3511808.3557220,Zenodo18164998. Raw source
and full ME032 parent bundle are external authenticated inputs. Source replay
requires `results/me022` original12checkpoints/snapshot/fingerprints AND original
`results/me032` manifests,context/teacher records and its integrity receipts.
Those delivered parent artifacts already exist in this workspace. New archive
pins their hashes; no new cohort/data download is part of this study.

Independent extraction to `/workspace/me033-restored` verified all289indexed
payloads before source replay. It rebuilt6982848context values,verified all54
optimizer/sampler/step records,replayed193968checkpoint and226296export/reference
forecasts exactly,rechecked real moment/merge and frozen-latent reconstruction,
and rejected a corrupted checkpoint before source. All18shared runtimes then
replayed64656forecasts exactly with aliases intact. All289original file digests
remained unchanged after both audits. No refitting occurred.

Original full audit additionally used54fresh Torch-only full pipelines (216real
forecasts exact) and collected inference/profiler estimates. Restoration used
--quick to avoid remeasuring those variable timings; it still repeats source,
streaming and forecast verification. Shared-storage verification uses the already
serialized exports, never regenerating/replacing their original bytes.

With final source checkout and authenticated parent directories:

```bash
sha256sum /workspace/me033-semantic-research-checkpoints-and-evidence.tar.gz
python -m scripts.me033_audit --out /workspace/me033-restored/results/me033 --quick
python -m scripts.me033_shared_audit /workspace/me033-restored/results/me033 --verify-existing
python -m pytest -q
```

Python3.12,Torch2.14.1+cpu,NumPy2.3.5,CPU thread1. Original payload digests are
checked before source; shared exports additionally check their own saved hashes
before source. This is a local archive, not a GitHub Actions training-artifact
claim. Component state bytes and tensor-weight savings exclude other forecast
buffers/activations; no total-system memory reduction is inferred.

Only to initiate a separate NEW54-fit reproduction, not original restoration:
`python -m scripts.me033_semantic_research --out results/me033-new-reproduction`.
Never replace original checkpoints or add post-hoc arms/open reserved cohorts.
