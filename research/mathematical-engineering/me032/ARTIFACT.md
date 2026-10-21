# ME032 original checkpoint and gradient evidence

Local archive: `/workspace/me032-centered-memory-checkpoints-and-evidence.tar.gz`.
**79,764,046 bytes**; SHA256
`dc5f9f86a2570d434f2b54d8870e58173727ed245e77df6f48753233964490d6`.
197 indexed payloads plus archive-index.json. See archive.json and per-file index.

Contains all36new original checkpoints with Adam/model/sampler RNG,36trained-model
exports plus9frozen context exports,9R/S/T feature caches/R normalizers,12,600update
loss/CPU receipts and8,100memory-gradient receipts, original forecasts, role/source
manifests, pre-score hashes, full and supplementary audits, code, protocol/math,
results and regression logs. Derived features and public pseudonymous user IDs
come from KuaiRec CC BY4.0, Gao et al CIKM2022 DOI10.1145/3511808.3557220,
Zenodo18164998. Raw source snapshot and original ME031 parent bundle are external
inputs, not newly downloaded source or new fitted bytes in this archive.

Full source replay requires authenticated `results/me022` (12source checkpoints,
raw known-source snapshot and canonical fingerprints) AND original `results/me031`
(parent manifests/R encoder checkpoints and its integrity receipts). Both are
available in this workspace from their original delivered archives. New bundle
pins parent results/encoder hashes and verifies all parent payloads before data.
Only parent R embeddings/preprocessing are inherited; S-trained weights are not.

Independent extraction to `/workspace/me032-restored` verified all197 indexed
payloads, then rebuilt10,474,272context values, checked all36optimizer/RNG records,
reproduced129,312checkpoint forecasts and161,640export/reference forecasts exactly,
verified all frozen/joint behaviors and positive memory gradients, and rejected a
corrupt checkpoint before source loading. All197indexed digests remained unchanged
after replay. This was restoration, not refitting. Original full audit additionally
ran27fresh Torch-only whole pipelines (108forecasts exact) and timing/profiler checks.
Restoration used --quick to avoid repeating those variable resource measurements.

From the final checkout with authenticated parent directories:

```bash
sha256sum /workspace/me032-centered-memory-checkpoints-and-evidence.tar.gz
python -m scripts.me032_audit --out /workspace/me032-restored/results/me032 --quick
python -m scripts.me032_audit --out /workspace/me032-restored/results/me032
python -m pytest -q
```

Python3.12,Torch2.14.1+cpu,NumPy2.3.5,CPU thread1. Original artifact hashes are
checked before source. This is a local downloadable archive, not a GitHub Actions
training artifact. The new study's physical CPU excludes inherited historical
training; charged per-arm costs include it and shared stages. Neither quantity
may be relabeled as the other.

Only to initiate a separate NEW36-fit reproduction, not restoration:
`python -m scripts.me032_conditional_research --out results/me032-new-reproduction`.
Never replace original bytes, change registered endpoints or open reserved cohorts.
