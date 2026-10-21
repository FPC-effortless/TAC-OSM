# ME031 original checkpoints and cross-fit evidence

Local archive: `/workspace/me031-crossfit-checkpoints-and-evidence.tar.gz`.
Size: **99,705,991 bytes**. SHA256:
`67f5a8bd7e0bccdf1f47556c5c9eb80b623343e52088835d5072941003b76b52`.
180 indexed payloads plus archive-index.json; see archive.json and archive-index.json.

Contains36 original weights/Adam/RNG checkpoints,36TorchScript exports,9frozen
feature caches/S-only normalizers,16,200 update receipts, exact role indices,
original forecasts, pre-score frozen digests, full audit, source files, test logs
and preserved pre-fit failure. Derived features, feedback forecasts and public
pseudonymous user IDs originate from KuaiRec CC BY4.0, Gao et al CIKM2022,
DOI10.1145/3511808.3557220, Zenodo18164998. The raw source snapshot is not included;
full replay requires the authenticated ME022 parent cache and its original source
checkpoint/fingerprint receipts, as already available in this workspace.

The archive was independently extracted to `/workspace/me031-restored`; all180
indexed digests passed before replay. Full feature reconstruction,36optimizer/RNG/
clock audits,64,656checkpoint residual forecasts,129,312export/reference forecasts,
metrics/contrasts and pre-source corruption checks passed again. Replay modifies
no original indexed payload. The first full audit additionally ran9fresh-process
whole-pipeline tests (72forecasts, bit exact) and measured profiler/inference costs.
Restoration used `--quick` to avoid rerunning benchmark/fresh-process measurements.

From the final research checkout, with authenticated parent cache in results/me022:

```bash
sha256sum /workspace/me031-crossfit-checkpoints-and-evidence.tar.gz
python -m scripts.me031_audit --out /workspace/me031-restored/results/me031 --quick
python -m scripts.me031_audit --out /workspace/me031-restored/results/me031
python -m pytest -q
```

Python3.12, Torch2.14.1+cpu, NumPy2.3.5; CPU thread count1. Checkpoint/forecast hashes
are verified before source loading. These are local original bytes, not a GitHub
Actions training-artifact download. No new training is required for restoration.

Only to initiate a separate NEW36-fit reproduction, never replacing originals:
`python -m scripts.me031_crossfit_research --out results/me031-new-reproduction`.
This stays inside original FIT and does not authorize opening reserved cohorts.
