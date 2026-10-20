# ME022 reproduction and recorded-only replay

Execution environment: Python 3.12, Torch 2.14.1+cpu, NumPy 2.3.5, Linux x86-64,
one Torch compute thread, AMD EPYC9 V74 CPU. Matplotlib 3.10.8 is needed only for
the report figure. This preserves the executed environment rather than silently
attributing differences to the older studies' Torch 2.7.1 environment. The first
original-arm score reproduced its historical result to rounding precision.

```bash
python -m pip install 'numpy==2.3.5' 'pytest==9.1.1' 'matplotlib==3.10.8'
python -m pip install 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m scripts.me022_native_readout --self-test
python -m pytest -q
```

Fresh research reproduction explicitly performs12 new native fits and3 unmatched
reference fits. Use a new output directory; never overwrite original evidence:

```bash
python -m scripts.me022_native_readout --out results/me022-new
```

This reads the pinned official publisher archive, then interprets only early
user%16=0 outcomes. It never opens reserved test targets. Original registration
is commit `f88abacbf9b74bcd3d317acc1ac4427e71d2ff87`; execution amendment is
`c7fd9ed` (see EXECUTION-AMENDMENT.md). `--resume` restored the already completed
first original fit; no repeated optimizer update was charged as a new result.

The delivered checkpoint evidence archive excludes the raw/derived event snapshot. It contains
all native checkpoints with optimizer/RNG, final per-event predictions, dense
traces, three tiny-reference weights/scalers/predictions, fingerprints and the
original result manifest. Preserve source attribution and publisher terms.

Recorded-only replay uses the original local snapshot if present. Otherwise
`--rebuild-source` reconstructs **only the same exposed early data** from the
official archive and requires exact canonical tensor/ID/time fingerprints. The
recorded SHA of the original Torch snapshot remains an original-file integrity
record, not a requirement for serialization bytes across machines. No fitting,
model selection or additional test population is involved:

```bash
mkdir -p results/me022-restored
tar -xzf /path/to/me022-checkpoints-and-records.tar.gz -C results/me022-restored
python -m scripts.me022_replay --root results/me022-restored --rebuild-source
python -m scripts.me022_summarize --root results/me022-restored --out results/me022-restored/report
```

Replay exports all 12 native modules with the exact normalization transform.
The attempted float32 fold is retained separately and does not qualify if any
model exceeds the originally fixed1 e-6 probability tolerance.
`torch.jit.load(path)` provides an inference module without
TAC-OSM source imports; call it with original `(items, metadata, action, duration)`
tensors. The exported model retains the 128-event history and analytic prior32.
It is an experimental forecast module, not a controller or deployed agent.

All sampled development scores are historical engineering diagnostics. CPU
figures count actual optimizer-step process CPU, with evaluation/serialization
and source building reported separately. Peak RSS is the process lifetime high
water mark, including acquisition, not isolated per-model activation memory.
Analytical forward affine MACs do not constitute a complete training FLOP audit.
No independent superiority or commercial claim follows from this study.
