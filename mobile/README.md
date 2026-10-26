# Mobile Edge reference — package and real-device qualification

**Status: contract + CI identity/tamper validator only. No mobile model exported, no Android device test, no production release.**

The native six-family TAC-OSM work remains under the [master research plan](../research/MASTER_RESEARCH_PLAN.md). This directory documents the first Edge-Lite engineering preflight. It is not an alternate seventh model or permission to train a different parameter scale.

## Model package layout

Provide a directory containing:

- mobile-manifest.json: exact model format, local file, SHA256, byte count, original checkpoint SHA256 and model-source commit.
- model.onnx or a supported single-file ExecuTorch/Core ML export path. **The validator checks the file's identity, not validity of an ONNX/Core ML model or whether its operations execute on a phone.** A real exporter must be source-specific and independently audited.
- Optional device.json (a separately hashed measurement receipt from an actual phone; see below).

Minimal manifest shape (PLACEHOLDERS MUST BE REPLACED with genuine bytes/hashes/metadata):

```json
{
  "schema_version": 1,
  "model": {
    "format": "onnx",
    "path": "model.onnx",
    "sha256": "REPLACE_WITH_REAL_64_HEX_DIGEST",
    "bytes": 12345,
    "original_checkpoint_sha256": "REPLACE_WITH_ORIGINAL_64_HEX_DIGEST",
    "source_commit": "EXACT_GIT_SHA",
    "trainable_parameters": 100000,
    "deployed_parameters": 100000
  },
  "interface": {
    "inputs": [
      {"name": "observation", "dtype": "float32", "shape": [1, 14]},
      {"name": "prior_state", "dtype": "float32", "shape": [1, 166]}
    ],
    "outputs": [
      {"name": "forecast", "dtype": "float32", "shape": [1, 14]},
      {"name": "next_state", "dtype": "float32", "shape": [1, 166]}
    ],
    "state_pairs": [{"input": "prior_state", "output": "next_state"}],
    "reset_semantics": "Source-model-specific verified state reset method"
  },
  "scientific_evidence": {
    "status": "FAILED",
    "evidence_ref": "Exact original research PR / study result URI"
  }
}
```

These names and shapes are **illustrative**, not an exported real System 1 signature. The real source graph may require actions, normalizer buffers, as-of masks and more state tensors, which must all be reflected in the manifest.

## Run the package preflight

```sh
python scripts/validate_mobile_edge_package.py /path/to/package
python scripts/validate_mobile_edge_package.py /path/to/package --require-device
python -m unittest discover -s tests -p 'test_mobile_edge_package.py' -v
```

Exit 0 checks only manifest structure, fixed parameter policy, file integrity, and (when requested) the format of an optional hash-checked device report. Exit 2 is invalid or blocked. It never certifies trained-model correctness, scientifically beneficial skill, signed real-device provenance, batch/stream parity, action authority or actual mobile execution.

The identity verifier is deliberately dependency-free, offline and no-fit. Tests use **constructed bogus model bytes** to exercise hostile paths, tampered files, missing state, unacceptable scales and spoofable device receipt formats; those fixtures cannot support any ML or hardware performance claim.

## Device report shape (not independently authenticated)

If measurement was genuinely taken on physical hardware, write and SHA256-pin a JSON file with the model hash, "source": "physical-device", "device_model", "android_version", "execution_backend", "benchmark_script", an array of at least 30 finite positive "latency_ms" observations and positive "peak_rss_bytes". Then add "device_measurement": {"path": "device.json", "sha256": "..."} to the package manifest. The validator outputs RECEIPT_FORMAT_PASS_INDEPENDENT_AUTHENTICATION_OPEN: **self-reported data is not trustworthy signed device qualification**.

Independent Android/iOS qualification must additionally check:
1. Original checkpoint bytes and exact legal source + as-of input snapshots, with a source-owned exporter and full computational graph operator support (not silent CPU fallbacks).
2. Output and recurrent state agreement between original Torch FP32/64 and the exported runtime across multi-step authentic histories; explicit numerical tolerance and missing-value cases.
3. Reset/batch/stream equivalence; no future-state leakage; source-authorization contracts; no model operation changing authoritative app or Woyengi state.
4. Real device identity, OS/SoC, backend, compiler/quantization, cold/warm p50/p95, peak resident RAM including activations/caches/runtime, battery/energy methodology, offline network-disabled verification and independent reproducibility.
5. Published complete scientific fail/pass gate and limits **separately** from device statistics. Current native System1/Temporal candidates are not qualified.

A real Android app, tested exporter, hardware results and independently signed evidence are follow-on engineering tasks requiring actual devices and model-specific source work. This is a preflight foundation, not an SDK release.
