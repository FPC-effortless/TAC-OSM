# ME111 frozen 100K native TAC neural model on Android: offline integration

**Draft engineering milestone. No real phone has run this implementation. Scientific prediction gate remains FAILED/MIXED.**

This stack is based on [Model01 Android PR #539](https://github.com/FPC-effortless/TAC-OSM/pull/539), with the original ME111 neural checkpoint and versioned ONNX exporter sourced from [PR #541](https://github.com/FPC-effortless/TAC-OSM/pull/541). The first family-wide device-only contract remains [PR #538](https://github.com/FPC-effortless/TAC-OSM/pull/538). Historical checkpoints, research results, and family count are unchanged.

## Exact model and local boundary

- Frozen original model: **ME111 conditional belief TAC**, seed **111001**, **104,663 active trained/deployed neural parameters**.
- Original trained `final.pt` SHA256: `337b882e9a10631dc85700846ef2b15ca8908699499668186ad3a248dc108449`.
- Derived fixed B=1/T=1 ONNX opset18 model SHA256: `dd65519a9e4f3388966538af53119b56e9292b46a9209d63ad4927b87f0a176b`, **726,515 bytes** in host export.
- Original pinned source/exporter commit: `3fc2fbe8e4974491349caecd8d6171e738810b54` from original ME111 research branch (no model fit; original real outcome data not opened).
- Android ONNX Runtime AAR version `1.23.2`, CPU-only; `OrtEnvironment.createSession(byte[])` from packaged asset, no runtime downloads.
- Local input tensors: `values [1,1,13] float32`, `observed [1,1,13] bool`, `ages [1,1,13] float32`, `clock [1,1,6] float32`, `owners [1] int64`, `hours [1,1] int64`. Source-aligned normalized features must originate from an authorized real station; UI's prefilled zeros are explicitly constructed-only diagnostics.
- Local persisted state: `source_sum [1,13]`, `source_mass [1,13]`, `evidence_sum [1,62]`, `count [1,1]`, `memory [1,8,8]`, `last_hour [1]`, and source owner. The app uses a model-fingerprint-bound `AtomicFile` in app-private storage. State is never put in a network request or remotely updated.
- Typed outputs: `log_location [1,6]`, `left_scale [1,6]`, `right_scale [1,6]`, nine derived state/chronology outputs. Values are in **standardized log1p pollutant units**, NOT raw micrograms/m³ or a qualified physical prediction. No independent calibration guarantee and **no action authority**.
- No Android `INTERNET` permission, other network clients, remote embeddings, speech pipelines or hidden fallback.

## Build from exact original model bytes (CI does this)

CI checks out immutable commit `3fc2fbe8e4974491349caecd8d6171e738810b54` into `pinned-me111`, loads the original checkpoint and reproduces the ONNX graph/CPU parity from source, checks exact derived SHA256, then places it at:

`mobile/android/model01/sample-app/app/src/main/assets/tac_me111.onnx`

The model asset is intentionally not checked into this draft branch as plain text. The build **fails closed** when the pinned asset is missing or corrupt; it does not fetch/download during inference. After synthesis, Gradle packages the real ONNX model and the pinned native ONNX Runtime CPU library into a debug APK. CI unpacks the APK and verifies the embedded model SHA256, native runtime libraries, and no `INTERNET` permission. Assets produced solely by temporary Actions artifacts are **not** a substitute for the original trained checkpoint, which remains Git-owned on the pinned research branch. For eventual release, persist exact packaged bytes and model manifest in the designated repository-owned release archive, not an expiring CI artifact.

Host-JVM constructed tests of `Me111Contract.java` check source/ownership, NaN/Inf, masks, shapes, monotonic hour, memory bounds and predictable input errors. **Passing Java compilation or static APK permission checks does not prove physical Android inference.**

## Device science and engineering gates — unrun

Target Android phone: **Samsung Galaxy A15 4G**, verify exact SoC, RAM, OS and ONNX Runtime version on device. Target iOS: iPhone 13 (**NOT RUN**, separate native iOS build required). Embedded: Raspberry Pi 5 8GB (**NOT RUN**, separate CPU package required).

The first phone qualification MUST include installation/start, actual ONNX session creation, at least 3 meaningful source-authorized streaming events and numerical comparison with original source PyTorch reference, input/station/time guard tests, local state restart recovery, physical airplane mode with zero inference-related networking, cold/warm p50 and p95 latency, peak RSS/Java+native RAM, energy/battery measurements with explicit method, and thermal profile. Capture raw traces, checkpoint IDs, OS/SoC, run counts and failures; no invented measurements.

**Known original scientific failures:** ME111's untouched source-development evaluation showed missing-value PM2.5 coverage **86.43%** versus nominal 90%, NO2 coverage **71.16%**, full joint legacy gate **FAIL across 5/5 seeds**. This local export does **not** fix calibration, prove useful new-site prediction, or qualify an intervention decision.

## Acceptance status

| Gate | At implementation creation |
| --- | --- |
| Original model exists with SHA-pinned in-repo training bytes | Previously established in ME111 research |
| Host ONNX CPU 3-step source-graph parity | PASS in prior PR #541 CI |
| Typed Android source and state validator | Implemented; CI verification required |
| Android neural wrapper and app-private state persistence | Implemented; APK/phone verification required |
| Android APK built containing exact ONNX graph and native CPU runtime | NOT RUN until integration CI concludes |
| Android native JNI inference, restart-state, airplane-mode | NOT RUN |
| Physical p50/p95, peak RAM, energy | NOT RUN |
| Original real-world independent scientific suitability | FAIL/MIXED for ME111, no new independent evaluation |
