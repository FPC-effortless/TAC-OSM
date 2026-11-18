# Rebuild the frozen ME111 ARM64 Android APK

**Purpose:** recover an installable, hash-verifiable research APK for the Woyengi host-replay bridge. This is a packaging/reproduction instruction only. It does not qualify the model, a phone, a customer workflow, or the original scientific result.

## Frozen model identity

- Original checkpoint: 104,663 learned parameters.
- Checkpoint SHA-256: `337b882e9a10631dc85700846ef2b15ca8908699499668186ad3a248dc108449`.
- ONNX opset: 18.
- ONNX SHA-256: `dd65519a9e4f3388966538af53119b56e9292b46a9209d63ad4927b87f0a176b`.
- Pinned exporter/source commit: `3fc2fbe8e4974491349caecd8d6171e738810b54`.

The build workflow checks out that exact exporter/source commit, recreates the ONNX graph, and fails if its digest differs. It does not retrain or modify weights.

## Rebuild path

The reproducible build is defined in `.github/workflows/model01-android-apk.yml` on draft [TAC-OSM PR #544](https://github.com/FPC-effortless/TAC-OSM/pull/544), branch `product/tac-neural-android-local-runtime-20261028`.

That workflow:

1. Recreates and SHA-checks the frozen ONNX graph.
2. Runs original PyTorch-versus-ONNX host CPU streaming parity and source-feature checks.
3. Generates the original constructed reference vectors and FIT-only source fixture.
4. Builds the universal debug APK and Android instrumentation APK.
5. Checks that the APK has no INTERNET permission and contains the exact ONNX graph plus ONNX Runtime CPU native libraries.
6. Runs Android emulator JNI/state tests and an Android 16 API 36 emulator install/launch check.
7. Builds a smaller ARM64-only APK from the same frozen model/native bytes and verifies APK signature, alignment and packaged asset identity.

On a successful run, the workflow publishes:

- `tac-me111-android-arm64-phone-small-debug` — ARM64-only debug APK, 14-day retention.
- `tac-me111-android-local-cpu-unqualified-debug` — universal debug APK, 7-day retention.

The workflow's final gate explicitly leaves physical Android execution, airplane mode, p50/p95, RAM, energy and phone restart replay as NOT RUN. Emulator success is not physical qualification.

## Trigger and artifact handling

The workflow has a push trigger for this exact branch and paths under `mobile/android/model01/**`, plus a manual-dispatch declaration. A docs-only change in this directory can therefore start the reproducible packaging workflow on this branch. No architecture, checkpoint, ONNX, optimizer, training, benchmark or PNDS files are changed by this note.

After a successful run, record the run ID, artifact ID, downloaded APK byte count and SHA-256. Do not assume a rebuilt APK has the same whole-file hash as an older debug build: build metadata/signing can change while the frozen ONNX asset remains identical. The Woyengi host bridge must pin the exact newly recovered APK bytes before consuming its assets; never weaken the hash check or substitute an unrelated APK.

## Qualification boundary

- Target physical device remains Samsung Galaxy A07 SM-A075F, Android 16.
- Android export/operator compatibility on that phone: NOT RUN.
- Physical numerical parity, p50/p95 latency, peak RAM and energy: NOT RUN.
- Airplane-mode/network trace and physical restart/state replay: NOT RUN.
- ME111 original scientific joint gate: FAILED/MIXED.

This workflow is an offline APK build and emulator test. It is not a product release, source-rights attestation, independent scientific qualification, or permission to deploy.
