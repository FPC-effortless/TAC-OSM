# TAC-OSM ME111 Android device qualification protocol — UNEXECUTED

**Status: PROPOSED physical test protocol, not physical evidence.** This protocol qualifies only fixed B=1/T=1 ME111 device execution, not ME111's FAILED original scientific coverage gate. Source inputs must remain independently authorized, physically available at forecast time, standardized using the original FIT-only rules and never selected on protected TEST. No customer actions are authorized.

## Target platform and reproducibility boundary

First proposed physical device: Samsung Galaxy A15 4G (record verified exact model, SoC, OS version, memory, available storage, thermal/power conditions, battery state, ABI, ONNX Runtime version, network state). Android debug APK must come from a pinned audited commit; inspect the exact deployed package—not just the workflow's source checkout.

- Expected original neural checkpoint: `337b882e9a10631dc85700846ef2b15ca8908699499668186ad3a248dc108449`, 104,663 active trainable/deployed parameters.
- Expected derived ONNX graph: `dd65519a9e4f3388966538af53119b56e9292b46a9209d63ad4927b87f0a176b`, opset18, fixed B1/T1, fully local CPU.
- App package: `org.tacosm.mobile.model01.demo`. Developer model demos are not qualified real-data outcomes.
- Source boundary: original normalized 13 observations and masks, 13 ages, six clock fields, source station 0–11, source-naive monotone hour. Do NOT treat phone wall-clock time, synthetic zeros or random sensor values as authentic observations.
- State boundary: one owner's source statistics, evidence, count, 8×8 associative memory, monotone hour persisted in app-private atomic storage with model fingerprint. No remote embeddings, inference, speech, state updates or fallback.

## Physical execution gate (required)

1. Build and SHA-pin the actual APK, examine its merged `android.permission.INTERNET` list and packaged ONNX `assets/tac_me111.onnx` SHA256. Install on the named physical Android phone using `adb install -r <debug.apk>` (avoid uploading any personal observations as build fixtures).
2. Install/execute the Android instrumentation APK on that phone (the test bundle and model fixture must come from the same pinned source export). Verify local JNI inference, original PyTorch-vs-ONNX comparisons, source owner and chronology rejects, numerical finiteness, model SHA checks and private atomic state save/restore. A constructed fixture qualifies **software equivalence**, not real-world model success.
3. With actual source-authorized observations, run non-overlapping temporally valid events and compare each output to frozen PyTorch with preregistered per-output absolute/relative tolerances; record count, owner, missingness, clock, exact shape, state continuity and all channel/horizon failures. Do not open protected TEST for tuning.
4. Force-stop and relaunch the app; prove it restores the **same** legitimate owner, state, count and hour, then rejects duplicates/gaps per the contract. Corrupt local state must fail closed, not silently reset.
5. Enable **physical airplane mode using phone Settings** and independently confirm mobile data, Wi-Fi and Bluetooth are disabled; run fresh and carried-state inference, force-stop/restart, and show device logs contain no model-related networking or fallback. The no-INTERNET APK permission is necessary but not alone a complete user-controlled-device audit.
6. Preregister measurement cohort/warmup (e.g. 100 warm single-step operations per device state and at least 20 cold loads); record every latency with a monotonic clock. Report p50/p95, count, min/max and distribution; separate model-load, input-validation, inference, serialization and UI times. The tentative p95 under 150 ms is an engineering target, **not a measurement**.
7. Record peak app/native RAM during cold/warm inference using a documented Android profiling method; distinguish Java heap from native allocations/total PSS. Proposed incremental peak <128 MiB is a target. Record battery/thermal settings, CPU governor/accelerator and memory pressure.
8. Measure energy with a documented device power measurement method, uncertainty/idle baseline and repeated runs; `batterystats` alone cannot be silently promoted to accurately measured per-inference joules. If no instrument exists, label ENERGY=NOT_MEASURED.
9. Preserve raw logs, operator/ABI inventory, original model and APK SHA256, independent tester, OS version, source rights, exact test code, and FAIL/NOT RUN for every incomplete gate. Veritas reviews independently before product qualification.

## Explicit status after APK and host-JVM CI, before physical tests

| Gate | Status |
| --- | --- |
| Frozen original Python/ONNX CPU 3-step streaming parity | PASS (host CI) |
| Android debug APK compiles with exact ONNX asset and native AAR | PASS (build CI) |
| No INTERNET permission in built APK | PASS (static build CI) |
| Android emulator JNI and private state test | CHECK CI; NOT presumed |
| Actual Samsung Galaxy A15 4G installation and native inference | **NOT RUN** |
| Physical airplane-mode operations and network traces | **NOT RUN** |
| Physical p50/p95 cold/warm latency | **NOT RUN** |
| Physical peak RAM and energy | **NOT RUN** |
| Independent real-source predictive/calibration suitability | **NOT QUALIFIED / historical ME111 full gate FAILED** |

Never replace missing physical evidence with benchmark estimates, artifact hashes, emulator-only status, or reference source scores. iOS (iPhone 13) and embedded (Raspberry Pi 5 8GB) require separate actual platform builds, input/state boundaries and equivalent physical tests.

## Repeatable on-phone CPU benchmark harness — IMPLEMENTED, NOT RUN

The dedicated Android instrumentation class `org.tacosm.mobile.me111.Me111PhysicalBenchmarkAndroidTest` rejects emulator fingerprints and requires airplane mode before local inference. It uses **constructed numerical observations** to measure hardware CPU costs, not scientific predictive accuracy or source availability.

On a named, authorized physical device connected by ADB, after the frozen-model asset is generated and the APK is built, enable airplane mode in Settings, then run:

```sh
adb devices -l
adb shell settings get global airplane_mode_on
gradle -p mobile/android/model01/sample-app :app:connectedDebugAndroidTest --no-daemon -Pandroid.testInstrumentationRunnerArguments.class=org.tacosm.mobile.me111.Me111PhysicalBenchmarkAndroidTest
adb shell run-as org.tacosm.mobile.model01.demo cat files/me111-physical-cpu-benchmark.json
```

The test records frozen model hashes, real device identity, model-load time, 20 warmups, 150 warm predictions, warm p50/p95/max, total app PSS before/peak, and atomic local state restore. It saves a JSON receipt only to app-private storage. The test leaves explicit **ENERGY=NOT_MEASURED**, independent network trace **NOT_CAPTURED**, radio isolation beyond airplane-mode setting **NOT_INDEPENDENTLY_VERIFIED**, and process restart **NOT_MEASURED**; these require separate independently observed measurements. The physical-only class is excluded from emulator CI.


## On-phone, ADB-free CPU self-test (research diagnostic)

The neural demo now has a user-triggered offline phone benchmark with no laptop required. It rejects emulators, requires the airplane-mode setting, and runs 20 warmups and 150 timed CPU predictions using CONSTRUCTED scratch-only observations. It does not read, reset, or modify persistent customer state. The result can be copied from the screen only after an explicit button tap. Reported metrics: model/session initialization, warm p50/p95/max latency, sampled total app PSS and exact frozen model/device identity. The app cannot independently verify all radios or measure energy, process restart, true cold process launch, or authentic forecast utility. These remain NOT_MEASURED/NOT_QUALIFIED. Airplane Mode is a prerequisite, not proof of network silence. This research debug APK is not a signed public deployment.

## State integrity, limits and migration

The bounded fixed-length 734-byte v2 persistent state includes model fingerprint, typed source/associative-memory state and SHA256 checksum. It rejects single-bit corruption even when altered float values remain finite and plausible. Android AtomicFile handles committed crash-safe writes and allowBackup=false prevents ordinary Android backup. The earlier unsigned v1 format is deliberately NOT silently imported or treated as new state: explicit reset or independently trusted source-event replay is required. Plain SHA256 is an accidental-corruption check, NOT encryption, a keyed authenticity guarantee, anti-rollback protection, or device attestation.
