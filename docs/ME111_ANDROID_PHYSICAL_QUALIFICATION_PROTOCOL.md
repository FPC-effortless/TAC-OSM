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
