# Model01 Android-compatible offline reference — first portable model export

**Status:** frozen real-source classical ridge translated to dependency-free Java; not a TAC-OSM neural model, Android APK, physical device test, independently qualified energy forecast, or production integration.

This code is **stacked on** [Model01 original checkpoint PR #481](https://github.com/FPC-effortless/TAC-OSM/pull/481). Its source-trained coefficients remain untouched. The company-wide six-family, device-only customer inference contract is proposed separately in [PR #538](https://github.com/FPC-effortless/TAC-OSM/pull/538). Neither proposal changes historical evidence.

## Input, state and output boundary

- **Inputs:** exactly 145 nonnegative finite energy observations in Wh, sampled at the original source's 10-minute cadence, with an explicit original-dataset-source-aligned integer step index in [144, 20,000,000]. **Not** raw smartphone-clock timestamps or an authenticated new household's sensor identity. Reject missing readings and wrong/gapped indices; never fill them silently.
- **Local state:** only `WindowState`'s bounded 145-reading observation buffer and source-aligned index. This is an ordinary application buffer, **not** TAC recurrent neural memory. Callers must persist `snapshot()` and `sourceAlignedIndex()` in private device storage for restart survival; persistence is not yet implemented or tested inside an Android app.
- **Outputs:** 60-minute-ahead point forecast (Wh), original historical CAL-derived uncertainty interval (not guaranteed), frozen checkpoint/source IDs, `NOT_QUALIFIED`, `ADVISORY_ONLY`. A prediction cannot authorize a device or business action.
- **Offline boundary:** all feature generation, numerical operations, buffer updates and typed outputs run in ordinary Java; no socket/network/remote inference, embedding, state, speech or fallback code exists. Fail invalid observations locally with an exception. The **full Android application**'s no-network behavior remains NOT RUN until tested on a physical device.

## Reproduce export and host JVM parity

From the checkout containing original Model01 PR #481 files:

```sh
python scripts/export_model01_android_java.py --check
python -m unittest discover -s tests -p 'test_model01_android_export.py' -v
```

The exporter verifies the original JSON checkpoint's exact SHA-256 and frozen scientific provenance, then deterministically renders `mobile/android/model01/Model01OfflinePredictor.java`. It never retrains, downloads or reads future labels. The host tests compile with `javac`, compare 48 constructed inputs and intervals against `specialist_models/model01_reference_api.py`, test invalid inputs/abstention, and check observation-window copy/sequence invariants. CI uses Temurin JDK 17. Tests are software/operator numerical agreement on a **host JVM**, not Android science.

Import `Model01OfflinePredictor.java` as Java source into a real Android application's local module; no ONNX, TensorFlow, PyTorch, Android API or cloud runtime is required for this fixed classical model. Example:

```java
double[] historicalWh = /* 145 previously observed source-aligned readings */;
try {
    Model01OfflinePredictor.WindowState state =
        new Model01OfflinePredictor.WindowState(500, historicalWh);
    Model01OfflinePredictor.Prediction forecast = state.predict();
    // Render advisory forecast and qualification warning locally.
    // On the next genuine contiguous observation, call state.append(501, observedWh).
} catch (IllegalArgumentException invalidOrGap) {
    // Abstain locally. No network fallback or fabricated reading.
}
```

Do **not** map index 500 to a new household without a verified phase/cadence origin and new-source evaluation. The original energy result is a development demonstration from one household with previously exposed DEV. It is not a verified new-site skill or real customer value.

## Physical-device qualification matrix — all outstanding

| Platform | Target hardware | Test status |
| --- | --- | --- |
| Android first | Samsung Galaxy A15 4G (exact SoC, OS, memory to record) | **NOT RUN** |
| iOS future | iPhone 13; requires separately implemented Swift/local library | **NOT RUN** |
| Embedded future | Raspberry Pi 5 8 GB; Java runtime/version to be pinned | **NOT RUN** |

Android physical-app gate: verify CPU operator availability and app packaging; compare actual device outputs to frozen Python cases using preregistered relative/absolute tolerances; measure warm/cold p50/p95 latency, RAM peak, energy (method/battery impact), process restart and local-state continuity, permission/network traces, and actual airplane-mode operation without cloud, networking or hidden fallback. Proposed initial targets, not results: p95 <150ms and incremental RAM <128MiB. Record model, device, OS, battery/power state, run count, raw distributions and failure/abstention cases.

**TAC-OSM neural export:** separate outstanding task. This Java model has only seven trained coefficients and is not a way to bypass the 95k–105k active trainable/deployed parameter rule for future TAC neural training. First audit an actual frozen 95k–105k native checkpoint, recurrent-state and custom operator graph, then test ExecuTorch/ONNX/alternative local CPU exporter with original weights and state carry. No neural graph, operator-support or physical-hardware qualification claimed here.

## Evidence status

| Claim | Status |
| --- | --- |
| Original ridge checkpoint available on PR #481 | PRESENT (hash-pinned; original scientific tier exposed DEV) |
| Java generated source and deterministic exporter | IMPLEMENTED; host checks separately recorded |
| Reference numerical agreement | HOST JVM unit check only; see actual CI conclusion |
| Full Android APK running offline | **NOT RUN** |
| Physical p50/p95, RAM, energy, airplane mode | **NOT RUN** |
| Independent new-user/sensor predictive gain | **NOT QUALIFIED** |
| Original TAC-OSM 100k recurrent neural deployment | **NOT RUN** |
