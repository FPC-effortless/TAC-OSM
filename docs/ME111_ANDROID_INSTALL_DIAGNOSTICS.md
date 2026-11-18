# TAC-OSM Android APK installer diagnostics — exact evidence and privacy boundary

**Status 2026-10-29:** User-provided real Samsung Galaxy phone screenshot
shows `There was a problem while parsing the package` while attempting
to install the previously released 80.45-MB original debug APK. This is an
**observed PHYSICAL INSTALL FAIL**, cause **UNKNOWN**. Emulator passing
is not real-phone validation; neither emulator numerical parity nor this
work establishes scientific usefulness or hardware performance.

## Pinned original APKs

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Original multi-ABI APK | 80,446,353 | `ff751d25dd53af3f1156046391398ff97ec6c680dfa967c847fa371558ba8505` |
| ARM64-only APK | 20,195,455 | `f17a9f9a5752c32ffe8da407b27754793925a87bddb7b51a8bae56d64fb1f318` |
| Pure-Java independent installer probe | 10,041 | `86a6f70db9ccbe5dae6489f7d21608b41620735a9a22595ef1c0ff4e1996ae03` |

The model APKs share **exact** original model graph SHA-256
`dd65519a9e4f3388966538af53119b56e9292b46a9209d63ad4927b87f0a176b`.
The original UCI historical source FIT sample is unchanged. Both original
and ARM64 APKs passed APK signature scheme v2, zipalign and APK ZIP CRC
checks. They require min Android 8 (API26), target Android 15 (API35).
The native ARM64 `libonnxruntime.so` ELF has 16-KiB LOAD segment
alignment, and both ARM64 shared libraries are APK-page-aligned to 16 KiB.
This does not establish compatibility with every OEM's policy or hardware.

The **original full APK** installed and launched on a generic Android 16
API36 x86_64 emulator in GitHub run [#40802404972](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40802404972).
This rules out *universal* Android 16 incompatibility. The **small ARM64-only**
APK was separately built from identical source with identical ARM64 runtime
bytes and model/sample assets, but had no independent physical Samsung
installation receipt. Its v2 signature, APK integrity and hash are verified.
No fresh signing key continuity between dynamically generated GitHub debug
builds is promised; updating a previously installed differently signed
package may require the user's **explicit** uninstall, which can erase
private recurrent-state history. Never silently uninstall or clear state.

## Zero-dependency Java-only phone installation probe

The optional module `mobile/android/model01/sample-app/installprobe`
builds a ~10 KB separate debug APK:
`org.tacosm.mobile.installprobe`. It contains no ONNX, native code,
network permission, network client, files/storage permission, model
weights, raw sensor rows, login or server transport. It only displays
the phone's reported Android version, ABIs, 64-bit process bit, memory
page size, and whether the separately named neural package
`org.tacosm.mobile.model01.demo` is already installed; if accessible,
it displays that installed package's signing-certificate SHA256 for
possible version-signature conflict diagnosis. User-controlled explicit
copy is the only outbound sharing action. It does not automatically
touch clipboard, send data, install another package, or change settings.

The probe has its own dedicated Android 16 emulator installation/smoke
workflow `me111-android-install-probe.yml`. **Physical Samsung test remains
NOT RUN unless the user reports an observed result.**

Interpretation:

- **Probe also fails to install:** damaged local download, Android unknown-
  source permission, Samsung Auto Blocker, file-manager policy, or system
  installer state become higher priority than neural runtime. Verify the
  downloaded APK SHA256 rather than renaming archives.
- **Probe installs, model APK fails:** investigate architecture/variant,
  package name already installed with different signer, APK-specific
  OEM install policy, storage and 16 KB native compatibility; obtain a
  real installer error, not the ambiguous popup, before selecting a fix.
- **Probe installs, model APK installs and launches:** physical installation
  success is newly established for those exact bytes/device, but neural
  output, latency, memory, energy, radio trace, user-state restart and
  forecast utility remain separate gates.

## User-only phone checks (do not bypass platform security blindly)

1. Confirm the downloaded file is the actual standalone `.apk`, not the
   GitHub Actions `.zip` artifact or a renamed split archive.
2. If available, verify file bytes on the **phone** with the separate
   `TAC-OSM-APK-Offline-Checker.html` user-visible file. It accepts a local
   selected APK, hashes it with browser WebCrypto, and compares against
   the exact expected sizes/hashes above. There are no network calls.
3. With informed user consent, check Samsung **Settings > Security and
   privacy > Install unknown apps**, selecting the exact app used to open
   the APK (My Files, Chrome or ChatGPT). Samsung's **Auto Blocker**
   may block non-store apps. If enabled, treat it as a meaningful
   security control, not something to permanently disable without
   a trusted source and user decision.
4. Try only a known-hash original APK. If the 10 KB probe installs,
   open it and **optionally** copy the diagnostic result into the research
   conversation. Do not post unrelated device identifiers or account data.
5. If the original or ARM64 app was already installed with a different
   debug certificate, do **not** uninstall it without consent: private
   saved TAC recurrent state can be lost. Review signing fingerprint first.

The screenshot's generic phrase **does not distinguish** damaged file,
unsupported ABIs, Android install-source protection, signing conflict or
OEM policies. Refuse categorical root-cause claims without evidence.

## Scientific and release gates

The neural model remains original 104,663-param ME111, no retraining.
Original real sensor calibration joint gate **FAIL / MIXED_TRADEOFF**.
Training-period raw CSV replay and emulator parity are **engineering**
evidence, not accuracy/health/production readiness. Physical installation
and p50/p95/PSS/energy/thermal/network-tracing results must use actual,
consenting hardware and exact version, not Android emulators.

All PRs remain draft unless the owner authorizes merging and deployment.
