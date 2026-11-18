# ME111 original FIT-source bridge for offline Android (draft, not a product)

**Purpose:** Establish real-source input preprocessing equivalence with the
unchanged, already-trained 104,663-parameter ME111 TAC-OSM graph.
This is **not** retraining, a new validated model skill, an automatic sensor
feed, independent sensor data, or a valid public-health predictor.

## Source and scope

Publisher: Chen, S. (2017), *Beijing Multi-Site Air Quality*, UCI Machine
Learning Repository, DOI [10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Archived publisher ZIP SHA256:
`d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`.
Original parsing and transformation `scripts/me100_sensor_data.py`, pinned
by commit `3fc2fbe8e4974491349caecd8d6171e738810b54`, loader SHA256
`57f4d9086251d6dd8f12feb675ebe96de1223b3016906ad867cf3795fd0c74cc`.

Strictly only FIT cohort `2013-03-01 00:00` through 2014 is used for
normalization. In particular **the newly added tests use original station 0
and station 5, first 128 hourly FIT events each**. No later 2015 REPORT or
2016–2017 protected numerical row is read or used to choose parameters.
The 256-event model replay uses only the **first 256 FIT events of station 0**,
compared to the original checkpoint's separate PyTorch execution. That
comparison is an *inference-portability check*, not forecast scoring.

## Exact original input transform

Original source numeric channels in order:
`PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM`.
The 12th/13th channels are the sine/cosine of the original publisher wind
direction tag (16 categorical compass bins). A missing wind tag masks BOTH.
The first six pollutant columns use `log1p(raw)`; the other numeric
columns remain in source units. All 13 are centered/scaled with the original
FIT-only global means/stds, clipped to `[-8,8]`, then stored as FP32. Missing
values are masked, assigned **zero** at the neural input, and never forward
filled. For every channel, the age is
`log1p(min(source_hour - last_true_observation_hour, 256)) / log(257)`.
The six clock features are
`sin(2π*[hour/24, weekday/7, (day_of_year-1)/366])` followed by the same
cosines. Source integer hour starts at **0** on 2013-03-01 at 00:00; it is
not the phone's wall clock nor an inferred timezone.

The Android Java encoder must receive the complete chronological archive
from the original hour zero to calculate observation ages. A missing,
out-of-order, repeated, cross-station or fabricated boundary is rejected.
The Java input contract was repaired to accept original `hour=0`, with
`initial lastHour=-1`, while still rejecting negative source hours.
The FIT moments are versioned in `Me111FrozenFitStats`; they are not
recomputed on the phone or calibrated with any user-selected later rows.

## Two independent engineering checks

1. Host JVM vs **unaltered original Python source**: 256 publisher
   FIT records from two stations, with raw numeric strings and published
   wind tags, are processed independently in Java and the pinned original
   `me100_sensor_data.py`. Compare all 13 normalized values, masks and ages,
   all 6 calendar features, original source owner and hour. Registered
   max element error `2e-5`; the test also rejects cross-entity, unknown
   source wind, negative pollutants and non-contiguous/mid-stream history.
   Evidence: [original FIT source parity run #40789120190](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40789120190),
   256 records PASS, max error `1.1920928955078125e-6`.

2. Actual Android **original FIT CSV -> Java feature encoder -> ONNX
   Runtime CPU**: in a fresh, scratch-only state, replay 256 hourly FIT
   inputs, then compare the six predictive locations, left/right scales and
   all returned recurrent state tensors with the *unchanged original trained
   PyTorch checkpoint* on the same exact published data. Fixed tolerance
   `3e-4 + 2e-4*abs(reference)`, with equal integer count/hour. The
   original source and model fixture is generated from the pinned Git
   archive/checkpoint in CI and placed in the **test APK only**. This
   check must remain `PENDING/FAIL` until actual CI evidence passes.

## Human-initiated Android offline research replay

The app's neural screen has a separate **Replay 256 original PRSA FIT CSV
rows offline** control. The user must first enable Airplane Mode and manually
disable Wi-Fi and Bluetooth, then select a **locally saved** original-style
station CSV in the Android document picker. The Android app itself has no
INTERNET permission and never calls a hosted inference or data service.
The document provider could in principle be cloud-backed: the airplane-mode
setting alone is **not an independent network trace or radio-isolation proof**.

The parser requires exact publisher column names, a single known station,
source hours starting at zero, contiguous chronology, positive original
pollutants or `NA`, and the first 256 rows in the 2013 FIT prefix. It
refuses malformed/partial data. The imported **user-provided CSV cannot be
cryptographically authenticated as the publisher's original source** by
syntax checks; the UI reports this explicitly. All neural computation and
source state stay in independent scratch memory; existing private recurrent
state is never read, committed, deleted or used as a fallback. Model output
remains **standardized log1p units** and should not be read as raw
concentration or a health/safety instruction.

## Gates that remain open

- No live authorized regional sensor source or device sensor ingestion.
- No usable authentic off-FIT future forecast scoring, calibration or
  independent site confirmation. Original ME111 real-world gate still
  **FAIL/MIXED**.
- No original or new weights changed, no further optimizer step, no sealed
  TEST or 2015 REPORT selection.
- Physical Android latency/p50/p95, energy, thermal stability, genuine
  network-disconnected execution, independently logged process recovery
  and customer acceptance remain **NOT QUALIFIED**.
- This bridge does not establish useful 256-step retrieval, causal
  forecasting, health risks, enterprise ROI or an investment claim.

All improvements must preserve these distinctions in generated receipts and
customer-visible text. There is no release, merger or commercial deployment
authorized by the software-only checks.

## One-tap bundled original FIT sample — implemented, software gate pending

To remove the need for a computer or dataset download, the debug Android APK
now embeds ONLY the first 256 original publisher FIT rows of the Aotizhongxin
station, obtained at build time from the original SHA256-verified UCI archive.
The CI build copies the exact source bytes from its independently generated
original-PyTorch test fixture. The published source is CC BY 4.0, attributed
to Chen (2017), UCI Beijing Multi-Site Air Quality (10.24432/C5RK5G).

The Android neural screen now offers **Run bundled original FIT sample offline**.
A single user tap runs the exact 104,663-param ME111 ONNX CPU neural model
through all 256 historical observations without a document picker,
external permissions, internet connection, source configuration, or any
changes to app-private persisted recurrent memory. The app displays its
unqualified standardized forecast and clearly identifies the historic
training-period sample and failed original scientific gate. It does NOT
represent current local air quality, an independent accuracy benchmark,
or evidence of real-world useful long memory.

An Android instrumentation test checks that the main APK's bundled CSV is
byte-identical to the UCI source-pinned test APK's original FIT CSV and
that native neural output/state match the untouched original PyTorch
checkpoint; any difference is a software gate FAIL. Do not claim it passed
until the exact code commit's CI workflow succeeds. The separate custom
user-selected CSV importer remains appropriately labeled as unauthenticated.
