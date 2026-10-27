# Model 01 — Genuine meter input: source admission before predictive deployment

**Product status: new strict input and safe advisory-fallback adapter.** A real customer target/source protocol is still not qualified, no original TAC neural model is trained, and neither the historical classical ridge nor its original 90% interval has shown reliable independent cross-site transfer.

## Input contract

The exact machine-readable `model01-meter-asof-ingestion-v1` payload contains an opaque ASCII meter ID, target `appliances_energy_wh_per_10min`, sampling=10 minutes, timezone `UTC`, explicit decision-time timestamp and exactly 145 consecutive completed 10-minute energy accumulations. Each sample provides `window_end_utc`, `available_at_utc`, and nonnegative finite `energy_wh`. Times use literal ISO 8601 UTC second precision, e.g. `2026-10-27T09:00:00Z`.

Every completed source window must have become available at or after its end but no later than the decision time. Discontinuous, duplicated, incomplete, future or prematurely available measurements are rejected, not interpolated. A latest sample more than 20 minutes before decision time yields DEFER. Unknown target/whole-house source yields DEFER; arbitrary textual JSON, NaN, Infinity, hidden future labels and unexpected fields are rejected. This is a strict development protocol, not proof that a real customer source actually publishes data that quickly.

**Important target mismatch:** Model01 original UCI374 learned household **appliances only** energy Wh; UCI235 cross-site whole-house energy transfer scored MAE 138.49 Wh vs local persistence 112.75 Wh and original 90%-nominal interval covered only 51.1%. The experiment [#40478750512](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40478750512) is new real-source stress but NOT same-target independent confirmation. Input adapters MUST NOT silently identify total power with appliance energy.

## Safe behavior

If current real-time target/source conditions pass and `allow_advisory_persistence` is explicitly true, the adapter returns only the last actual observed 10-minute appliance energy as an **unqualified reference baseline**; no original ridge inference is permitted, no probability interval, no alert authority, no device-control command and no production promise. If the caller does not request baseline advice, returns DEFER. A production site-model-specific pipeline requires new independent same-target data and source visibility authorization; the old ridge cannot be reclassified as deployable by altering a field.

CLI: `python specialist_models/model01_meter_admission.py --request /path/to/meter_request.json`. No passwords, operator personal data or physical device state should be committed or logged. This does not expose any network service. Use behind a separately authorized, properly secured customer connector if deployment is later approved.

## Next meaningful customer capability

Choose an operator question: predicting 60-minute-ahead appliance energy (Wh), or whether a real consumption anomaly warrants a manual check. Obtain a legitimately consented building deployment with independent actual reading-available-at timestamps, 145 chronological 10-minute appliance energy readings at each decision, confirmed future metered outcomes, and an appropriate withheld site/period. Compare to strong no-fit persistence and site-adapted classical references while reporting error, interval miscoverage, outages, missing data and actionable cost; preserve original trained artifacts. Only then promote a specific model behind authenticated cloud/device APIs.

Failure/success cases in `tests/test_model01_meter_admission.py` are **constructed software fixtures**, not genuine real-world value or model skill. No original UCI374 protected TEST was accessed; the UCI235 stress results remain exposed and cannot be reused for independent confirmation.
