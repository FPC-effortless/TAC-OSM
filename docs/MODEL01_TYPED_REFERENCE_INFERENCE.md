# Model 01 — Frozen classical forecasting reference API (local only)

**Working software, not a production service or new TAC neural capability.**

Source: actual fitted UCI Appliances Energy (CC BY 4.0; DOI 10.24432/C5VC8G). The original Git-owned checkpoint under `research/replay-bundles/PRODUCT-MODEL01-UCI374-CLASSICAL/ORIGINAL_CHECKPOINT.json` is SHA256-pinned to `cb756a734b026602bb13f6af11b08711e2029f78aad1e95bb469d55131bc6c82`. No retraining or external model download occurs at inference.

## Exact, typed offline input

Supply a UTF-8 JSON object with **exactly** three keys: `schema` = `model01-energy-source-aligned-v1`, integer `source_aligned_step_index` at least 144, and `last_145_energy_wh` containing exactly 145 original 10-minute nonnegative finite measurements. The sample index is aligned to the **original UCI source-series origin**; it is NOT an absolute clock or automatically mapped to any arbitrary IoT source. This experimental interface has not established a real-world cross-site timezone/availability contract. Future targets and undeclared fields are strictly rejected.

Example constructed **non-real** request (generates a valid 145-point window):

```sh
python - <<'PY' > /tmp/model01_request.json
import json
print(json.dumps({'schema':'model01-energy-source-aligned-v1',
                  'source_aligned_step_index':500,
                  'last_145_energy_wh':[float(95+i%4) for i in range(145)]}))
PY
python specialist_models/model01_reference_api.py predict --request /tmp/model01_request.json
python specialist_models/model01_reference_api.py serve --port 8765
# A second terminal can POST application/json to http://127.0.0.1:8765/v1/predict
```

CLI returns a 60-minute-ahead numeric prediction, empirical CAL-derived 90% interval, original model/source SHA256 and explicit `NOT_QUALIFIED`, `LOCAL_REFERENCE_ONLY` and `ADVISORY_ONLY` labels. The interval can have a negative lower bound: the signed original calibration construction is preserved rather than silently changed. It is not independently proven to cover future observations or guarantee safe decisions.

HTTP server binds **127.0.0.1 exclusively**. It has no authentication, TLS, quota control, audit logging, multi-tenant isolation, live source authorization or deployed SLA; **never expose it to external network interfaces or treat it as the cloud product**. Strict JSON Content-Type, 4KB request limit, invalid-field/label rejection, no-store and other basic headers are not full production hardening.

## Verified measurements

- [Original reference endpoint CI #40470088700](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40470088700) PASS; **14 constructed interface and concurrency tests**, including original-model numerical parity, hidden-label/NaN rejection and exact output metadata.
- On the GitHub Actions Linux x86_64 Python 3.12.15 VM with reported four CPUs and constructed 145-point input: serial HTTP+JSON single-client 200 requests **p50 0.910499 ms, p95 1.018362 ms**; four concurrent clients 300 requests **p50 3.714642 ms, p95 5.432252 ms**. Socket connect/HTTP+JSON time included, warm up 20 requests; CPU/OS known from receipt. These values may vary significantly across runner machines; there is no cloud networking/TLS/auth or price data.
- [Original source FIT/CAL/DEV fit run #40467535435](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40467535435) trained *classical ridge*, not TAC. Original n=1,968 exposed DEV MAE 44.7923 Wh and empirical nominal 90% interval coverage 90.5996%; comparison to earlier persistence DEV is exploratory/post-exposure.
- The separate **real source API parity** workflow `model01-authentic-api-parity.yml` checks every exposed DEV observation against the original frozen training prediction graph and exact recorded metrics **without fitting or opening protected future labels**. Consult its original GitHub run conclusion; no future reference-equivalence result is assumed merely from the existence of its workflow.

## Path to a real developer service

1. Obtain actual customer job and source permissions, independent source-visible-at-time latency, legitimate sensor cadence/masks and trustworthy timestamps; define an input interface aligned to reality rather than publisher original row indexes.
2. Validate a new held-out source cohort and calibration under drift/missingness, including economically useful alarm decisions and group errors. Retain original low-performance and failure evidence.
3. Protect a production serving process with TLS, authn/authz, request-tenant isolation, quotas, network access policy, durable model provenance and operational observability. Benchmark real cloud p50/p95 with representative internet paths/concurrency, RAM, failure rates and actual cost per useful prediction; alternatively benchmark a real Android implementation.
4. Hold separate Veritas independent qualification and customer's decision authority. This reference implementation is a reproducible developer prototype, not a production release.
