# Model 01 — Authenticated loopback gateway, cloud-serving candidate

**Implementation status:** Original source-frozen classical UCI374 ridge forecast, exposed historical development results only. No publicly reachable cloud endpoint, customer access, TAC neural fit, independent model confirmation or security certification.

This gateway wraps the original replayed typed predictor with a deny-by-default bearer-token gate, constant-time credential digest comparison, process-local token bucket and concurrent request limit. It always binds to 127.0.0.1, and cannot be configured to listen on public network interfaces.

## Run privately

Create a random key through a secrets manager and pass it only in the process environment. Never put real credentials in Git, GitHub Actions logs, URLs or shared artifacts.

```sh
export MODEL01_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
python specialist_models/model01_authenticated_gateway.py --port 8766
# POST a JSON request to http://127.0.0.1:8766/v1/predict with Authorization: Bearer <secret>
python -m unittest discover -s tests -p test_model01_authenticated_gateway.py -v
```

Payload schema: schema=model01-energy-source-aligned-v1, integer source_aligned_step_index, and exactly 145 historical source-aligned 10-minute energy readings. Unrecognized fields, hidden labels, future data and nonfinite values are denied. All outputs remain advisory and historically calibrated only.

The server requires a >=32-character ASCII key from the environment; no default or test credential is hard-coded in runtime code. Even /health requires authentication. 401 responses never echo token bytes. Transfer-Encoding and duplicate Content-Length headers are rejected. Underlying inference request bodies are limited to 4KB. Admission and concurrency limits are **per process**, not distributed safeguards.

## Mandatory future security and customer deployment gates

- TLS termination, independently validated network perimeter and proxy trust, endpoint/tenant identity, source permissions and tenant-scoped authorization.
- Centralized rate limits, usage accounting, secret rotation, API versioning, request provenance, incident response, negative/replay security testing and rollback.
- Real cloud hosting measurements (end-to-end p50/p95/p99, concurrency, retries, uptime, request failures, source freshness, memory/CPU consumption and cost per useful prediction).
- New independently source-qualified real customer/site data, fresh evaluation and honest coverage/error evidence. Historical exposed one-household development accuracy does not establish operational model quality.

**Not production-ready:** Python development reference has no built-in TLS, credential lifecycle, hardened tenant isolation, distributed quota, service monitoring or externally audited security. Never directly expose it to the internet.

Predecessor [localhost loopback measurement #40470088700](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40470088700) had 0.91ms p50 / 1.02ms p95 serial; these are **not** yet authenticated gateway or production cloud latency measurements.

Implementation is stacked on [draft Model 1 #481](https://github.com/FPC-effortless/TAC-OSM/pull/481); original Git checkpoint and research failure/evaluation gates remain unchanged.

## Actual authenticated loopback benchmark (October 27, 2026)

[Source-pinned authenticated gateway CI #40476041423](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40476041423) passed **13 constructed security/auth tests** and ran a new end-to-end localhost HTTP+JSON measurement on the Linux x86_64, four reported CPUs GitHub Actions runner (Python 3.12.15), with a fresh ephemeral bearer token that was not printed or preserved.

| Workload | Requests | p50 latency | p95 latency | p99 latency |
| --- | ---: | ---: | ---: | ---: |
| Authenticated, single client | 200 | 0.839 ms | 1.038 ms | 1.073 ms |
| Authenticated, four concurrent clients | 300 | 4.241 ms | 6.323 ms | 7.117 ms |

These are **measured local reference prototype results** and can change between GitHub virtual machines. They do not measure production network routing, TLS/auth proxy, real customer traffic, availability, server provisioning, egress costs, economic value or new-model predictive quality. The previous unauthenticated loopback measurements remain separately attributed to their older run.


## Original publisher timestamped input integration (October 27, 2026)

`specialist_models/model01_source_timestamp_adapter.py` accepts exactly 145 consecutive original UCI374 source ten-minute observations with naive-local timestamps and numeric historical energy values. It constructs the required source-aligned step index using **2016-01-11 17:00:00** as the original publisher-series origin, and rejects gaps, duplicate/out-of-order timestamps, timezone conversions, unexpected source identities, hidden outcome fields, negative/nonfinite energy and malformed records. The existing original reference API and authenticated localhost gateway still consume the same typed payload; **the model, calibration, weights and training results are unchanged**.

Example command: `python specialist_models/model01_source_timestamp_adapter.py --window ORIGINAL_SOURCE_145_OBSERVATIONS.json`. Add `--predict-offline` to perform the original pinned-model inference locally. Schema is `model01-original-source-timestamped-window-v1`, `source_series_id` is `UCI374_ENERGYDATA_COMPLETE_ORIGINAL_PUBLISHER_CLOCK`, and `observations` contains 145 objects each with exactly `observed_at` (`YYYY-MM-DD HH:MM:SS`) and `energy_wh`. Historical no-zone timestamps should **not** be mistaken for authenticated universal UTC time or prospectively available source readings.

[Genuine original publisher DEV timestamp-parity CI #40479028016](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40479028016) **PASS** on all **1,968 already-exposed real DEV source windows**: the adapter's decoded index and forecast match the original checkpoint's training-graph prediction exactly, observed max absolute forecast difference 0.0; no refit, no protected TEST target parsed and no independent customer data. [Timestamp schema+gateway CI #40478458648](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40478458648) **PASS**, 11 time-grid and 13 pre-existing auth tests. The original-source parity receipt is Git-owned at `research/source-evidence/MODEL01-UCI374-REAL-SOURCE-TIMESTAMP-DEV-PARITY.json`.

**Cross-source onboarding remains unimplemented** until an actual customer/second-source timestamp contract, physical sample availability times, DST policy, learned model calibration and independent quality evidence exist. Changing `source_series_id` to an arbitrary building is deliberately rejected; there is no blanket out-of-domain model safety claim.