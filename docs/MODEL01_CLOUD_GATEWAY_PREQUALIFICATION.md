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
