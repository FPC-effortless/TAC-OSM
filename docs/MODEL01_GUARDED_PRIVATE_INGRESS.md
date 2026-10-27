# Model 01 — Guarded private ingress for the frozen energy forecaster

**Status: local authenticated reference implementation, NOT deployed production cloud, NOT an independent security qualification, NOT model-product release.**

Implements `specialist_models/model01_guarded_ingress.py`, retaining the original exact Git SHA256-pinned classical UCI374 ridge checkpoint and its DEV-only interval. This changes only serving admission; no model fit or science qualification.

## Run locally

Provide only a long random API secret through an independent secrets manager; the process reads its **SHA256 digest** from `MODEL01_KEY_SHA256` and never accepts a token through command arguments or repository files. An example to create a *new test-only* key and digest outside repository:

```sh
python - <<'PY'
import hashlib,secrets
key=secrets.token_urlsafe(36)
print('Set an externally protected MODEL01_KEY_SHA256 to:',hashlib.sha256(key.encode()).hexdigest())
print('Keep the matching bearer token in a secret manager, never in Git:',key)
PY
MODEL01_KEY_SHA256='<sha256 of your external token>' python specialist_models/model01_guarded_ingress.py --port 8766 --requests-per-minute 600 --max-active 4
# Only 127.0.0.1:8766 can accept the authenticated POST /v1/predict.
```

The process fails on missing, empty, malformed or default hash config. It requires a single `Authorization: Bearer <key>` header for health and inference, SHA256s the presented token before `hmac.compare_digest`, rejects invalid or duplicate JSON fields, hidden targets, invalid inputs and ambiguous HTTP framing. The request body max is 4096 bytes and processing admission uses a monotonic global rolling-minute quota plus bounded simultaneous model inferences. Unauthenticated users cannot obtain a model health fingerprint. Responses remain labeled `LOCAL_REFERENCE_ONLY`, `ADVISORY_ONLY` and `NOT_QUALIFIED`.

## CI

[Guarded private ingress tests #40475101732](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40475101732) — **13 constructed-source tests PASS** including wrong/missing tokens, hidden labels, NaNs/duplicate fields, oversized requests, rate-limited path, authenticated health, strict loopback binding and rate-window expiration. These are software regression tests, not a penetration test. Previous original-source [1,968-case no-refit API parity #40470974719](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40470974719) still checks that underlying prediction values have not drifted.

Earlier **unguarded reference** local-loopback benchmark [#40470088700](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40470088700) reported p50/p95 0.91/1.02 ms for one local client and 3.71/5.43 ms for four clients. **These numbers are NOT security-wrapper measurements**, cloud hosting latency or deployed TLS/auth performance. Do not reuse them as guarded-service SLA results. The guarded service still requires representative authenticated latency measurement.

## Release/security blockers

- This code binds 127.0.0.1 *only*, uses plaintext HTTP on private loopback, and has no trusted TLS ingress deployment. Do not proxy onto public networks without dedicated threat-model review, mutual network/ingress authentication, strict firewall/ingress rules, TLS termination and request-header hygiene. In particular, `X-Forwarded-Proto` is not evidence of a trusted TLS session.
- A single shared bearer secret and in-process rolling rate limiter do **not** provide tenancy, persistent distributed quota, storage, key rotation/revocation history, model rights verification, CSRF protections for browser sessions, socket/thread-count hard limits, monitoring or abuse/DoS resistance. Authentication can improve basic private prototype hygiene but is not standalone production authorization.
- Environment configuration and measured latency never justify claiming buyer usefulness, independent source qualification, cross-site distribution, interval coverage guarantee or a TAC-OSM neural model result.
- Required: externally managed secrets, key rotation, per-tenant claims/ACL, hardened ingress, rigorous stress/adversarial tests, actual cloud cost and p50/p95 at representative deployment load, operational observability, legal data license and independent evaluation of unchanged original model on a previously untouched source population.

See [PR #481](https://github.com/FPC-effortless/TAC-OSM/pull/481), [original inference reference](MODEL01_TYPED_REFERENCE_INFERENCE.md), and [product issue #450](https://github.com/FPC-effortless/TAC-OSM/issues/450).
