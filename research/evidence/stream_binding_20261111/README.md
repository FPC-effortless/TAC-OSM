# Original stream/model binding evidence

[TAC source0d2378e2](https://github.com/FPC-effortless/TAC-OSM/commit/0d2378e2de94d20ad56ae5a8bc214ee85f1983c0) /
[CPU42976770585](https://github.com/FPC-effortless/TAC-OSM/actions/runs/42976770585) /
[artifact13836280503](https://github.com/FPC-effortless/TAC-OSM/actions/runs/42976770585/artifacts/13836280503):
**161tests PASS** (55new,106inherited). [Compulsory42976770579](https://github.com/FPC-effortless/TAC-OSM/actions/runs/42976770579) also PASS.

Lossless HOSTED.zip SHA256
`9c31f5808405c73599dace9eb02800e30aeb945fdaa2d374e23bdb07eb2c5409`.
It contains JUnit, exact current source hashes, frozen preregistration and router
receipt. verify_saved.py verifies hashes/coverage/negative-log retention with
stdlib only, no model/trace/weights/corpus execution.

Initial prototype failure remains in ../..//stream_binding_20261111/ as an exact
base64 transcript and SHA record:1failed/53passed, omitted non-persistent RoPE
buffer. The same buffer mutation test now passes by rejecting that changed model;
it was not removed. Generic research_probe._module_snap already enumerates
parameters+buffers; research_anticheat.fingerprint is a diagnostic JSON-value
fingerprint. Neither provides this pinned restore/model-config/mode/context API.
The new byte-oriented representation avoids converting all model values to
Python lists or cloning every tensor into a model snapshot; no memory/speed
advantage is inferred from that design choice without measurement.

Inherited PNDS118c8744f69 scalar/NumPy/Lean scope remains unchanged. No new neural
operator/cryptographic theorem, fresh Lean build or whole-model/source/IEEE/AD
qualification. Current default AGENTS/CHARTER/CODEX were also checked (see
INSTRUCTION_REVIEW.json); the user's explicit standard100k constraints remain
for this task and no new fit occurs.

These are internal untrained fixtures, not independently qualified or learned
intelligence/security/safety/reward evidence. Trustworthy initial capture and an
external frozen pin are necessary; replacing both or stamping an already-mismatched
state cannot establish origin. Executable code/hooks, raw observations, original
checkpoint-file bytes, optimizer/RNG history and concurrency are outside the hash
contract. No actual checkpoint/data/TEST/optimizer/GPU/promotion/merge/deploy.
Same-checkpoint cross-host replay and physical target/operators/numerical parity/
p50-p95/RAM/energy/airplane-mode NOT RUN. Old failures and strict gates preserved.
