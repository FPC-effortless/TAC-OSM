# STREAM_BINDING_20261111 — caller-pinned model/state integrity

Internal untrained engineering, **not independently qualified**. No trained-model
capability, security certificate, checkpoint/source authenticity or promotion.

## Verified unfinished work and priorities

Reviewed master planaf307318, current AGENTS/CHARTER/CODEX/FPC Universal,
real-world/capability/anti-cheat/reward/phone-lab/execution-level rules, current
PRs724–727, Issue472 and PNDS118/117. Current user100k/matched-budget/offline-only
and two execution levels prevail. Ordinary internal work is not stalled on
external signers. Strict protected TEST/reward/safety/promotion gates remain.

Top3: bind model/config and carried-state identity before restore, preserve
numerical/replay caveats, then authentic matched learned-memory utility and
physical offline inference qualification. Six families stay aligned without
cross-family capability claims. This closes a concrete #727 limitation: state
from seed11 passes numeric ingress on same-geometry seed22. The new binding
rejects that accidental swap. No previous full-size copy screen is repeated.

## Smallest implementation

Capture a JSON-compatible receipt while the generating model owns the stream.
It fingerprints exact named learned parameters **and all named buffers**, model
class, block classes/memory precision mode, train/eval flag, full dataclass config,
state class/shape/dtype/logical bytes and caller stream/depth/token cutoff.

Freeze `binding_sha256(receipt)` outside the mutable restore payload. On reuse,
verification checks that independent expected pin, the trusted caller context,
the unchanged numeric ingress contract, and actual model/state fingerprints.
Correct compact/detached states and independently allocated identical weights
pass; model/config/mode/buffer/state/context changes fail before continuation.
The utility never calls the neural forward, edits state, installs precision,
detaches live state, loads pickle/checkpoint files, signs or accesses a network.

```python
# Capture at known owned generating execution, with all writers quiesced:
receipt = capture_stream_binding(model, state, depth=d, stream_ids=ids, observed_tokens=n)
trusted_pin = binding_sha256(receipt)  # freeze/store in separate trusted record
# Later, using the previously frozen pin (never regenerate it from restored data):
verify_stream_binding(model, restored_state, restored_receipt,
    expected_binding_sha256=trusted_pin, depth=d, stream_ids=ids, observed_tokens=n)
```

The pin is a **canonical JSON payload hash**, not a raw receipt-file/archive hash
or signature. Parameters/buffers use native logical tensor bytes with dtype/shape;
receipt byteorder is explicit. Storage strides/aliases and original backing
allocations are deliberately excluded so value-preserving compaction passes.
Hash cost is O(model+state bytes), CPU-only and intended for capture/restore,
not each decode token. No performance/energy advantage was benchmarked.

## Falsifier, repair and actual tests

Protocol/routerf554dc70 committed before execution, GitHub CPU5min/no fit.
Initial run **1failed/53passed**: prototype state_dict-only fingerprint missed
the non-persistent RoPE inv_freq buffer. Exact initial transcript is losslessly
base64-backed with SHA256 in INITIAL_FAILURE.json. The failing buffer mutation
was preserved, not removed/weakened. Repair hashes named_parameters and
named_buffers explicitly; state_dict is not complete runtime identity.

**55 new tests passed locally**, plus **106 inherited source/agnostic tests**.
Includes three seeds, FP32/FP64, exits2/4/prefix0/17/65, JSON roundtrip, cloned
model objects, original/detached/compact state, every emitted head/repair/candidate
next-token output, all participating parameter/incoming-state gradient checks,
RNG/state/weight identity, independent struct/hashlib byte references, parameters,
non-persistent buffers, config/precision/training mode, bounded state edits,
receipt/schema and caller-context negatives. These are constructed engineering
fixtures, not authentic-data measurements. Dedicated hosted receipt is recorded
separately; full repository suite and physical-device checks NOT RUN.

## Prior art, mathematics and strict limits

Reuse established [PyTorch serialization/state_dict semantics](https://docs.pytorch.org/docs/stable/notes/serialization.html)
and standard SHA256 content integrity (NIST FIPS180-4), not a novel learning or
cryptographic mechanism. No new neural operator/variant/Parameter is added.
PNDS118c8744f69 and the exact NumPy/Lean domain links are inherited in MATH_SCOPE:
conditional scalar read bounds do not establish hash collision resistance,
source/whole-model IEEE/AD or provenance/security qualification. No vacuous new
Lean identity theorem or independent signature is claimed.

**Required trust boundary:** caller must quiesce all writers during capture,
verification and use. This is not an atomic snapshot or TOCTOU defense. If an
actor may replace BOTH payload and trusted pin, or capture an already-mismatched
state under a fresh receipt, integrity cannot establish origin. Code/forward
hooks, arbitrary unregistered Python attributes, optimizer/RNG history, raw
observations and checkpoint-file bytes are not authenticated by these tensor
fingerprints. A passing report explicitly says source/code authentication
NOT_ESTABLISHED and independently_qualified=false. It cannot authorize protected
TEST, safety/reward certification, training, deployment or checkpoint promotion.

Prior cross-host initialization mismatch remains; this does not create an
identical original-checkpoint cross-host replay. Real checkpoint/data execution
is NOT RUN. No corpus/optimizer/GPU/TEST/merge/deploy/product work. New standard
95k–105k active/deployed,<=1%matched/identicaldata/train+tune unchanged; no125Mfit
exemption or500khistory rewrite. H64/483,11/84FP32 and COLLECT/FLOCK negatives
preserved. Source722/723 synthetic/no-outcome/no-call/template caveats remain.
Actual target/operators/numerical parity/physicalp50-p95/RAM/energy/airplane-mode
NOT RUN. Full inference/preprocessing/embeddings/state/speech remains offline on
user-controlled devices, no remote fallback. Review before import integration.
