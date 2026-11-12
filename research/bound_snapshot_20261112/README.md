# BOUND_SNAPSHOT_20261112 — persisted inference state, untrained fixtures

**Internal engineering, not independently qualified.** These are newly captured
random-weight fixtures, never historical trained checkpoints or real-data scores.

## Verified unfinished step / priorities

Master planaf307318, current branch/default AGENTS/CHARTER/CODEX/FPC Universal,
real-world/capability/anti-cheat/reward/phone-lab/two-level rules, current drafts
725–728, Issue472 and PNDS118/117 were inspected. User100k/matched-budget/offline
instructions prevail over default scale-v2 wording. No new fit occurs.

Top3: persist/restore owned model+stream state, retain honest replay/numerical
contracts, then authentic learned-memory utility and physical offline runtime
qualification. #728 binds only in-memory state; this implements its missing
persistence boundary. Prior memory-copy screens/H64/11of84/cross-host initialization
mismatch are not repeated or erased. Six families align without transfer claims.

## Implementation and established prior art

Use [Safetensors0.6.2](https://huggingface.co/docs/safetensors/en/index), not a new
raw tensor format or pickle loader. Existing CASM/benchmark torch.save paths and
all original model/precision/ingress/binding sources stay unchanged. No dynamic
class imports or metadata-selected model factory. Caller supplies a trusted
already-built CPU prototype and independent stream/depth/token cutoff.

`save_bound_snapshot` writes a NEW directory with only fixed manifest.json and
tensors.safetensors. All named parameters, **all buffers including non-persistent
RoPE**, and logical carried-state tensors are serialized; SHA256 binds payload
and binding receipt. Return value is SHA256 of exact RAW manifest bytes. It is
not #728's canonical payload hash or an external signature. Freeze the raw pin
outside the mutable snapshot before later restore. Overwrites refused.

`restore_bound_snapshot` checks pin, fixed schema/context/payload hash/budgets,
key/shape/dtype layout, then loads only into a **new deepcopy** of the prototype.
State classes come from source initial-state templates, never input strings;
cache positions/IDs come from trusted caller context. Reuse #726 compaction to
avoid retaining decoder backing storage, then run unchanged #727/#728 validation.
The caller's model/state are unchanged on success and failure. Wrong precision,
config/training mode, unknown keys, corrupt payload, malformed JSON, symlinks,
missing data and budgets fail closed. CPU only; first format limits64MiB payload/
1MiB manifest, not an authorization to train/deploy at any parameter count.

This is **inference-only**. Optimizer, RNG history, autograd graph and training
continuation are NOT saved/reconstructed. Original resumable checkpoints remain
immutable in their own formats. Export uses detached *copies*; live state is not
detached. Restored tensors carry no saved AD history. Copy/encoding/decoding costs
and total/device/peak RAM are not benchmarked or claimed efficient here.

## Preregistration / actual local evidence

Protocol/router64af0c74 committed before tests/evaluation; GitHub CPU10min/GPU0.
58new tests PASS, plus161inherited tests PASS. Three seeds, FP32/FP64, exits2/4,
prefix0/17/65; exact original snapshot binding and all next-head/repair/candidate
outputs, fresh initialization + deliberately perturbed RoPE buffer restored,
prototype unchanged, restored state backing equals logical size. Negative tests
cover byte/pin/context/schema/key/shape/dtype/mass/symlink/budget and overwrite.

Separate local processes captured and replayed two fixed untrained fixtures
(fp32/fp64, seed11, exit4, prefix17) into seed101 prototypes. Both passed exact
model/state binding and exact next outputs/state (max abs error0 in the recorded
local run). LOCAL_REPLAY.json preserves that run. Later source checks added
explicit finite outputs/fixture-file hash coverage without changing model,
inputs, tolerances or original receipt. Dedicated hosted capture/cold-replay
are separate receipts, not invented from local execution.

Cross-run prediction tolerances were frozen **atol1e-5/rtol1e-5**; integer IDs must
match exactly. Byte-exact output flags and all per-tensor max errors are reported
separately. Producer index hash passes outside the artifact via job output;
reader predicts BEFORE opening expected outputs, never using them as inputs.
These are ordinary controlled CI jobs, not access-separated scientific assessors.

## Trust, mathematics, limits / next

Trusted capture/prototype/source code/pin custody and quiesced model/filesystem
remain caller obligations. Replacing both pin and content or capturing already-
wrong state cannot establish origin. No source authenticity, protected TEST,
safety/reward certificate, independent qualification or promotion follows.
Read/hash and decoder use the same bounded bytes, but no hostile filesystem/
concurrency or package-security certificate is claimed. Raw observations and
checkpoint provenance cannot be inferred from numerical validity or hashes.

MATH_SCOPE.json preserves PNDS118c8744f69/NumPy/Lean conditional scalar scope;
no new neural law/variant or fresh Lean claim. Full graph/source IEEE/AD,
serialization/hash/provenance refinement OPEN. Standard95k–105k active/deployed,
<=1%matched/equal originaldata/train+tune unchanged; no125Mfit exemption,500khistory
separate. No optimizer/corpus/TEST/GPU/promotion/merge/deploy/product work. Source
722/723 synthetic/no-outcome/no-call/template caveats and historical negatives
remain. Original **trained** checkpoint cross-host replay NOT RUN. Physical target,
operators, numerical agreement, p50/p95/RAM/energy/airplane-mode NOT RUN. No remote
inference/preprocessing/embedding/state/speech fallback is introduced.

Next: source-review actual preserved trained-checkpoint import and authentic
matched learned-memory utility; separately qualify full offline physical-device
runtime. Fixture cold replay is a prerequisite engineering result, not that result.
