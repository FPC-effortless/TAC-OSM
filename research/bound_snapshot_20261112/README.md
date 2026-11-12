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

## First hosted runtime blocker and recovery

Original capture43173654413 passed219tests and preserved actual untrained
50,205-parameter FP32/FP64 fixtures. Cold reader failed BEFORE import/replay: the
Safetensors Torch adapter imports packaging, absent in its clean environment.
This is a runtime prerequisite failure, not a numerical/model result.
Original artifact13923122104/ZIP and exact relevant error excerpt are preserved.
Recovery pins packaging25.0 explicitly, reads the SAME first capture, and passes
its independently checked index hash via job output from REPLAY_SOURCE.json.
No weights/model/code/inputs/tolerances are regenerated or changed.
The two50,205p objects are untrained software fixtures, never compliant new fits
or deployed learned models. Standard100k/matched gates remain unchanged.

Build-time research environment dependencies: torch2.10.0 CPU, numpy2.3.5,
safetensors0.6.2, packaging25.0; pytest8.4.2 for controls. Runtime never invokes
pip or downloads a model. Physical platform/operator packaging NOT RUN.

## Hosted result and permanent evidence

[Recovery run43174631095](https://github.com/FPC-effortless/TAC-OSM/actions/runs/43174631095)
passed219controls and cold replay on a new runner. It downloaded the original
[capture43173654413](https://github.com/FPC-effortless/TAC-OSM/actions/runs/43173654413)
instead of generating replacements. Both FP32/FP64 cases restored the original
binding byte-exactly into different-initialization/perturbed-buffer prototypes;
all40output/state tensors per case were byte-exact, maximum absolute error0.
These are observed results, stronger than the preregistered1e-5 tolerance for
these fixtures, not a general cross-platform guarantee. No real data was scored.

Original capture artifact13923122104, recovered replay artifact13924430231 and
controls artifact13923427080 are preserved losslessly as ZIPs under
`research/evidence/bound_snapshot_20261112/`. Raw ZIP digests are independently
pinned in `verify_saved_evidence.py`; that stdlib-only command checks archived
source/manifest/payload/expected-output identities,219tests, CPU plans and two
receipts without importing Torch or executing a model:

```sh
python -B research/bound_snapshot_20261112/verify_saved_evidence.py
```

The first packaging import failure remains recorded in REPLAY_SOURCE.json and
INITIAL_COLD_READER_ERROR.log. Compulsory preflight43174631235 also passed. This
closes fixture persistence/cold replay only. Authentic trained-checkpoint import,
learned-memory utility and physical offline qualification remain NOT RUN here;
PNDS Lean was not rerun and serialization/provenance refinement remains OPEN.
