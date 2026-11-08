# COMPACT_STATE_20261108 — explicit carried-state ownership

**Internal engineering, no fit. Not independently qualified.** Stacked on
[review #724](https://github.com/FPC-effortless/TAC-OSM/pull/724), implementing
ME-R01. This is storage ownership engineering for the unchanged CSL lineage,
not a new recurrence, memory-retrieval architecture or trained quality result.

## Source and work selection

Master plan `af307318a18d7fbd97b94ada9f383ff5ed614a4d`, AGENTS/CHARTER/CODEX,
FPC Universal, real-world/capability/anti-cheat/reward/phone-lab rules, Issue472,
current drafts #721–724 and PNDS #115–117 reviewed. The execution-level policy is
absent from this historical source lineage/default branch; its existing local
operator-amendment text was inspected in tac-phone-lab. The user's current two
execution levels prevail: this internal diagnostic needs no external signatures;
protected evaluation/reward/safety/promotion gates are unchanged.

Priorities: (1) compact streaming-state ownership, (2) remaining machine-domain
and normalized backward defects, (3) source-qualified learned-memory utility
with matched budgets and full offline device measurement. #721 already provides
an optional FP64 remedy; #722/#723 already cover source preparation/byte lineage.
None is repeated or merged. Original H64, COLLECT/FLOCK and 11/84 FP32 failures
remain intact. No result transfers automatically across the six families.

## Change and mathematical scope

`compact_stream_state(state)` explicitly clones each supported tensor with
contiguous layout, preserving state class, stream IDs, positions, depth,
shape, dtype, device, values and autograd connectivity. Original model sources,
Parameter objects, checkpoint names/values and `detached()` are unchanged.
Unknown state subclasses, quantized/sparse/meta tensors fail closed. Only the
source-reviewed CSLState/CSL125State with LayerState/R1LayerState/LocalKVState
versions are supported; later R4/FP64 state types require separate source review.

Mathematically, the value map is identity: C(S)=S, D C(S)[v]=v, and
F(x,C(S))=F(x,S). Copying changes backing allocations and strides, which are
extrinsic to the unchanged learned value graph. Source tests check all heads,
state tensors and every participating reduced-model parameter/incoming-state
AD tensor; this is not a formal complete IEEE/AD refinement. Existing PNDS116
exact-real scalar hypotheses/proofs are unchanged. No new whole-model or
allocation theorem is claimed; PNDS source/IEEE/AD mathematical-readiness remains
**OPEN**. No canonical variant registry is rewritten for a value-preserving
storage utility. New neural operations would require both PNDS formulations.
`reference.py` independently expresses value-copy and identity pullback in NumPy
without importing the Torch implementation; tests compare exact bytes and a
shared-view input adjoint. It is explicitly partial, not a new full-model proof.

This uses established [PyTorch storage/view semantics](https://docs.pytorch.org/docs/stable/notes/serialization.html#saving-and-loading-tensors-preserves-views),
[differentiable clone](https://docs.pytorch.org/docs/stable/generated/torch.clone.html)
and [storage-sharing detach](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html).
These are framework behavior, not architectural novelty.

## Actual local evidence

Protocol and CPU router receipt published first in `d9f8d25bfeb0b0aa3260bc8fe7cebce40afb2084`.
13 new controls + 80 selected original/model-independent tests passed. Fixtures
are untrained engineering evidence, not authentic-data prediction results.

Local full original 123,822,574-parameter model, one CPU thread, no optimizer:

| Exit | Prefix | Original backing bytes | Copied bytes | Median copy ms |
|---|---:|---:|---:|---:|
| 2 | 64 | 4,599,808 | 77,824 | 0.032308 |
| 2 | 128 | 4,993,024 | 77,824 | 0.031176 |
| 8 | 64 | 18,595,840 | 507,904 | 0.133807 |
| 8 | 128 | 20,365,312 | 701,440 | 0.134067 |
| 20 | 64 | 46,391,296 | 1,171,456 | 0.338888 |
| 20 | 128 | 50,716,672 | 1,558,528 | 0.361732 |

All six rows PASS: exact same-partition next-token outputs at every emitted head,
all carried tensor values/metadata, RNG identity and no-grad compact backing
bytes equal logical bytes. V1 raw receipt is retained unchanged. Subsequent
receipt adds before/after weight digests; no scoring or tolerance change.
Samples include all seven repetitions after two warmups; timings are CPU-local
copy overhead, not end-to-end inference or device latency. Hosted evidence is
recorded separately. Full repository suite and fresh Lean compilation **NOT RUN**.

## Use and limitations

```python
with torch.no_grad():
    outputs, state = model(observed_tokens, state=state, stream_ids=ids)
    state = compact_stream_state(state)
```

The utility never detaches; attached training graphs can still retain old
storage. Release old states/aliases to reclaim their allocations. Copy briefly
coexists with the old backing storage and adds work; it is opt-in, not installed
on every token. This changes no context window, memory precision, retrieval
utility or model parameter count. Storage figures exclude weights, allocator
reservations, autograd and runtime. Process lifetime RSS high-water is separately
labeled, not copy-isolated peak RAM or total-device memory. An isolated allocator
peak copy measurement is **NOT RUN**.

No corpus, trained checkpoint, TEST, optimizer/GPU, promotion, deployment or remote
inference fallback. No new 125M fit exemption; standard95k–105k active/deployed,
<=1% matched arms and identical data/training/tuning gates remain. Historical
500k language work stays separate. The default-size object is random-weight
storage verification only. Actual target device, operators, numerical parity,
physical p50/p95, RAM, energy and airplane-mode test: **NOT RUN**.

Next: source-review support for later precision-state versions before adoption;
qualify actual full local streaming inference on a physical authorized device.
Reliable learned historical retrieval still requires authentic matched evidence.
