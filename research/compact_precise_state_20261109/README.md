# COMPACT_PRECISE_STATE_20261109 — preserve FP64 state while compacting

Internal engineering fixtures only, **not independently qualified**, no learned
capability/quality/speed superiority or mobile qualification claim.

## Verified unfinished work and priorities

Current master plan `af307318a18d7fbd97b94ada9f383ff5ed614a4d`, repository/FPC,
real-world/capability/anti-cheat/reward/phone-lab rules, Issue472 and draft721–725
and PNDS115–117 were inspected. Current user100k/matched-budget/offline-only and
two execution levels prevail over ancestor wording. The existing two-level
operator policy was read in tac-phone-lab; it is absent from this source lineage.
No blanket signer setup is imposed on this internal source-only work.

Top3: (1) close precision-state ownership integration, (2) preserve/resolve
remaining machine-domain/numerical gaps, (3) source-qualified learned historical
memory utility plus physical offline inference qualification. #725's R2 storage
result and #721's FP64 numerical repair are already completed, separate evidence.
This increment joins their compatible interfaces; it does not repeat original
84-case numerical qualification or retune a model. Six families remain aligned;
source-specific results do not transfer automatically between them.

## Smallest source change and prior art

Task branch is based on #721 exact head087eee4b. Reuse #725's helper, original
13-test suite and partial NumPy identity reference from f9b6aec0. The **only helper
algorithm delta** is importing source-reviewed PreciseMemoryState and allowing
that exact class. The dataclass adds no fields or hooks to R1LayerState; it is an
explicit FP64 state-version marker. Replacement via dataclasses.replace preserves
it while tensor.clone preserves dtype/value/device and differentiable identity.
Unknown subclasses still fail. Importing the module does not install precision.

Uses established [clone AD semantics](https://docs.pytorch.org/docs/stable/generated/torch.clone.html)
and [detach storage sharing](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html),
not a novel memory/attention mechanism. Original R2/R3/R4/precision/scans/model
Parameter/checkpoint semantics, widths and defaults are unchanged. Calling the
copy on an old FP32 state does not convert it to FP64 or satisfy precision ingress:
reset/replay is still mandatory. Copy is not state-validity or source authority.

## Preregistration and actual checks

Protocol and CPU plan published before runtime in commit
`6e522f9a42b8cc7228a4e128b188ba10d093b00d`.

- Preserved first rejection: ORIGINAL_HELPER_FAILURE.log (old helper rejects exact
  PreciseMemoryState). This is the expected unsupported-version falsifier, not a
  neural/scientific failure hidden by later passing tests.
- **18 new + 13 reused controls PASS**. Includes actual R4 both FP32/FP64 modes,
  three seeds, two exits, multiple continuation chunks, every emitted head,
  repair logits/candidates, all participating parameter/incoming-state gradients,
  independent NumPy precision copy, old FP32/unknown-version rejection, changed
  stream/depth rejection, future poison and CPU BF16-autocast.
- **42 selected inherited/agnostic tests PASS**. Total73selected; not full suite.
- Actual full-size **untrained123,873,263p R4**, FP64 numerator/mass and FP32 learned
  weights, all three exits × two prefix lengths: **6/6 rows PASS**, bit-exact state
  and next-token all-head/repair/candidate outputs at identical partitions.
  Before/after original weight hashes match. No optimizer updates or corpus use.

| Exit | Prefix | Original backing bytes | Compact bytes | Local median copy ms |
|---|---:|---:|---:|---:|
|2|64|8,794,112|143,360|0.031206|
|2|128|9,187,328|143,360|0.032198|
|8|64|35,373,056|770,048|0.133628|
|8|128|37,142,528|963,584|0.139938|
|20|64|88,334,336|1,826,816|0.330911|
|20|128|92,659,712|2,213,888|0.359854|

LOCAL_RESULTS.json includes all seven samples after two warmups, original source
pins, full model geometry, metadata and unchanged weight digests. Full prefill
skips unused logits only; next-token checks include every R4 head. Dedicated
hosted replay is reported separately when available.

## Mathematics and limits

Value copy C(S)=S with identity pullback; unchanged recurrence/normalization,
state dimensions/time and legal attention. Existing NumPy partial formulation
is reused. Actual AD comparisons use original5e-6/5e-5 tolerances, not relaxed.
PNDS116 remains conditional scalar exact-real mathematics; full source/Lean/
IEEE/AD/allocation refinement remains **OPEN**. No new neural computational graph
or canonical variant is registered for a value-copy state-version extension.
Fresh Lean compilation **NOT RUN**; original historical build evidence unchanged.
These software checks do not qualify safety, reward integrity or checkpoint
promotion, nor prove useful learned memory or authentic tool execution.

Copies briefly coexist with original storage and add latency. Drop old aliases
for reclamation; attached AD graphs can still retain allocations, and the helper
never silently detaches. Figures count unique retained tensor backing storage,
**not total RAM, allocator peak, weights or physical-device memory**. Process
lifetime RSS is separately labeled and cannot isolate copy peak. Device target,
operators, numerical parity, physical p50/p95/RAM/energy/airplane mode: **NOT RUN**.
No remote inference/preprocessing/embedding/state/speech fallback is introduced.

No dataset/trained checkpoint/optimizer/GPU/TEST, model promotion, merge or
product work. Standard95k–105k and<=1% matched-arm/identical data/train/tune gates
unchanged; no125M fit exemption, historical500k language separate. Original H64,
COLLECT/FLOCK and11/84FP32 failures remain intact. Source pipeline #722/#723 still
has synthetic outcomes, missing no-call cases and unresolved semantic-template
leakage, so no real tool competence follows from its preparation.

Next: review before adoption at the explicit inference-state boundary, then
obtain actual physical-device evidence. New learned-memory experiments need
preregistered authentic data, eligible model counts and replayable checkpoints.
