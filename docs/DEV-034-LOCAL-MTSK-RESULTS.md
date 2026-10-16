# DEV-034 pilot: five completed arms, local multi-retention numerical failure

The six-arm pilot is **incomplete**, not a successful full factorial experiment.
All five completed arms passed exact separate-process model/action replay and
independently reconstructed condition, split, oracle-label, receipt and physical
cost checks. [Partial inspection](DEV-034-PARTIAL-AUDIT.json) explicitly records
`audit_passed:false`, failed arm and unavailable interactions. Partial inspection
was added post-hoc; it does not relax the original six-arm acceptance rule.

Source frozen before outcomes: `ec0ff80b058308dad8de57eab7d223ec9a81dcba`.
One initialization,216 updates, common AdamW recipe, H512 training, public IDs,
known adaptively used synthetic generator. These are validation outcomes, not
untouched test or fresh-seed confirmation. Each history has96 mature primary
cases and48 cancellation diagnostic cases; physical policies use all144 cases.

| Model | H128 primary | H512 primary | H512 cancellation | Status |
|---|---|---|---|---|
|Global writer, tied retention|51.04%|51.04%|50.00%|Replayed|
|Global writer, independent retention|51.04%|52.08%|50.00%|Replayed|
|Local writer, tied retention|97.92%|88.54%|77.08%|Replayed|
|Local writer, independent retention|Unavailable|Unavailable|Unavailable|Failed at update140|
|Existing masked-local TAC|97.92%|91.67%|83.33%|Replayed|
|Learned schema41|100.00%|98.96%|97.92%|Replayed|

The local tied-retention adaptation learned much more of this distinction than
the global tied writer in this run. It did not beat the existing reference models
on primary accuracy. The failed independent-retention arm leaves the intended
locality-timescale interaction unavailable. No missing performance is imputed,
no seed confidence interval is reported, and no superiority or learning ceiling
is established. Tied retention is not an assertion of identical effective timescales.

## Physical outcomes

H512 selected-inspection policy over144 worlds (1152 entity opportunities):

| Model | Failures | Unnecessary services | Inspections | Total cost |
|---|---|---|---|---|
|Global writer, tied|0|558|0|1152.0|
|Global writer, independent|0|558|0|1152.0|
|Local writer, tied|4|18|6|624.6|
|Existing masked-local TAC|1|16|26|615.6|
|Learned schema41|10|12|0|636.0|

Both global controls serviced every entity; zero failures did not establish
useful tracking. Conversely, schema41's higher primary accuracy did not imply
zero failures or minimum physical cost: that accuracy scores a designated
entity, while costs cover all eight. Model-only and matched-random totals,
H128 costs and selected-minus-model-only differences are in the partial audit.
Local tied retention's H512 selected-minus-model-only cost is positive,
0.00052083 per entity; masked TAC's is negative,-0.0046875. Preserve these signs
without treating tiny differences as robust population effects.

Completed-arm training times were approximately52–127 seconds each; overall
two-process dispatch including validation took352.59 seconds before partial
audit. These observed CPU timings are not FLOP matching or a total economic
efficiency result. Failed-arm CPU time was not separately recorded.

## Numerical finding and prospective repair

The original local independent-retention trial stopped before optimizer update140
when the FP32 gradient-norm guard raised. The original failure snapshot contains
pre-update parameters, optimizer and RNG, not the original gradient tensors.
A [one-batch post-hoc reproduction](DEV-034-LOCAL-MULTI-FAILURE-DIAGNOSIS.json)
from that snapshot reproduces the exception with finite forward loss0.91052896,
finite pre-update parameters, **zero non-finite gradient elements**, and FP64
gradient norm5.4823e19. Here the norm computation overflowed in FP32. This differs
from DEV-032's GRU failure, which had actual NaN gradient elements.

A prospective CPU-FP32 clipping helper computes the norm in FP64 and continues
to reject non-finite gradient elements. On the recomputed failing gradients it
produced finite clipped gradients with norm1.000000007, without an optimizer
step: [diagnostic](DEV-034-PROSPECTIVE-CLIPPING-CHECK.json). Two tests cover overflow
and rejection before mutation. It was **not** used in any original pilot trial,
does not supply a recovered prediction result, and does not establish stable
training. Such large gradients still warrant a dynamics/optimization investigation.
Any continuation using it must be a separately registered cohort, with the same
numerical policy across controls and the original failure preserved.

### Separately registered numerical continuation

After the original failure was documented, a
[post-hoc diagnostic protocol](../research/dev034/numerical-continuation.json)
was committed before continuation outcomes. It resumes the139-completed-update
snapshot through216, changing only norm accumulation to FP64. This is an
adaptive numerical diagnostic, **not** a new complete six-arm comparison or a
replacement for the failed table entry. A future comparative cohort must apply
the numerical policy across controls.

The77 remaining updates completed in39.88 seconds, with maximum measured gradient
norm1.2914e25. Exact checkpoint reload replay and a separate-process evaluator/physical replay passed: [continuation replay](DEV-034-CONTINUATION-REPLAY.json). H128 and H512 primary
accuracy were both62.5%; cancellation was75% at both histories. H512 selected
decisions produced321 failures,232 unnecessary services,136 inspections and total
cost1802.6 over144 worlds. Those actions cost more than the service-everything
controls'1152 on these same worlds. Training completion therefore did not
establish adequate learning or economically useful decisions.

This distinguishes a repaired norm calculation from stable, effective learning.
The full six-arm pilot remains incomplete, the original independent-retention
entry remains unavailable, and the planned factorial interaction stays withheld.
No independent-seed or architecture-ranking claim follows from the continuation.

## Reproducible evidence

Raw checkpoints, reports, failures, gradients, logs and the bounded execution
wrapper are packaged in the workspace at `/workspace/dev034-evidence.zip`.
[The manifest](DEV-034-EVIDENCE-MANIFEST.json) records every payload hash and
the archive hash. Raw run artifacts stay outside Git in accordance with the
repository's artifact policy. The separate numerical-continuation package and
inspection are recorded independently; they do not overwrite original evidence.
The original smoke-only CI success is not a full trained-pilot success.

The implementation remains a memory/prediction candidate, not a trained full
TAC/PLM–CDL–CASM–OSM system. Native MTSK is not included, read addressing is supplied,
and the rank-three reader is a limitation. Retention semantics, independent-seed
reliability, adequate optimization, larger independent evaluation and complete-system
contribution remain open. No new model branch is merged.

Continuation raw evidence: `/workspace/dev034-continuation-evidence.zip`, with
[separate manifest](DEV-034-CONTINUATION-EVIDENCE-MANIFEST.json). All1163 local
tests passed after adding the two prospective clipping tests.

CI [38967409222](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38967409222)
passed integrity, the full1163-test suite, six-arm smoke and smoke replay.
It does not turn the failed trained pilot into a success.
