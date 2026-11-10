# STATE_INGRESS_20261110 — explicit restored-memory contract

Internal engineering fixtures, **not independently qualified**, no model fit,
corpus, checkpoint/TEST opening, optimizer, GPU, promotion, merge or deployment.

## Choice, boundaries and reuse

Reviewed current master plan af307318, AGENTS/CHARTER/CODEX/FPC Universal,
real-world/capability/anti-cheat/reward/phone-lab rules, execution levels, Issue472,
drafts721–726 and PNDS115–117. User100k/matched-budget/offline-only and internal
versus independent-qualification instructions prevail over ancestor text.

Top3: restoration-time machine-domain checks, preserving numerical counterexamples,
then authentic matched learned-memory utility/physical offline qualification.
This implements ME-R03/04 from review724 after completed copy work725/726. It does
not repeat those resource screens, H64, original11/84failures or precision repair.
Six families stay aligned, with no automatic cross-family capability transfer.

`validate_restored_stream_state` is explicit, read-only and never installed in
forward. Caller supplies expected stream IDs, depth and observed-token cutoff
independently of restored metadata. Existing source configurations, model law,
Parameter objects, gradients, checkpoints, state values and copy semantics stay
unchanged. Invalid state is rejected, never clipped/cast/repaired. For currently
reviewed R1/R2/R3/R4 with FP32 learned weights, require:

- exact known state/block versions, shapes/dtypes/device, explicit stream/depth;
- finite numerator/mass/conv/KV, D>=0 and abs(N)<=D with **zero tolerance slack**;
- epsilon positive **normal** in actual FP32/FP64 memory dtype, <=1e-3;
- local KV position equals caller cutoff, and length=min(cutoff,window-1).

The positive-normal epsilon rule is a conservative new ingress policy to avoid
underflow/subnormal flush dependence. It is not a rewrite of the old constructor:
1e-100 still constructs in old source but the FP32 ingress guard rejects it; it
is normal and allowed with FP64 memory. Full native forward can still be called
without this optional validator. It does not fix all invalid live writes, NaN
observations, nonlinear/normalized AD errors or authenticate cached history.

Read-only checks may synchronize tensors: use at restoration, not each token.
Caller must own/quiesce writers; no atomic snapshot or thread-race protection is
promised. Even correctly shaped/bounded state can be forged from another source
or checkpoint. Independent source/hash/access checks remain separate.

## Protocol and actual checks

Preregistered protocol/router72c4551f before execution; bounded GitHub CPU5min.
**60 new tests passed locally**: 36 seed/mode/exit/prefix combinations on actual
reduced original R4 plus16invalid-state controls,2dtype-aware epsilon cases,
2NumPy randomized domain checks,3all-parameter/incoming-state AD checks and1AMP/
context check. Original/detached/compact state metadata/values/RNG preserved;
paired next-token outputs/state unchanged; gradient tolerances5e-6/5e-5 unchanged.
Selected inherited checks are recorded separately. Fixture sizes are no-fit;
no125M full resource rerun or fit exception, standard95k–105k/<=1%trained-arm and
identical realdata/train/tune gates unchanged. Historical500k remains separate.

Independent NumPy predicate in reference.py; a partial PNDS Lean/Python companion
is prepared against exact implementation source. Formal gate status must be read
from its actual build evidence: uncompiled source is not a proof. Normalized-read
bounds are conditional exact-real scalar results, not full source IEEE/AD,
parser/provenance, checkpoint authenticity or safety/reward certification. No
new learned graph/canonical model variant is claimed. Whole-model refinement
remains OPEN and blocks mathematical qualification claims.

Primary prior art is the existing source/PNDS cone assumptions and standard
PyTorch finfo/isfinite/dtype semantics, not an architectural invention. Source
pins freeze the inherited recurrence/copy implementations; invalid machine
values are a boundary diagnosis rather than evidence of failed trained quality.

## Reproduction / next

```sh
python scripts/phone_lab.py plan research/phone_queue/state_ingress_20261110.json
PYTHONPATH=src:. python -m pytest -q tests/test_state_ingress.py
```

Review before adopting at a real checkpoint import. Cross-host random initializer
mismatch from726 remains, same-checkpoint cross-host replay NOT RUN. No trained
memory gain or independent qualification inferred. Dataset pipeline722/723 still
has synthetic/no-observed-outcome, no-call and template-leakage caveats. Actual
device target/operators/numerical parity/physicalp50-p95/RAM/energy/airplane-mode
NOT RUN. No remote inference/preprocessing/embedding/state/speech fallback.
