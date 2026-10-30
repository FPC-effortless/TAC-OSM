# LACUNA-RETRIEVAL003: storage, address and readout decomposition

Prospective registration, 2026-10-30, before acquiring or scoring this cohort.
Internal research; three unchanged original LANGMEM087 640-update checkpoints
(1701/1702/1703). No fit, tuning, TEST, GPU, promotion or reopening of the failed
SURVIVAL001/MATCHED002 utility gates. Parent source f92842b83a42fa82366612e15f17a0f1b992fe4b.

## Fresh source cohort

PSF CPython v3.13.0 commit 60403a5409ff2c3f3b07dd2ca91a7a3e096839c7:

| Document | Publisher Git blob | Full bytes |
| --- | --- | --- |
| Doc/library/fractions.rst | 7e615ed230a340bdf4c8c0d6f4bf5640771bc31c | 9283 |
| Doc/library/weakref.rst | 2a25ed045c68bd2da075611f0caf83a143aa0c0a | 21908 |

Only publisher tree metadata has been accessed at registration. Paths are absent
from the inspected original FIT/DEV receipts and source/results repository text.
This is a new document cohort for this diagnostic, not a claim of globally novel
phrases or an independent publisher. PSF License v2, code examples additionally
0BSD; preserve original LICENSE. Acquire exact blobs after this registration,
record SHA256 before inference; fail on size/Git blob mismatch. Keep TEST closed.

Observe first4096 bytes, with next-byte labels up to offset4096. Query every16
clocks t=256..4095. At each clock select the most recent observed continuation j
in each age bin [8,32), [32,128), [128,240), whose preceding four bytes equal the
currently observed four bytes. No availability or cohort selection uses x[t+1]
or equality between x[j] and x[t+1]. All seeds reuse identical source cells.

## Fixed four-condition comparison

The native slow matrix is **256B**, not512B (8x8 float32). Reference raw ring is
exactly256B =8B uint64 clock +248 raw bytes, with no saved key/value array. Derive
historical keys from their four original context bytes using the unchanged
lexical embedding/writer; derive values using the original byte, absolute clock,
position/value encoder. Validate reconstruction against live source traces.
Historical candidates j<t must have all four context bytes retained in the ring.

Let native source update be S+=H+C, H=rho*S*(I-eta*k*k^T),
C=eta*v*k^T. Native query q and mixing gate are computed from unchanged prefix
state. The historical diagnostic key a is the writer immediately before j,
and encodes no byte x[j]. Use the same set of bounded exact records for C/D:

| Arm | Address | Historical read |
| --- | --- | --- |
| A | native q | H*q |
| B | supplied historical a | H*a |
| C | native q | sum_i softmax(8*dot(q,k_i))*v_i |
| D | supplied historical a | sum_i softmax(8*dot(a,k_i))*v_i |

Thus the query-to-raw-history rule is explicit, prefix-only cosine attention
(normalized native keys/query), temperature8 fixed without tuning. It is a
diagnostic reference, not an exact-match oracle: colliding keys can retrieve
different continuations. Report historical target attention mass, key rank,
equal-key multiplicity and native matrix singular values/effective rank.
Also separate the transported target event from all other old writes under
the supplied key; report target/interference norms, their ratio, read/value
cosine, and different-context key cosines>.9. These remain evaluator telemetry.

Decode every historical read through the same frozen byte-code cosine readout:
softmax(8*byte_codes*normalize(read)); report actual past-byte NLL/top1. Its
temperature is fixed, untrained and uncalibrated. Report original encoded value
and literal historical byte as separate oracle references. Oracle values,
event decomposition and supplied addresses never enter autonomous predictions.

## Decoder intervention and controls

For a one-clock controlled decoder experiment, rescale each historical read to
||H*q|| (zero stays zero), add the **unchanged C*q**, replace only the slow read
at all three native-read consumers in the original graph (fast, slow, norm).
Keep fast read, query, mix, direct current write, other prior state and weights
fixed. Discard every counterfactual next state. A must match original logits
within3e-5; independently validate slow-read-zero against source slow_zero.
This tests directions at native amplitude; it is not matched retraining or an
autonomous alternative rollout. Report zero/near-zero read counts.

Wrong address: select an available retained j' with distance>=5 from j, different
four-byte context, and target j excluded from its four-byte context; choose
closest source key cosine to a, tie oldest. Selection uses prefix only. No clean
wrong address => INCONCLUSIVE, report counts. Both compressed and raw banks use
this same wrong address and identical readout/amplitude convention.

Shuffled values: cyclically shift already observed value vectors by one record,
leaving all keys, times and transport weights fixed. For native H, use exact
conditional event coefficients and ALL old events; for raw history use all
retained records. Both preserve their respective value multisets. Their record
sets differ; report this explicitly. Freeze original trajectory gates/keys for
conditional transport; never call it a complete nonlinear counterfactual rollout.

Every original/arm/wrong/shuffle prediction is committed before revealing y=x[t+1].
Measure next-byte NLL of original decoder separately from past-byte recovery.
Each query is charged; raw reads examine every retained candidate. No budgeted
selective policy or speed superiority is claimed. Record CPU/RSS and logical
bytes, including evaluator coefficients, keys, values and raw reference ring.

## Integrity, support and decision

Run archive/FIT clean replay, unchanged model count100521/state1920B, weight
hashes, owner reset, reconstructed key/value parity<3e-5, conditional sum<1e-5,
identity/zero-read hooks<3e-5, finite outputs, future-suffix poison of actual
prediction preparation, and one local read VJP/finite difference per owner/seed
(h=.001, tolerance.002). Unimplemented probes remain INCONCLUSIVE. Register
the existing actual086 ResearchProbeAdapter plus the evaluator-only read patch.

For each age bin report shared distinct source/query cells, all owners/seeds,
and every A/B/C/D, wrong and shuffle result. Require>=20 distinct cells, no
missing control, mean original-minus-arm NLL>=.001, better than its wrong and
shuffled counterparts, and positive original-minus-arm means in each seed and
owner for a provisional **diagnostic** utility signal. Supplied-address signals
remain privileged. No result automatically reopens any prior training gate.
No iid significance claims across repeated seeds or overlapping contexts.

Storage substitution is bank-budget matched only: both banks256B; original
tensor state1920B +clock8B. The actual evaluator also holds the reference ring
and reconstruction/decomposition workspace. Native upper/query states were
generated by the original model, so these are conditional read interventions,
not equal-budget independently rolled-out architectures. Any architectural
comparison requires a new registration and genuinely matched full state.

Interpret main effects and interactions as evidence about these defined
operators, not identification of a unique biological/architectural cause.
Original 001/002/088 negatives stay immutable. Longer-horizon training remains
untested. Full neural IEEE/AD, retrieval qualification and learned-policy efficacy
remain OPEN. PNDS must provide scoped Lean/Python read equations and exact source
bridge before scientific execution. CPU GitHub Actions, maximum15 minutes.
