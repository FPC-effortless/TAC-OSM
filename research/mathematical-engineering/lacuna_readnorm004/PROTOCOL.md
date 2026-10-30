# LACUNA-READNORM004 — paired historical/total slow-read norm control

Registered before any004inference. Parent TAC9dfc585115c75ae1ba19c5e3e0dc007b861d9666,
PNDS ee5c51ddf1235865a313b27da803f107b74724b2. Internal no-fit research only.
Use original LANGMEM087640checkpoints1701/1702/1703. The trained sweep stays
stopped; this protocol cannot reopen001/002/003gates, promote a checkpoint or
establish autonomous retrieval. No new model, fit, tuning, GPU, TEST or merge.

## Paired source and information boundary

Reuse the **already exposed**003DEV cohort: exact registered CPython fractions
and weakref blobs, first4096observedbytes; t256..4095 every16clocks, bins[8,32),
[32,128),[128,240), same most-recent prefix-context match and wrong-address rule.
Source receipts, byte hashes, license and immutable original results are in003.
No fresh-data generalization or model selection is claimed. Reusing these
sources isolates the control change on the same226source/query/bin cells,
154current positions,218past positions, repeated across3models=678measurements.
Verify exact cell/control identities and historical-only predictions against
the original003artifact (tolerance3e-5NLL). Missing/drifting cells fail integrity.
Past target x[j] and future x[t+1] enter scoring only after all004predictions.
Supplied historical addresses and conditional event arrays remain evaluator-only.

## Only experimental change

Reuse003native model, trajectory, ten candidate/control reads A/B/C/D,
matrix_wrong/raw_wrong and their four value shuffles, byte-code readout,
256Breference ring, source-derived keys/values and decoder consumer patches.
Keep original fast read, gates, query, current write and all other prior state.
Discard all patched next states. Original source trajectory always advances.

Let h=Hq, c=Cq and candidate v. Compare both constraints side-by-side on every
candidate: L=003historical-only norm matching, and T=the new dual-norm transform:

u=c/||c||, a=dot(h,u), hp=h-a*u, r=||hp||,
vp=v-dot(v,u)*u, T(v)=a*u+r*vp/||vp||.

Then ||T(v)||=||h|| and ||T(v)+c||=||h+c||; dot(T(v),c)=dot(h,c).
Use float64 construction, stable r=norm(hp), no subtractive square-root clamp;
cast the **total slow read** T(v)+c to nativefloat32 before source injection.
Record squared norms/dots, all final vectors, angular displacement and errors
in float64 and after native cast. Fixed absolute numerical epsilon1e-12.

Degenerate cases, never silently dropped or arbitrary reoriented:
- Exactly c=0: norm-match v to h; if v is nearzero return h and flag fallback.
- Nonzero ||c||<=epsilon: return original h, flag INCONCLUSIVE_NEAR_ZERO_CURRENT.
- r<=epsilon: return original h, flag NO_ORTHOGONAL_FREEDOM.
- ||vp||<=epsilon: return original h, flag INCONCLUSIVE_CANDIDATE_PARALLEL.
- Nonfinite, nonfloating, non-[8] inputs: reject before prediction.
All fallback rows stay in descriptive scores. Any fallback/missing control in
an arm/bin makes its useful-direction gate INCONCLUSIVE; do not interpret a
fallback's unchanged logits as a successful candidate. A must remain identity.

These constraints fix the slow read after the current write. They **do not**
fix ||mix_fast*r_fast+mix_slow*(T(v)+c)||. Measure and report this mixed-read norm
separately; no claim of preserving all decoder inputs/activations. Projection
can alter historical-value content; this diagnoses compatibility of transformed
directions, not a storage improvement. Report past-byte recovery before/after.

## Integrity and scoped math

Before scoring: CPU phone plan, diagnostic parity,003saved-evidence check,
original9checkpoint archive/FIT cold replay, source086adapter tests and pinned
PNDS Lean/Python scoped formulation. Full IEEE/AD and qualification remainOPEN.
Unit tests: random double-norm/dot invariants, identity, all degenerate cases,
invalid values/shapes, independent list reference and local projection AD/FD.
Actual source: original counts100521parameters/1920Btensors and frozen hashes;
transport<1e-5; source key/value reconstruction<3e-5; original003NLL replay<3e-5;
identity/zero-read hooks<3e-5. For every Tread require historical/total norm error
<=1e-10 in float64 and <=3e-6 after native cast; independent reference<=1e-10.
Sample actual prefix-future poison and one local composed projection→original
decoder VJP/FD perowner/seed (h=.001, error<=.002), without optimizer updates.
Fail on any nonfinite result. Record uninstru­mented checks asINCONCLUSIVE.

## Metrics, controls, decisions and budget

For every age/seed/owner/arm: NLL gain original-minus-L/T, T-minus-L NLL,
correct-minus-wrong and correct-minus-shuffle improvements; past-byte NLL/top1;
both norm errors, mixed-read norm change, rotation cosine, degeneracies and
same-source counts. These are paired conditional effects, not iid seed/token
significance or an equal-budget retrained architecture comparison.

For each Tarm/bin provisional direction utility requires>=20distinct cells,
no fallbacks/missing controls, mean original-minus-T NLL>=.001, better than
its Twrong and Tshuffle controls, and positive means in all3seeds and2owners.
Same numerical threshold as003; oldgates/results remain immutable. A signal
with a supplied address remains privileged. No result authorizes training.

Memory budgets unchanged: native256Bslow bank and reference256Bring,1920Bnative
tensors+8Blogicalclock; actual evaluator also retains ring,786432Bdecomposition
arrays and temporary vectors. Record CPU/RSS and read/decoder-call counts. No
mobile latency or full-state efficiency claim. Run once in GitHub CPU Actions,
maximum15minutes. Preserve original outputs, failures and source hashes inGit;
subsequent report fixes must recompute saved artifacts, not rerun inference.
