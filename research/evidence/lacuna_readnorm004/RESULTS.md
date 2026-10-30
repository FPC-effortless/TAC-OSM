# LACUNA-READNORM004 — one narrow privileged short-age signal

Internally measured, not independently qualified. Paired frozen-checkpoint
diagnostic on the **already exposed RETRIEVAL003 DEV cohort**, not a fresh
generalization evaluation. No model fit, tuning, GPU, protected TEST, promotion
or merge. All earlier001/002/003failed gates remain unchanged.

The new dual-norm control passed its registered exploratory utility gate only
for **D: evaluator-supplied key plus bounded raw history, ages8–31**. It improved
NLL by0.005613 versus the original model, but only0.00007078 versus its wrong-key
control. That tiny correct-address margin, exposed data and limited source
support do not establish robust address-specific or autonomous recall.

## Paired method and scope

Original three LANGMEM087640checkpoints1701/1702/1703, original two source
documents fractions/weakref, same prefix-selected contexts and hard wrong keys.
No candidate is selected using the future next byte.226distinct source/query/bin
cells repeat across3models=678measurements;154distinct current positions and
218historical positions. Each measurement contains10candidate/control directions
under2methods. Original trajectory advances; all counterfactual states discarded.

L reproduces003historical-only norm matching. T additionally preserves total
**slow** read norm after adding the unchanged current write. For h=Hq,c=Cq,
candidate v: u=c/||c||,a=dot(h,u),r=||h-au||,vp=v-dot(v,u)u,
T(v)=au+r*vp/||vp||. Explicit original-history fallbacks handle degeneracies.
The same original source consumer patch, fast read, gates and other state are
used. Historical-only next-byte NLL exactly reproduces every saved003score.

All6780Tdirections were regular: no missing controls or fallback rows. Both
norm constraints were independently recomputed from the exact saved vectors,
including the nativefloat32cast. Native A remains an identity control.

## Next-byte utility under the new control

Values are **original NLL minus intervention NLL**; positive is favorable.

| Age | Cells shared across seeds | B supplied key/matrix | C native query/raw | D supplied key/raw | D wrong key/raw | D value shuffle | Passing new Tarm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8–31 | 28 | +0.006281 | +0.005170 | +0.005613 | +0.005542 | +0.004381 | D only |
| 32–127 | 97 | −0.005401 | −0.004778 | −0.005646 | −0.005681 | −0.006852 | None |
| 128–239 | 101 | +0.001216 | +0.000699 | +0.000696 | +0.000598 | −0.000288 | None |

Gate: at least20distinct cells, no fallbacks/missing controls, mean improvement
>=0.001, better mean than its wrong and shuffled controls, and positive means
in all3seeds and2owners. This rule requires positivity versus the original in
each group; it does not require per-group superiority over the wrong control
or a minimum correct-minus-wrong margin. That distinction limits the signal.

D8–31passes exactly as registered. Its original-model gains byseed are
0.014382,0.000278,0.002180; bydocument0.006646(fractions),0.000863(weakref).
The28short-age source cells comprise23fractions and only5weakref cells, reused
across all3seeds. Correct-minus-wrong gains byseed are0.000133,**−0.0000296**,
0.000109; bydocument0.0000819,0.0000196. Thus the correct-address advantage is
not consistent across seeds even though the registered exploratory gate passes.
Its advantage over the shuffled-value control is0.001232.

Bshort-age fails weakref(−0.000122); Cshort-age fails seed1703 and weakref and
is inferior to the wrong-key raw mean. B128–239exceeds the original-model mean
threshold but fails seed1702 and weakref. No middle/older bin passes. All arm,
control, seed and owner scores are retained in the original artifact.

## What changes relative to the original norm control?

| Age | Dgain under003L | Dgain under004T | Dpast-byte accuracy L→T |
| --- | ---: | ---: | ---: |
| 8–31 | +0.004923 | +0.005613 | 34.52%→20.24% |
| 32–127 | −0.009198 | −0.005646 | 21.31%→12.37% |
| 128–239 | −0.000979 | +0.000696 | 21.45%→11.55% |

The stricter norm control changes prediction effects and moves short-age
weakref from unfavorable to favorable. Loss effects improve versus L even
while literal historical-byte recovery decreases. This is evidence that read
compatibility matters under these conditional interventions, not proof that
stored historical content caused the improvement. Correct and wrong controls
often move together. Projection also changes value content, so this is not a
pure scalar-rescaling intervention or a proof that magnitude alone caused003.

The mixed fast/slow vector norm is still free to change. For Dshort ages its
mean signed change is−0.003464 underT versus−0.005267 underL, despite preservation
of both slow-read norms. No claim of controlling every decoder input/activation
magnitude is made. The decoder was not retrained for any substituted direction.

## Integrity, memory and compute

- Source protocol4bc354941f9b3154235b71f3b076a306589cfb31 preceded004inference.
  Exact source blobs/receipts, old source helper hash and checkpoints authenticated.
  Source/control cell identities unchanged and all original003next-byte scores
  reproduce exactly (maximum NLL error0).
- Maximum Tnorm errors: historical1.50e-15,total1.45e-15 in float64; after native
  cast both<1.42e-8. Required thresholds1e-10/3e-6. Independent list reference
  error<=4.17e-17; native A maxlogiterror3.82e-6; source slow_zero error0;
  transport<=1.34e-7; reconstructed key/value<=2.69e-7/8.95e-8.
- Six first-eligible prefix-future poison checks passed with maxerror7.63e-6;
  six composed projection→original-decoder local VJP/FD checks passed with
  maxerror0.000229<0.002. These sampled/local checks are not a global theorem.
-39Actions source/projection/probe/replay tests passed. Original9checkpoint
  replay passed within the documented historical-output/086RNG limitations.
  Local full suite1639passed/4skipped; one existing profiler warning only.
- PNDS90scoped Lean/Python proofs passed corrected CI41010497977; bridge CI
  41010648702also passed. Initial failed proof build41010142268 is preserved.
  Full neural IEEE/AD, all-parameter verification, independent qualification
  and coordinated legacy registry migration remain OPEN.
- Native slow bank256B; raw reference ring256B including clock; native tensors
  1920B+8Blogicalclock. Actual evaluator additionally holds the ring,786432B
  decomposition arrays and temporary vectors/reports. Peak process RSS415002624B.
  These are conditional bank-budget comparisons, not autonomous full-state parity.
- Main stream24576observed-byte steps;13560main counterfactual decoder calls,
  plus separately scoped integrity checks. Process CPU122.9252s with2threads.
  No production decode latency or mobile efficiency claim.

## Original evidence and decision

[Original successful run41010825969](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41010825969),
job123916118312, artifact12957059683, execution
e2a997d882e536dc706c21c88a8219ad57458b5d; diagnostic code
747ddf536014409975285b759247593cc96f8eee.
Original ZIP SHA2565275c236670b0fb3d27e059b60d8cb713549e9810ec96c3c4021d7ef304a6dcf.

`ORIGINAL_ARTIFACT.zip` preserves the full exact33.9MB RESULTS.json, vectors,
receipts and proof-CI metadata. `SUMMARY.json` is a compact exact projection
excluding rows. The optional local extracted RESULTS.json is not duplicated
inGit. The stdlib-only verifier authenticates originalZIP, paired source
identities, baseline scores, both saved norm constraints and all summary/gate
arithmetic without model execution. No rescoring was performed for reporting.

Preserve the new privileged short-age exploratory signal and its small control
margin. Retain all older negative gates and keep the trained sweep stopped.
Autonomous address selection did not pass; fresh-owner confirmation with a
prospectively meaningful correct-versus-wrong criterion is needed before a
broader capability claim. This exposed-cohort diagnostic alone does not justify
training a retrieval index or integrating another memory architecture.
