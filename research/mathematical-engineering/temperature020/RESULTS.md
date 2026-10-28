# TEMP020 — source audit and scalar-calibration engineering

Internally measured engineering evidence, NOT independently qualified. No
original language score replay, calibration fit or new model training. All
original H64, GRAPH017–019 and first-open cache failures are preserved.

## What the exact source shows

[Source] GRAPH019 commit2aeb6313, original WaveCED source SHA256
acf80cb9051b7f2d2c3900d1676eed212890208cd5213088c6751f9cfd24392c:
`e=tanh(embedding(x_t)+pos(phi_position(t)))`, native key=normalize(native_key(e)).
It receives current symbol AND position, not ctx/fast/slow; keys are not limited
to128 literal character identities. The native prewrite memory read uses that
key. Its values also encode current e; there is no shifted context→next-symbol
write in that pathway. Describing this as proven useful context retrieval would
be unsupported, and calling it exactly a bigram table would also be inaccurate.
The decoder separately consumes cumulative prior basis, waves, concept and
source-ordinal aligned reads. No useful associative retrieval follows from this
static graph observation.

[Source] Existing reports already contain native top1/top5 (PR528 full report),
count mixtures (GRAPH009–016), and gradient clipping. Preserve existing genuine
five-seed Vim TAC NLL2.537438 vs NEW GRAPH014 SSM2.518730; the latter is NOT the
older historical SSM2.39256. Unequal complete training compute remains a blocker.
[Conj] A0.018708-nat gap alone identifies neither data/optimization nor architecture
as the cause. A one-step trigram score cannot be inferred from h4 forecasting;
its matched genuine score and train-vs-dev curves are NOT RUN here. The original
120*8*128 schedule exposes122880 target positions/model, with repeated sampled
windows; not122880 unique source bytes.

## Parameter arithmetic checked on untrained original constructors

`python scripts/audit_language_parameter_budget.py --original-root ORIGINAL_CHECKOUT --output RECEIPT`
verifies exact source Git/SHA, constructs models and counts every registered
learned tensor; no forward/backward or optimizer call. PARAMETER_AUDIT.json:

| Vocabulary | FFN width | Embedding | Main output | Concept output | Total |
|---|---:|---:|---:|---:|---:|
|128|605|6144|77568|2048|499290|
|256|605|12288|155136|4096|585050|
|256|501|12288|128512|4096|499978|
|4096|605|196608|2482176|65536|3157850|

[Proven arithmetic/source count] Expanding128→256 at unchanged widths costs
85760, not6144; the head must expand too. Width501 restores the500K band and
changes capacity, requiring new preflight/data/compute/gradient gates before
training. These counts do not prove every coordinate is active; the original
512 graph q/k cancellation is preserved, not silently repaired. No BPE/byte
model was fitted. Raw BPB requires reversible byte-complete source scoring:
the historical >=128→? codec is lossy, so report bits/mapped-symbol instead.

## Implemented mathematical layer

[Lit] Guo etal2017 scalar temperature. For a=1/T>0:
CE(a)=log(sum exp(a*z_i))−a*z_y;
dCE/da=E_p[z]−z_y, d²CE/da²=Var_p(z)>=0 in exact real arithmetic.
[Proven scoped property] Multiplication by positive a preserves all pairwise
logit orders and ties, hence top1/top5 rankings. Class-specific bias does not.
Finite-precision underflow/rounding and tied-mode policies remain explicit limits.
The implementation uses stable centered FP64 arithmetic, fixed64-step bisection
on[.25,4], candidate selection including identity, immutable scalar/data digest,
document-disjoint score API and explicit codec metrics. No score labels select
parameters. Same-calibration-loss nonincrease is enforced by identity fallback;
there is NO unseen-loss guarantee. Regression includes score-set harm despite
calibration improvement. One scalar per fitted seed must be included in deployed
calibration/resource accounting; it is not a free accuracy gain.

[Lit] Cache mixing is established (Grave etal2017), and already implemented in
this repo. GRAPH016's new-source Python mixture FAIL is preserved. Learned
state-dependent gate, class-bias fit, causal conv/shifted writes, warmup/cosine,
TBPTT and dynamic evaluation remain future separately registered experiments,
not implemented gains. Dynamic adaptation must use only same-owner observed
past with exact resets, added compute and no scoring-label lookahead.

## Evidence and next step

Constructed tests cover derivatives, Torch local AD, document isolation, positive
ordering, bias counterexample, boundaries/flat logits, metric denominators and
score-harm negative control. PNDS companion provides scoped exact-real ordering
and identity-guard theorems plus independent executable Python derivative law;
full model IEEE/AD and generalization remain OPEN. Fixture/source counts have
zero weight as learned capability evidence.

Actual source/checkpoint/logit calibration: **BLOCKED / NOT RUN**. This workspace
still lacks the original language archive and authenticated36-document protocol;
no fabricated source hashes, new checkpoint substitutes or fitted scalar.
Next restore original immutable artifacts, freeze actual calibration/score case
identities and obtain exact original logits through authorized source replay.
Then use the preregistered disjoint exposed-DEV comparison before mixing or
changing training. Any independent fresh-cohort claim requires its own protected
authorization; this increment keeps TEST sealed. No GPU or promotion.
