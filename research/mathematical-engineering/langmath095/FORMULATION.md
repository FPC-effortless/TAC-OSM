# LANGMATH095 mathematical representation

Status: **PARTIAL OPERATOR FORMULATION; WHOLE-MODEL CERTIFICATION OPEN**.
This is an additive, no-fit diagnostic/candidate record. It preserves the
original source, model bytes, failed experiments and historical proof statuses.
It does not establish prediction quality, learned structure, useful persistence,
transfer, verified learning, resource superiority or theory-level intelligence.

## Source and typed computation

Ancestor: `FPC-effortless/TAC-OSM` at
`d08fbba33990c503446c2a7d5073219a767f2f62`. The inherited operator sources are
`context026/context_transition.py` and `langlayer038/recurrence_only.py`.
`langmath095/candidate.py` wraps a copied eligible descendant; default policies
delegate its original forward. New policy combinations are explicitly untrained
candidate graphs, not unchanged checkpoint inference. A concrete publication
must pin the candidate implementation commit in addition to the ancestor SHA.

Input is observed token IDs `X[B,T]` with `T>0`, integer dtype, vocabulary IDs
`0..V-1`; output is `logits[B,T,V]` predicting the next token. At index `t`,
`X[:,0:t+1]` is available; future tokens and targets are not. Initial base states
are zero and time starts at zero. The inherited forward remains window-reset
only. Carrying whole-model state requires a separately qualified adapter.

The base graph is:

1. `e[B,T,48] = tanh(embedding(X) + position(phi(t)))`. The lexical key path
   separately uses `tanh(embedding(X))` plus the preceding three lexical vectors.
   A source-position prefix sum produces `context[B,T,88]`, excluding current
   position and divided by elapsed positions. These features are not separate
   recurrent fast/slow memories.
2. Native keys and values have width eight. Native write at `t` uses the
   preceding lexical key and the current observed value, then reads the updated
   memory with the current key. The first native write is disabled. Two aligned
   memories have shape `[16,8]`, positional write keys, and a read from their
   prewrite state at the preceding ordinal address. Native memory is `[8,8]`.
3. Eight two-coordinate wave states use damped rotational affine updates.
   Six 16-coordinate persistent concept seeds use input-only convex updates.
   Their prewrite states enter the current prediction. The 88-wide `fast` and
   `slow` quantities are feed-forward tanh projections of current/legal features.
4. Current concept evidence `G[B,T,6,16]` defines a symmetric nonnegative
   six-node adjacency, Laplacian `L`, and solve
   `C=(1.25 I + .25 L)^(-1)(G + .25 previous_seed)`. The evidence mean defines
   softmax pooling weights. `C` is not fed back into persistent concept seeds.
5. Concatenating widths `88+88+8+16+16+16+48` gives 280 features. The inherited
   layer norm/upper blocks and gated feed-forward head produce vocabulary
   logits, plus a concept output projection. Gradients can reach all these
   paths; numerical nonzero gradients must be measured rather than assumed.

`write_mask[B,T]` gates associative-bank writes only. It does **not** remove
tokens from lexical context, the position clock, prefix averages, waves, concept
seeds or upper recurrence. A false associative write substitutes `(I,0)` and
preserves that bank's bytes. Entity separation and caller-owned state are
adapter obligations. The reference performs no optimizer updates and consumes
no future labels, rewards or verification feedback.

## Exact equations and optional changes

For row-state right-affine updates, `S' = S P + Q`, chronological composition is

\[
(P_1,Q_1)\circ(P_2,Q_2)
 = (P_1P_2,\ Q_1P_2+Q_2).
\]

The left pair is executed first. Composition is associative in exact arithmetic
and has identity `(I,0)`. Arbitrary initial state and chunk continuation follow
from this action law. Floating-point arithmetic is not associative; finite
tolerances, output/state/backward comparisons and explicit rounding witnesses
are separate obligations. Balanced scan work/span counts must not be inferred
from a scalar Lean identity.

The associative update is

\[
M' = \alpha M(I-e k k^\top)+w v k^\top.
\]

For `||k||₂≤1`, `0≤e≤1`, `0≤alpha≤1`,
`||alpha(I-e kkᵀ)||₂≤alpha`. If `||v||₂≤V` and
`0≤w≤c(1-alpha)`, then

\[
\|M'\|_F\le\alpha\|M\|_F+wV\le cV
\quad\text{whenever}\quad\|M\|_F\le cV.
\]

The bounded-write candidate uses `w=(1-alpha)*sigmoid(raw_write)` and `c=1`.
Aligned tanh values have 16 coordinates, so `V≤4`. This sufficient bound applies
to arbitrary key schedules but conservatively couples long retention to small
writes. It says nothing about retention usefulness. The original independent
gates have fixed-key gain `w/[1-alpha(1-e)]`, which can be large; a constant-key
denominator bound alone is not an arbitrary-key matrix norm invariant.

For a scalar upper coordinate, `h'=alpha*h+(1-alpha)*offer`. If both inputs are
in `[-1,1]` and `alpha∈[0,1]`, the interval is invariant. At a fixed input
schedule, two previous states differ after one update by
`alpha*(h-g)`. The original `.995*sigmoid` cap bounds direct carry by `.995^L`;
it does not bound the full model's history dependence. The multiscale candidate
explicitly changes the cap/initialization to target log-spaced decay horizons;
it is a reinitialized candidate, not a frozen-checkpoint lesion.

For the concept solve, symmetric nonnegative adjacency gives `L` positive
semidefinite, `lambda_min(H)≥1.25`, and `||H^(-1)||₂≤.8`. With six concepts the
source adjacency gives the conservative `condition₂(H)≤2.2`. Uniform pooling
eliminates graph influence because `1ᵀL=0`:

\[
\tfrac16\mathbf 1^\top H^{-1}B
 = .8\,\tfrac16\mathbf 1^\top B.
\]

The graph-disabled candidate sets `L=0` while retaining the prior-seed mixture.
Near-uniform pooling may make a graph weakly observable; this is not proof that
the trained graph is useless. The positional candidate replaces only
`log1p(t)` by `log1p(t)/(1+log1p(t))`, bounding that feature in `[0,1)` and leaving
the existing trigonometric features unchanged.

## Formulations, tests and remaining obligations

The companion PNDS paths are `PNDS/LANGMATH095.lean`,
`foundation/v1/langmath095_reference.py`, and
`foundation/v1/test_langmath095_reference.py`. The Lean declarations cover
scalar composition/identity/action, nonwrites, retention and a **conditional**
norm-envelope implication. In particular, the actual matrix one-step norm
inequality is a stated premise of the Lean envelope theorem; its matrix proof
has not silently been replaced by scalar algebra.

The independent NumPy reference covers chronological rectangular-memory
composition, arbitrary-state serial transitions, bounded aligned writes,
six-concept solve, fixed-key amplification and bounded log positions. Its tests
are constructed mathematical/engineering checks only.

Recorded validation: [PNDS CI](https://github.com/FPC-effortless/pnds-lean/actions/runs/41107033496)
compiled this operator module and inspected all nine theorem axiom reports on
Lean 4.14.0. Only standard Mathlib axioms were reported. The implementation and
companion commits are pinned in `PNDS_LINK.json`; this does not certify the
whole model. CPU source-forward/parameter-gradient fixtures and a historical
trained-checkpoint numerical probe also passed; see `VALIDATION.json`.

The following remain **OPEN** until recorded execution and review:

- Matrix-level formal scan/state norm proofs and the full typed composed
  observation-to-logit graph in Lean.
- Actual-source forward, state, mask, input/parameter gradient and precision
  correspondence for every registered candidate and shape.
- Real-checkpoint causal persistence after source removal, transfer/composition,
  verified future utility, and complete acquisition/retrieval/verification costs.

No operator proof, passed fixture or output dependence upgrades these last
scientific obligations into PASS. Existing real-data and fair-comparison gates
remain binding for a subsequent trained study.
