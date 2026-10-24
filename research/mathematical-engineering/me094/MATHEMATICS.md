# ME094 — explicit median, multimodal shape and causal value retrieval

ME093 full gateFAIL is preserved: H1/H4/H16MAE0.05053248/0.13904939/0.59860214,
quietH1MAE0.02078656; unchanged quiet limit0.00187161 and stricter paired CI fail.
No extra ME093 fits. This is ONE new integrated architecture/optimization recipe.

## Input → state → parallel readout

Same real120FIT/40exposedREPORT, pinned licensed numeric observations and exact
Git-owned overhead pixels. Same known-at inputs: observed qpos/capture elapsed/
images through t and recorded reference through t-1. No future or protected data.
Native fast/slow/8x8associative dynamics, shared CNN and numeric encoders, local
causal filters and1331scalar explicit state are retained from ALL5indexedME093.
Only independent batch/time encoding and horizon/coordinate/value retrieval are
parallel; the actual nonlinear native recurrence retains its temporal dependency.

One shared SwiGLU hidden vector feeds a fused183x108 projection:99retained native,
42retained noise/shape and42new decision/retrieval fields. Total101,173active
parameters=96,595retained+4,578new. No padding or frozen learned modules.

## Relevance-conditioned observed-value retrieval

For coordinate j and horizon h, let r3 be the last-three observed-value median,
q_l the existing eight-value causal bank and age_l its observed elapsed age.
With FIT-only band c=sqrt(noise*delta_unit)>0 and learned time rate tau,

w_l=softmax_l(-age_l/(.25+softplus(tau)) - log(1+((q_l-r3)/c)^2)).

r=sum_l w_l q_l; r*=r3+sigmoid(zeta)(r-r3).

The retained local blend produces m_quiet=x+(1-sigmoid(blend))(r*-x).
This is a convex combination of actual observed values, not reconstructed future
information or an oracle. Content and time relevance are conditioned on the
shared learned query/state. The retained native physical expert yields m_dyn.
New independent point gate g gives m=m_quiet+sigmoid(g)(m_dyn-m_quiet).

## A learned median with a coherent multimodal distribution

The retained proper three-Laplace base distribution B uses hold/local/dynamic
locations and ordered scales. Its probability weights and scale parameters do
not directly define the new point forecast m. Let q=B^{-1}(.5). Translate all
base component locations by m-q:

P(y)=B(y-m+q), hence P(m)=B(q)=.5.

All components, probability mass, multimodality and relative shape are preserved;
the distribution is normalized and its median is m. Intervals and likelihood are
from THIS translated distribution. This is not an uncalibrated separate point
estimate paired with unrelated intervals. Physical expert parameters are shared
with m, so this does not claim all point/shape optimization is independent.

The centering quantile is differentiated with the implicit CDF derivative
q_theta=-B_theta(q)/b(q). Detaching q's parameter derivative would be wrong.
Implementation retains this gradient while solving the root numerically in
float64. The exact explicit m is returned for the median; remaining quantiles
invert the translated CDF at .05/.95 under the unchanged numerical contract.
Independent tests check CDF(m)=.5, zero median derivative with respect to shape-only
weights/scales, unit derivative to m, and likelihood gradients against finite
perturbations with the entire shape recentered each time. Numerical finite roots
are subject to original tolerances, not symbolic exactness at arbitrary floats.

## Complete objective and execution

CRPS+0.2physicalNLL+20medianMAE+0.2factual global motion BCE; median weighting is
increased for BOTH quiet and moving examples, not a quiet-only patch. Same proper
importance correction, quietH1/movingH1/long task partition and deterministic
rotating PCGrad. NLL remains proper and trains shape/point; no optimizer guarantee.
New decision fields start atzero weights, pointgatebias-2, timebias7.75,retrieval0.
All parameters trainable; retainedlr.0001/newlr.0003, AdamWwd.0001,clip1.

4,096new updates perseed. Pipeline17,408updates, retained native/visual own-lineage
13,312each,dynamic-location9,216,retained local-noise7,168,new4,578values4,096.
All previous parent work/failed recipes remains charged. Final-only selection,
fiveindexedparents, no bestseed/checkpoint/ensemble/retry. All5GitbeforeREPORT.

Training still decodes dense prefix targets. Scoring uses verified final-only
decode: compute necessary native states but avoid discarded prefix forecasts and
CDF centering roots. Full-prefix last output must agree with final-only, chunk and
stream execution. CNN/projections/value retrieval/horizon operations reuse shared
tensors. The pure physical boundary and exact-retry versioned entity cache retain
their explicit state/input contracts. No silent stale-feature reuse across updates.

Every unchanged predictive gate remains, plus original95%and additional99.61538%
paired interval for13declared recipes and strict independent original replay.
This is adaptive exposed development, not an independent architecture-winner or
real action-policy/customer/general-intelligence claim. Direct horizon marginals
remain distinct from a joint future trajectory or a closed-loop simulator.
