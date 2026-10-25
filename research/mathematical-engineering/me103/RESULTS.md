# ME103 — completed equal continuation, no full-recipe qualification

All15 original ME102 models completed exactly1,536additional guarded updates;
each native AdamW state advanced1536→3072. These are continued models, not15new
independent seeds. Source/registration freezefb50159b and mathematical-link
freeze330526b6 preceded every update; all15original child model/optimizer/RNG and
runtime receipts were Git-frozen at1a728fdb BEFORE any REPORT evaluation.

## Original exposed-development scores

Five-seed means, six-hour forecasts, full256hour histories. Lower errors/scores
are better. Coverage is nominal90%, registered90–98%per seed.

| Recipe | ME102 gap PM2.5 MAE | ME103 gap PM2.5 MAE | ME103 gap coverage | ME103 gap NLL | ME102→ME103 gap interval score | Full-gate seeds old→new |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| light_floor |38.7814|38.2721|82.80%|5.1915|229.2109→224.0540|2/5→0/5|
| strong_reconstruction |39.7432|37.5242|95.66%|5.1510|316.7059→379.6232|2/5→1/5|
| physical_score |44.1222|41.8294|93.29%|5.1954|221.6744→225.6172|1/5→2/5|

Mean gap point error falls1.31%,5.58%,5.20%respectively versus each recipe's exact
ME102parents. Ordinary PM2.5 MAE falls from34.3075/33.8963/33.3507 to
33.1687/33.4031/32.5165. All15pass the ordinary point guardrail after continuation.
Gap density NLL worsens in every recipe versus ME102, although every model still
passes the historical ME101mean6.2345 density threshold.

**Every recipe FAILS its all-five-seed full gate.** Only3of15individual models
pass all five criteria (0light/1strong/2physical), versus5of15in ME102. Light-floor
coverage fails in every seed. Strong reconstruction preserves coverage but fails
the interval-score guardrail in4of5seeds; physical-score gap point error fails in
3of5seeds. These are genuine trade-offs, not an overall qualification or a reason
to select favorable seeds. All original gates and failed results are unchanged.

The strong recipe's mean gapMAE is slightly below historical ME101's37.7470, but
its intervals are much broader and worse-scoring. The physical recipe improves
ordinary point error below historical ME101's32.7709, but its gap point failures
remain. Different model sizes/data augmentation/training recipes prevent a
controlled architectural win over ME101. The valid within-lineage result is
point improvement with important uncertainty regressions after extra training.

## Denominators, uncertainty and reference integrity

Authentic UCI501/CC-BY-4.0 source, FIT2013–2014; exact already-exposed2015cohorts:
143natural-gap queries in21calendar-week resampling clusters and4,288ordinary
queries in53overlapping clusters. Six actual pollutant future labels per eligible
query, with natural missing labels masked. No synthetic primary labels, expanded
cohort, independent confirmation or protected2016–2017numerical access.

Paired posthoc week-cluster bootstrap of gapPM MAE(continued minus own ME102parent):
light−0.5093,95%interval[−2.8438,2.6478]; strong−2.2190,[−7.2933,2.8331];
physical−2.2928,[−4.0669,−0.2786]. These descriptive intervals condition on the
fitted seeds, are not multiplicity adjusted, and do not establish independent
superiority. Repeated seeds are not more independent real observations.
PANEL_SUMMARY.json includes all original per-channel/per-seed metrics, paired
changes, interval widths/tails and publisher-row label/eligibility audit.

## Training and mathematical scope

The98,495-parameter graph, objective recipes, optimizer/hyperparameters and input
features were unchanged. Same-seed new augmented-input/order checksums and
postfit ATen operator/shape/count/dense-FLOP signatures match across recipes.
Each model has24,576additional sampled examples and2,686,976history-event
exposures; samples repeat real training records. Dense forward/backward estimate
is1,483,140,983,296FLOPs/model; unsupported pointwise, optimizer and hardware FLOPs
remain unquantified. New training used857.674aggregate CPU seconds, two workers
maximum/one torch thread each; CPU time is an outcome, not a matching requirement.
Before/after deliberately doubles cumulative training exposure: not equal-compute
architectural superiority. All15models/attempts are preserved; no further fits.

All984local tests pass. Every fitted model passes independent serial-reference
FP32/FP64 output, state, read, chunk and gradient checks under the original tensor
tolerances; all30dtype checks are recorded in FITTED_GRAPH_AUDIT.json. The PNDS
ME102exact-real scalar kernels/Python graph are inherited unchanged; PNDS PR16
records the source-pinned continuation, and its Lean build passed. Whole-neural
and IEEE refinement, useful memory attribution, independent generalization,
embedded-hardware latency, action/customer utility and safe autonomy remain OPEN.

## Decision and next capability target

Do not extend these runs again, change gates or promote a recipe. More updates
help point fitting but do not resolve the combined accuracy/sharpness/coverage
problem. The next integrated architecture should preserve a strong point forecast
while learning a coherent, source/channel-conditioned predictive distribution;
that requires a separate source-pinned prospective protocol, not another silent
budget extension or an attribution campaign. Protected cohorts remain closed.
