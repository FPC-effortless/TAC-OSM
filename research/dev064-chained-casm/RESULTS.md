# DEV064 chained-dependency attribution preflight — NO_HEADROOM

Status: **PRELIMINARY DESIGN CLOSED; no architectural evaluation, no promotion.**
No confirmatory 20-seed fits or test-set inspection were conducted.

## Evidence

The remote smoke and five unit tests passed on the prior preflight commit
`e0e02d9baaf66d9d53f12738852c597188c562d8`, GitHub Actions
[run 39295844591](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39295844591).
All five arms trained for only **8 updates**, which is a software smoke, not
a scientifically interpretable architecture comparison. Those smoke outcomes
are not accepted model-quality estimates.

The **validation-only** dataset, namespace 640642 / 1,024 situations, gives:

- Exact public oracle minus one-step greedy expected-cost headroom
  (mean regret of greedy): **0.027931743662873525 cost units per situation**.
- Fraction of greedy decisions with nonzero exact-oracle regret:
  **0.08984375**.
- Validation input SHA-256:
  `99806f83943eb32da580bf6dc7fe18a53f964a60f165f59f4596801bd02766a9`.
- Validation target SHA-256:
  `287ce591f050436854b9ed9f9b5f481315675c70fb080ef252ce89bc798f5852`.
- Training input SHA-256:
  `0a52bb9e50f66fbf4a7473460198839b507280bcbd157dedebc893626d00c0e1`.
- Training target SHA-256:
  `f53fae6291d18d096c36fb32b06e52c47acc7582e2a514b6df1b21722ddeeb06`.

The headroom threshold in the protocol is **at least 0.05 cost units mean
greedy regret, plus at least 5% nonzero-regret choices**. The benchmark passes
the prevalence part, but **fails mean practical headroom**. Do not use this
benchmark to assert CASM or no-CASM superiority. Rerunning or widening the
holdout on the same task after inspecting this result would not be a
prospective confirmation.

**Timing caveat:** the numerical 0.05 gate was added after the initial smoke
generated these validation data but before its artifact was opened and read
in this review. It is therefore a transparent *pre-pilot* decision rule, not
an independently preregistered, before-any-data confirmatory threshold.
The larger architectural study must use a new benchmark, new source-frozen
protocol, independent seeds and an untouched evaluation before qualification.

## Parameter-matched repair

The original smoke disclosed 2,132–10,084 trainable parameters across arms
and was not eligible for matched-compute attribution. The corrected software
smoke used:

| Arm | Active trainable parameters |
|---|---:|
| Compositional TAC | 7,725 |
| No-CASM | 7,816 |
| GRU | 7,704 |
| Attention | 7,844 |
| MLP | 7,729 |

This closes the most obvious size mismatch for subsequent *new* studies, but
has no bearing on the failed oracle-headroom gate. Architectural prior,
operations, wall time, and supervision matching also remain open.

The code and reports are retained; test namespace 640643 remains unopened.
No DEV068 / REAL-MAINT-004 data or checkpoints were accessed, overwritten
or re-evaluated. The existing DEV067 / REAL003 scientific decisions remain
unchanged.

**Next:** only a new source-frozen chained-dependency task with stronger
validated oracle headroom and independent, retrained controls can support
a CASM-specific claim. A practical example would introduce prerequisite
setup, delayed collateral harm, shared resources, and actions that have no
immediate improvement but enable later repairs. Do not label a redesigned
study a passed rerun of this one.
