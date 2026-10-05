# G-CASM-017A — Exact Evidence Index

**Status: PREREGISTERED / IMPLEMENTATION UNDER AUDIT**

## Scientific purpose

The synchronized G-CASM-015 branch defines the activation-trace observation channel. 017A does not assume a confirmatory 015 result; it uses that channel implementation as a fixed parent instrument.

The remaining problem was computational: exhaustive probe selection scans every
candidate for every possible row, and trace evidence makes that scan six times
wider than the scalar channel.

G-CASM-017A changes only this candidate-side computation.

## Intervention

trace_exhaustive reproduces the G-CASM-015 action selector.

trace_indexed stores exact candidate evidence signatures in per-row posting
bitmaps and evaluates action partitions by bitmap intersection and popcount.

The indexed selector must reproduce:

- the selected probe row;
- information gain;
- expected remaining candidate count;
- information gain per expected environment work.

Any divergence is an infrastructure failure, not scientific evidence.

## Cost accounting

Both arms pay the same trace candidate-cache work.

The exhaustive arm additionally pays:

    M * 16 * 6

candidate prediction units per selection.

The indexed arm additionally pays one exact index-build cost amortized over the
registered 160 tasks per seed, then pays:

    sum_actions(signature_buckets(action) * ceil(M / 64))

bitmap-word units for each action scan.

Environment acquisition cost is unchanged from G-CASM-015.

The primary endpoint uses full accounted work:

    information gain /
    (cache amortization + selector work + environment acquisition)

This is deliberately more demanding than the environment-only metric from
015.

## Leakage

The index contains only public candidate-predicted evidence. The target index
and target identity are never used by the selector. The realized target trace
is only read after the row has been selected.

## Interpretation

A positive primary result would show that structured evidence can be made
computationally addressable without requiring learned routing.

A negative primary result with exact selector equivalence would mean the
remaining computational cost is not solved by this exact index structure, but
would not invalidate structured evidence itself.

A non-equivalent index invalidates the run because the intervention would no
longer be a pure computational replacement.

## Scope

No learned probe policy, asymptotic scaling, hardware speedup, language,
vision, audio, or generic active-learning claim is licensed.


Confirmatory trigger record: [run-graph-casm-evidence-index-017-full] after successful same-run smoke.


Final confirmatory trigger record: [run-graph-casm-evidence-index-017-full] on synchronized test-count head.
