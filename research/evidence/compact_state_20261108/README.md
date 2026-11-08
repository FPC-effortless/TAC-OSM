# Hosted compact-state evidence

Both original hosted ZIPs are preserved losslessly, including all timing samples,
93-test JUnit receipts, CPU plan and six-row source-pinned default-model results.
No trained weights or dataset are present. Run `python verify_saved.py` from this
directory to verify archive hashes/coverage/decisions without model execution.

- [Initial run 42457059397](https://github.com/FPC-effortless/TAC-OSM/actions/runs/42457059397), artifact13607463675, source62d01da8.
- [Reference-complete run 42457145307](https://github.com/FPC-effortless/TAC-OSM/actions/runs/42457145307), artifact13608460999, sourcec719ab06.
- [Reference-head compulsory preflight 42457145228](https://github.com/FPC-effortless/TAC-OSM/actions/runs/42457145228): PASS.

Both source runs passed all93selected checks and all six exit/prefix rows.
The second source includes the independent NumPy value-copy/adjoint reference;
model/helper source and tolerances are unchanged. Same-partition predictions
and carried state are byte-identical with/without the copy. Full weight hashes
match before/after on both runners; originals and all samples are retained.

At exit20, T64 backing storage drops46,391,296→1,171,456bytes, and
T128 drops50,716,672→1,558,528bytes. The initial hosted copy medians were
0.531/0.558ms, versus initial local0.339/0.362ms. Independent repetition samples
are in REFERENCE_RESULTS.json; do not pool hosts/repetitions into a quality gain.

State backing bytes exclude weights, allocator reservations, AD and runtime;
they are not total or peak RAM. Copy has time/transient allocation costs and old
aliases/attached graphs can retain storage. The API never detaches automatically.
The full-size object is untrained; no optimizer/corpus/TEST/GPU was used.

Internal engineering fixtures only, not independently qualified. Learned memory
utility, full PNDS IEEE/AD/allocation refinement, fresh Lean build and physical
device p50/p95/RAM/energy/airplane-mode remain OPEN/NOT RUN. Original negative
evidence is untouched. Review before adopting; later precision-state classes
are unsupported until source-reviewed. No checkpoint/model was promoted.

After the measured source head, CI runs ordinary tests plus this saved-evidence
checker; the full default resource experiment is frozen to c719ab06 or explicit
manual dispatch. A later source change needs its own protocol review/resource
receipt; skipped full-model execution is never evidence of a new head passing it.
