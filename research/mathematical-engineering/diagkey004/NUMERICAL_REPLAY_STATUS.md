# Independent replay and numeric exception

A separate clean shallow Git clone at ebdd7737 restored all25 original models,
50 milestone checkpoint+optimizer/RNG files and the actual guard receipts,
verified the frozen source/registration/schedules and recomputed all50 TEST sets.
It passed the ORIGINAL1e-5 abs/rel logit and1e-5 absolute NLL contracts, exact
labels and classification bits, without an optimizer update. Its original
receipt is `research/replay-bundles/DIAGKEY004/INDEPENDENT_REPLAY.json`.

The external GitHub runner in run39983850256 failed the SAME registered logit
contract. This failure is preserved, never reclassified as a pass. Training and
the local independent clone use AMD EPYC9V74, PyTorch2.14.1+cpu, NumPy2.3.5,
AVX512. The complete audit now shows exact predicted classes and NLL within the original
contract for all50 sets, but2/819,200 logits exceed the fixed limit. Both runtimes
report AVX512, so a simple advertised instruction-set difference is not a cause
that has been established. Host/backend cause remains unresolved. No weights, original logits, model source,
registration, test cases, threshold or tolerance have been changed.

A read-only audit is added to report all50 sets even after the strict replay
fails. It separately records exact classifications, NLL errors and every logit
outside the ORIGINAL tolerance. Its diagnostic completion cannot turn a strict
numerical failure into a qualified external replay. A later numerical contract
must be prospectively justified; no posthoc tolerance widening in this study.

The original synthetic recall gate and external numeric portability gate are
separate. The new candidate passes the former on the original saved results;
all-platform numerical qualification remains unresolved. No authentic-data
capability, architectural superiority, commercial reliability or long-retention
claim follows from either check. Keep draft/unmerged; no automatic curriculum.


## External audit recovered from run39985095618

`GITHUB_CROSS_CPU_AUDIT.json` contains the verbatim complete JSON printed by the
GitHub runner. Every original checkpoint/receipt was verified. No labels,
correctness bits or predicted classes changed across50x512 predictions. All NLL
scores pass their original1e-5 limit, maximum difference5.5928280368e-8. Exactly
two logit elements fail abs/rel1e-5; overall maximum abs error3.6239624023e-5.
The status is FAIL_REGISTERED_NUMERICS_PRESERVED, never a substituted success.
`LOCAL_AVX2_COMPATIBILITY_AUDIT.json` separately records a complete same-host
AVX2 pass. Neither a CPU label nor one successful host proves portability.
