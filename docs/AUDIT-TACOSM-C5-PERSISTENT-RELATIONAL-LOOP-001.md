# Scientific Audit — TACOSM-C5-PERSISTENT-RELATIONAL-LOOP-001

## Final historical disposition

**CONDITIONAL / REUSABLE WITH BOUNDS**

The authoritative run is real and reproducible at the artifact/provenance level, but the reported P90/K90 columns require a statistical qualification: the published rows are means of five seed-specific quantiles, not quantiles computed over the pooled 320 evaluation trials. The experiment therefore supports the bounded mechanism conclusions below but those P90/K90 values must not be described as pooled-trial 90th percentiles.

## Benchmark validity

PASS within the registered synthetic scope.

The benchmark requires a target descriptor generated from two persisted operands and a relation. The target descriptor is not directly stored in persistent state. Candidate populations contain exactly one target and otherwise random 16-bit descriptors. The task includes XOR, XNOR, AND, and OR, and the target is compositionally derived.

The benchmark is not a language benchmark and does not license semantic-memory or general-intelligence claims.

The exhaustive relation oracle is a valid capability ceiling for this synthetic task because it deterministically derives the registered relation from the state contents.

## Leakage audit

PASS for the principal dense-routing question.

Inspected boundaries:
- query text carries an opaque address rather than the target descriptor;
- query context contains a corrupted operation hint;
- persistent state stores operands plus operation, not target;
- target identity is stored in the evaluator trial object for scoring, not router features;
- persistent router receives only its state/query feature map and candidate descriptors;
- environment success is evaluated after candidate selection;
- reset clears the state before routing;
- calibration and held-out trials use separate deterministic offsets;
- sparse verification derives the reference from persistent state.

No direct target-descriptor path into routing was found.

## Temporal audit

PASS for the bounded persistence mechanism.

The state write has a registered three-boundary availability delay. Three unrelated writes are inserted across those boundaries. The evaluation reads the addressed record after the delay. The reported boundary-read and persistent reconstruction controls are 100%, while reset reconstruction is 0%.

This supports persistence in the registered synthetic state control, not indefinite or semantic long-horizon memory.

## Instrument audit

PASS with one interpretation restriction.

The benchmark has an exact persistent-composition oracle, reset control, operand-Hamming reference, unique-target construction, deterministic seed paths, and an explicit verifier path.

The learned router is trained exhaustively against outcome-derived target indices. This makes the result evidence for learning a synthetic relational mapping from persistent state, but not evidence of autonomous target discovery: the supervision supplies the correct candidate index after each action.

The sparse funnel is downstream of that learned relation. Its degradation must be separated into dense ranking, LSH admission, and bounded execution/repair, as the result report does.

## Statistical audit

**QUALIFIED**, not failed.

The published dense rows at each M use mean(seed-specific P90 rank) and mean(seed-specific K90). That quantity is not equivalent to a P90 over all 5×64 held-out trials.

No raw per-trial rank artifact was retained in the authoritative JSON, so the exact pooled-trial P90 cannot be reconstructed from the artifact alone. Therefore the values remain valid descriptive summaries of the registered per-seed analysis but must retain that label.

Top-1 values are averages of seed-level trial proportions and can be reported as seed-averaged Top-1. They are not evidence for a p-value because the artifact does not retain the paired per-trial observations needed for a paired test.

The finite-range power-law fit gamma=1.0650 is correctly treated as descriptive; its unstable local exponents preclude an asymptotic complexity inference.

## Provenance audit

PASS at artifact level.

Verified:
- workflow run 36845079770;
- successful job 110313168397;
- source commit bae076eb6d06b2b6abda6261b6014858b549d604;
- artifact 11152968667;
- artifact SHA-256 e3ce62436294d43e702dfdcd76faa03b42753579f27fcef160e08bc98771b563;
- artifact downloaded and hash-checked independently.

## Scientific disposition

The following bounded conclusions are supported:

1. Persistent state is load-bearing on the registered synthetic relational workload. At M=64..1024, persistent Top-1 is 77.81%, 68.13%, 63.44%, 58.44%, 47.50%, while reset Top-1 is 13.13%, 7.19%, 5.63%, 4.38%, 1.88%. The gap remains large across the tested populations.

2. The task is not solved by direct Hamming proximity to either persisted operand. The operand-Hamming reference remains below the learned persistent router at every registered M.

3. The main unresolved bottleneck is compositional routing under population growth. The dense router's P90 rank rises substantially by M=1024, and the sparse funnel adds additional admission and bounded-execution loss.

4. This is not C5 core scaling evidence. The experiment does not establish that useful computation scales with |R| rather than total population/history because it does not isolate total routing cost from relevant execution cost as a capability-preserving scaling law.

## Reuse rule

Future experiments may inherit the bounded persistence mechanism, anti-target-descriptor query boundary, reset intervention, operand-Hamming reference, and synthetic compositional workload generator.

They must not inherit the numerical P90/K90 as pooled-trial quantiles, the gamma fit as an asymptotic law, the Top-1 gap as a general memory/intelligence result, or the online endpoint as causal learning evidence.

## Required successor

The next composition-representation experiment should preserve the successful persistence boundary and intervention controls, retain per-trial raw ranks, and compare representation/operator variants under matched training and evaluation. Its primary endpoint should be a pre-registered seed-level statistic with raw per-trial artifact retention so both pooled and hierarchical uncertainty can be audited.

## Statistical sensitivity bound from published marginals

The published persistent/reset Top-1 rates correspond to 249/320 persistent successes and 42/320 reset successes. Because the reset arm uses the same deterministic held-out episode stream, a paired test is the conceptually correct comparison, but the artifact does not retain the 320 paired binary outcomes.

A worst-case paired sensitivity bound can nevertheless be derived from the marginals alone. The largest possible discordant set consistent with the two marginals is b=249 (persistent success/reset failure) and c=42 (persistent failure/reset success), giving b+c=291 and b-c=207. Under McNemar's exact null, the two-sided binomial tail is approximately 5.83e-37. Every other feasible pairing has no larger p-value because it has no larger discordant set with the same observed difference.

This is a sensitivity bound, not the primary reported inferential statistic. The raw paired outcomes should be retained in all future experiments so the exact paired analysis is directly auditable.
