# TACOSM-FUSED-CDL-CASM-001 Results

## Completed run

- Branch: research/fused-cdl-casm-full-loop-001
- Final implementation commit: 3099feb81e56dc74cc13d5834ec6d6a2e501232f
- CI run: 36826606939
- Unit tests: 5 passed
- Random seeds: 10
- Held-out evaluation: 900 steps

## Pooled result

| Measure | Result |
|---|---:|
| Fused CDL + CASM top-1 success | 0.2467 |
| CDL target admission recall at K=4 | 0.5733 |
| Mean target rank | 4.0844 |
| Mean CASM active nodes | 7.0 |
| Mean CASM candidate edges | 6.0 |
| Mean state slots inspected | 0.6667 |
| Mean state pool size | 41.30 |
| Mean verifier-derived student updates | 213.1 |
| Mean verified experience writes | 33.1 |
| Static routing control | 0.0356 |
| Oracle routing control | 1.0000 |
| Reset-state fused control | 0.2178 |

For an eight-candidate task, random top-1 chance is 0.125. Random K=4
admission chance is 0.5.

Across the ten seeds, top-1 mean was 0.2467 with population standard
deviation 0.0354; admission recall mean was 0.5733 with population standard
deviation 0.0505.

## Family breakdown

| Family | Top-1 | K=4 admission | Mean target rank |
|---|---:|---:|---:|
| relational | 0.2600 | 0.5300 | 4.250 |
| state_lookup | 0.2700 | 0.6333 | 3.723 |
| replay | 0.2100 | 0.5567 | 4.280 |

## What passed

1. The complete causal traversal executes without bypassing the existing loop
   boundary: state addressing, routing, computation selection, outcome,
   verification, learning, and verified state writing all occur in order.

2. The explicit CASM selector is exercised on every routed action. Exact CASM
   execution and the environment verifier agree; the oracle arm reaches 1.0.

3. The CDL student can learn from post-execution verifier feedback, including
   lookup/replay failures where the hidden gold is only exposed after the
   action.

4. Verified learning writes no longer overwrite environment-owned addressed
   world facts. Experience is written under its own namespace.

5. The route boundary remains target-free. The hidden target is used only by
   the post-hoc verifier/learning path.

## What failed

The fused system does not currently produce strong final selection.

K=4 admission recall is only 0.5733, which is modestly above the eight-candidate
chance level of 0.5. More importantly, top-1 success is 0.2467 and mean target
rank is 4.08. Therefore the current learned CDL student is not a reliable
decision rule even when it sometimes puts the target into the short list.

This separates two problems:

    information admission
        target enters a small candidate set

    computation/action selection
        choose the correct candidate from that set

The first is weak but non-random in this benchmark. The second is the larger
current end-to-end bottleneck.

The persistent-versus-reset comparison also does not show a large state effect
yet (0.2467 versus 0.2178). That should not be interpreted as evidence against
persistence in general: this environment is only a short-horizon mixed task
stream, and the current router is weak enough that state benefits can be masked.

## C5 interpretation

C5 is still not established.

The fused runtime CDL student scores the full candidate set, so the measured
admission fraction is not an end-to-end compute reduction. CASM execution is
bounded for the selected action, but candidate scoring still grows with the
candidate population.

The correct reading is:

    CDL -> candidate admission is partially functional
    CASM -> exact selected computation is functional
    verifier -> post-hoc learning signal is functional
    full fused decision quality -> insufficient

## Next research phase

The result points to a fixed-K second-stage selector:

    full candidate set
          |
          v
       CDL student
          |
          v
      Top-K admission
          |
          v
       CASM on K
          |
          v
    verifier / action score
          |
          v
        action

The next experiment should hold K fixed while increasing candidate population
M. It should measure:

1. admission recall at fixed K;
2. final top-1 success after CASM evaluates only admitted candidates;
3. absolute CASM execution work as M grows;
4. total routing work separately;
5. capability retention against the oracle ceiling.

This is the cleanest bridge from the current fused loop toward the actual C5
question without conflating candidate scoring cost with executed computation
cost.

## Non-claims

This result does not establish sublinear end-to-end compute, hardware FLOP
reduction, semantic open-world retrieval, learned CASM gating, or the C5 claim
that executed computation is a function only of the routed subset.
