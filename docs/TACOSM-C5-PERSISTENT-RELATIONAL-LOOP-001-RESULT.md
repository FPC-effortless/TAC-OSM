# TACOSM-C5-PERSISTENT-RELATIONAL-LOOP-001

**Status: REGISTERED**

## Research correction

The preceding admission/scaling audit showed two remaining limitations:

1. dense rank growth becomes close to linear at high M;
2. the one-bit binary task is exactly solvable by a Hamming/codebook shortcut and therefore does not establish learned semantic relevance.

The audit also showed that the main noisy funnel had no persistent-state input.

This phase removes both limitations while preserving the C5 execution funnel.

## Main task

At the start of each episode, the environment samples:

- left operand: 16 random bits;
- right operand: 16 random bits;
- one operation from XOR, XNOR, AND, OR.

The target candidate is the result of R(left, right, operation).

Only the operands and operation are written to persistent state. The target descriptor is not stored.

The write becomes readable only after 3 temporal decision boundaries, with unrelated decoy writes interleaved during those boundaries.

The public query contains only an opaque state address and a one-bit-corrupted operation hint. It does not contain the target descriptor.

Candidate populations contain the target plus random unique descriptors.

Thus the relevance relation is a derived composition of state, not direct Hamming similarity to an exposed target.

## Main path

```
persisted operands + noisy public hint
              ↓
         CDL Q/K router
              ↓
          OR-LSH admission
              ↓
        conformal K shortlist
              ↓
         CASM execution
              ↓
       environment outcome
              ↓
          verifier
              ↓
       bounded repair
              ↓
     verified online update
```

Persistent state is therefore inside the actual routing/execution path rather than a separate upper-bound control.

## Controls

### Reset

The same episode is evaluated after persistent state is cleared. The router receives the address but no readable operands/operation.

### Operand-Hamming reference

Candidates are ranked by minimum Hamming distance to either persisted operand. Because the target is a compositional result, this is not the task's defining relevance relation.

### Persistent composition oracle

The exact relation is computed from the persisted operands and operation without learned routing. This is an upper-bound control and not a learned result.

## Primary question

Does a state-aware learned CDL representation preserve enough of a compositional persistent relation to feed the sparse C5 execution loop?

The primary learned endpoint is dense persistent-state Top-1 routing. Dense P90 rank, reset routing, sparse admission, execution, and online verified learning are secondary diagnostics.

## Anti-leakage requirements

The target descriptor must never enter:

- persistent-state values;
- query text;
- query context as a target encoding;
- router inputs;
- verifier inputs as hidden metadata.

The verifier may derive the expected candidate descriptor from the public state relation after execution, because the target is a function of the persisted world/task state rather than a hidden answer stored for the verifier.

## Temporal requirement

The state record is written at the beginning of the episode and becomes available only after three boundaries. Three unrelated state writes occur during that interval.

The experiment therefore tests finite-horizon persistence with interference, not arbitrary long-term memory.

## Sparse measurement

For each M:

1. fit K90 on a calibration stream;
2. measure p1/p2 on calibration;
3. derive the OR-LSH table count from rho;
4. evaluate admission and final verified success on disjoint held-out episodes;
5. report the entire registered M series.

No held-out outcome is used to choose an operating point.

## Scope

This is a synthetic relational persistence benchmark. It is intentionally not a language-level semantic benchmark.

A positive result would establish that the integrated state-aware funnel can learn a compositional routing relation whose answer is absent from the public query and only derivable from temporally persistent state.

It would not establish general intelligence, language reasoning, or indefinite memory.
