# TACOSM-AXON-STRUCTMEANS-SECA-002 RESULT

## Execution

Phase-2 runner: scripts/run_axon_structmeans_seca_002.py
The runner passed its scientific assertions in CI. The surrounding validation
PR initially contained stale test/provenance failures; those did not invalidate
the runner output.

## Results

- verified training transitions: 360
- held-out transitions: 180
- PST held-out transition accuracy: 1.0000
- REGM verified records retained: 360
- unverified rejected transition committed to durable store: false
- PST reconstructed from durable verified experience: 1.0000
- StructMeans clusters: 3
- StructMeans kind purity: 1.0000
- StructMeans exact-signature purity: 1.0000
- verified transitions per centroid: 120
- AXON fixed executable macros: 3
- AXON single-operator route success: 1.0000
- composite evaluation selected only cases where no primitive macro solved the target
- pre-SECA composite success: 0.0000
- independently verified SECA compositions: 6
- target composition discovered: true
- post-SECA composite success: 1.0000

## Interpretation

This is positive evidence for the structured-transition stack on the synthetic
workload:

verified experience -> PST -> structural abstraction -> AXON operator
consolidation -> bounded SSA retrieval -> SECA composition -> verification.

The important nontrivial result is the pre/post distinction: the composite target
was constructed so that no single primitive macro solved it; success appeared
only after SECA admitted the verified two-operator composition.

This does not establish universal reasoning, broad continual learning, or
large-scale sparse addressing.
