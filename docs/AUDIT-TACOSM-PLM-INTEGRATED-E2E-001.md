# Audit — TACOSM-PLM-INTEGRATED-E2E-001

## Pre-measurement disposition

**Status: READY WITH REGISTERED AMENDMENTS**

The experiment is a clean one-experiment boundary from
`master@eadc884341b7ca42d52b2965eddfabb428eb3621`.

### Benchmark validity

The payload is sampled independently for every episode and entity, so entity
identity and query indices do not determine the target bits. The evaluation
query compositions are held out exactly, and because the Boolean operators are
commutative, both index orders are excluded by construction.

The final benchmark uses 12 latent bits distributed across modalities:
text 0–3, image 4–7, audio 8–11. Every primary held-out query combines two
different modalities. No single modality therefore contains both queried bits.

The trivial query-only operator prior is explicitly charged. For uniformly
sampled independent bits, XOR and XNOR have Bayes accuracy 0.50, while AND and
OR have Bayes accuracy 0.75. The registered eight held-out compositions contain
two of each operator, giving a theoretical operator-prior baseline of 0.625.

### Leakage/data-flow

Before action, the model receives only:
entity identifier, two bit indices, and operator identity.

The target answer is not an argument to `model.query`. It enters only through
the environment outcome passed to the post-action verifier and update path.

The evaluation generator is created only after all training steps. Held-out
query compositions are never used for training or model selection.

q1 and q2 target different observed entities. Therefore q1's verified outcome
cannot directly equal q2's target state.

### Protocol

Five seeds, fixed 300 training steps, fixed batch size 96, fixed model size,
fixed optimizer settings, fixed held-out composition set.

Controls are evaluated on exactly the same held-out episodes within each seed:
normal, no-memory, shuffled-image, text-only, image-only and audio-only.

### Interface / implementation

The end-to-end model has trainable parameters in all major chain surfaces:
text encoder, image encoder, audio encoder, shared representation, state
write, state read, CASM selector, verifier, and write gate.

The CASM Boolean execution primitive is exact and differentiable. This is a
deliberate execution scaffold. It does not establish learned arbitrary
operator synthesis.

### Statistical interpretation

The primary endpoint is the mean q2 accuracy with all modalities and persistent
state active. All five seed values are retained. No seed is removed for
unfavorable behavior.

The mean is a descriptive aggregation; seed-level values are the unit for
between-seed uncertainty. This phase does not authorize scaling-law claims.

### Remaining scientific limits

- synthetic modalities, not natural language/audio/images;
- within-episode persistence, not lifelong cross-episode persistence;
- entity-slot addressing, not autonomous semantic memory search;
- exact Boolean CASM, not a learned arbitrary operator population;
- auxiliary representation/operator losses remain training scaffolds;
- no hardware or asymptotic efficiency result.

No confirmatory result is contained in this audit.
