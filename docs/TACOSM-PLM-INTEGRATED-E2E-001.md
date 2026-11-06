# TACOSM-PLM-INTEGRATED-E2E-001

## Research position

This phase moves the main PLM research object from a collection of isolated mechanisms to a jointly trained computational chain:

```
multimodal evidence
 -> shared representation
 -> persistent state write
 -> state addressing
 -> computation/operator selection
 -> action
 -> environment outcome
 -> verification
 -> verified state update
```

The experiment is deliberately small. Its purpose is to establish whether the integrated interfaces can be trained jointly before scaling the system or replacing synthetic modalities with large pretrained encoders.

## Multimodal design

Each observation contains three synchronized views of the same latent binary state:

- text: token sequence encoding entity identity and latent bits;
- image: spatial grid encoding entity identity and latent bits;
- audio: temporal waveform encoding entity identity and latent bits.

The model has separate modality encoders, but they feed a common latent representation. No modality is converted into a text caption during the benchmark.

At query time the raw observation is absent. The query contains only:

- entity identifier;
- two latent-bit indices;
- an operator identity (XOR, AND, OR, XNOR).

q1 and q2 target different observed entities. The model must retrieve the persistent state for each target entity and execute the requested operation; a verified q1 outcome therefore cannot directly provide q2's target state.

## End-to-end trainable chain

```
E_text,E_image,E_audio
        |
        v
   shared z_t
        |
        v
persistent write
        |
        v
query q_t -> state read R_t
        |
        v
operator router C_t
        |
        v
differentiable CASM operator
        |
        v
action a_t
        |
        v
environment outcome o_t
        |
        v
verifier V_t
        |
        v
write gate -> S_(t+1)
```

The first implementation uses exact differentiable Boolean operator forms for the CASM execution primitives. This isolates the question of whether the learned chain can route the correct state and operator before adding a learned arbitrary operator executor.

## Training signals

The primary action losses are applied to both queries.

Auxiliary losses are used only during training:

1. observation entity identification;
2. latent-bit reconstruction from each modality's shared representation;
3. operator-selection cross-entropy.

These are explicit scaffolds. A later ablation will remove them one at a time.

## Benchmark separation

The training query set excludes eight exact operator/index compositions. Test episodes use only those held-out compositions. Entity IDs, latent bit patterns, and modality encodings are independently resampled.

The primary metric is held-out q2 action accuracy with all modalities present.

No evaluation outcome is visible to the representation writer, router, or optimizer during the episode.

## Controls

### No-memory

Persistent values are zeroed immediately before the query. This tests whether performance depends on information carried through the persistent substrate rather than on the query alone.

### Cross-modal misalignment

Image observations are permuted across entities while text and audio remain aligned. This tests whether the fused representation is exploiting cross-modal agreement rather than one modality alone.

### Unimodal

Two modalities are removed. This is a diagnostic, not the primary endpoint.

### Held-out composition

The exact operator/index tuples used at evaluation are excluded from training, including their reversed index order. Because the Boolean operators are commutative, only canonical i<j pairs are admitted to the benchmark.

## Scientific decision rules

- Primary success requires mean held-out q2 accuracy >= 0.80 across the five registered seeds and no seed-level failure rate above 0.40.
- The no-memory result is interpreted only as a causal control on persistence.
- A large shuffled-modality drop is interpreted as evidence that cross-modal alignment contributes to the learned representation; it is not sufficient to prove semantic multimodal reasoning.
- All raw seed results are reported. No seed or control is dropped because it is unfavorable.
- No model selection is performed using the held-out query tuples.

## Claim boundary

A positive result establishes only that a small, jointly trained synthetic multimodal persistent computation chain can learn the registered task. It does not establish language understanding, image understanding, audio understanding, AGI, open-world world modelling, or asymptotic efficiency.

