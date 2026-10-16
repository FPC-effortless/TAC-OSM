# DEV-020 mechanism hypotheses from a code-level read (NOT results)

This note describes **testable candidate mechanisms**, not empirical conclusions about the trained model. Do not substitute it for the frozen feature-level probe measurements or weaken prior failed gates.

## H1: Long-horizon write-by-write decay and interference

The [actual native `DeltaMemory` writer](../src/tac_osm/development/noisy_memory.py) computes:

```text
proposal = decay(joint) * state + assignment(address) * (write - erase)
decay(joint) = sigmoid(linear(address, public-value-onehot))
```

The decay linear layer is **initialized with zero weights and bias +5**, so its untrained multiplicative factor per write is sigmoid(5) ≈ **0.993307**. If it stayed fixed and ignoring subsequent erase and write interference, a memory trace preserved through **503 intervening events** would retain only `sigmoid(5)^503 ≈ 0.0341` (3.41%) of its initial magnitude. This is an *initialization calculation only*. Actual post-training decay, erase, write, assignment and key interactions can dramatically change retention; the formula alone does **not** demonstrate that trained memory is erased.

**Probe:** compare paired causal-state differences at prefix H32, H128 and H512; measure the trained decay factors and write-key interference on held-out disjoint-event histories **in a later preregistered experiment**. A transient decay is not necessarily loss of linearly accessible information, and pairwise L2 changes can reflect noncritical content.

## H2: Writer-key versus read-key misalignment

The native TAC writer learns `DeltaMemory.key(address)` inside the memory module, whereas the controlled DEV-018 predictor decodes using an independent `MemoryBaseline.read_key(query_embedding)`. Their key projections start separately and are not explicitly tied, and writing addresses and decoding queries have different inputs. End-to-end learning may align them, but the DEV-018 final output failure does not prove it did.

**Probe:** compare frozen latent state and exact query-conditioned read features on the same opposite-outcome pairs; test independent *diagnostic* readout probes. If the state probe generalizes and the native read features do not, preregister controlled key-sharing or aligned-address ablations in a successor experiment. Do not retroactively change DEV-018.

## H3: Explicit entity indexing versus learned content addressing

The strong schema41 comparator uses `entities` to scatter state updates into one of **eight fixed, public entity-addressed slots**. The native DeltaMemory call in the DEV-018 comparator updates by the encoded event address and public value and **does not use the `entities` argument directly**. The event encoder still receives public entity information, so the native architecture is not entity-blind. But it must *learn* isolation of overlapping entities, rather than having that isolation enforced structurally.

**Probe:** if H1/H2 do not resolve the failure, preregister an entity-indexed TAC variant with the same total state budget and matched reader/optimizer comparison against schema41, GRU and unchanged TAC. Sweep entity counts, clutter and intervening writes. This would test address isolation rather than prove novelty.

## Interpretation hierarchy

1. The DEV-018/019 output failure is observed.
2. DEV-020 frozen state/read/logit measurements determine where usable signals can be **diagnostically** recovered.
3. H1–H3 are *mechanistic hypotheses*: their relative importance remains unknown.
4. Architectural intervention (key tying, retention floor, entity-factorized write) should come **only after** a preregistered causal ablation, with both long-horizon accuracy and physical unnecessary-service cost.
