# TACOSM-PLM-INTEGRATED-E2E-006 — Result

## Status

**Measured; primary criterion passed.**

This is the first fresh confirmatory integrated multimodal PLM result in this lane that clears the preregistered synthetic capability threshold after the E2E-005 instrumentation correction and the capacity selection on a separate development split.

## Primary result

| Seed | q2 accuracy |
|---:|---:|
| 0 | 0.8725 |
| 1 | 0.8775 |
| 2 | 0.8500 |
| 3 | 0.8600 |
| 4 | 0.8725 |

Mean q2 accuracy: **0.8665**

Seed-level bootstrap 95% interval: **[0.8575, 0.8745]**

Minimum seed: **0.8500**

Registered primary threshold: **0.80**

Therefore **primary_pass = true**.

## End-to-end integrity

The measured model path is:

**text + image + audio → modality encoders → shared representation → explicit entity-addressed persistent state → query-conditioned fixed CASM → action → post-action environment outcome → verifier/outcome-gated update**

The following gates passed:

- classifier decision integrity: zero argmax(logits) vs action >= 0.5 mismatches across all seeds;
- gradient-surface gate: passed;
- training/evaluation semantic episode-key overlap: zero for every seed;
- held-out composition exclusion: passed;
- evaluation episodes generated after training: passed;
- control episodes reuse the exact same episode objects within each seed: passed;
- no payload-bit auxiliary supervision: confirmed false;
- q1/q2 entity targets are distinct;
- oracle q2 accuracy: 1.0.

## Fresh-test separation

E2E-006 permanently excludes three prior sets from training:

1. the sealed E2E-005 held-out compositions;
2. the registered 32-composition development set used for model selection;
3. the eight E2E-006 confirmatory held-out compositions.

The E2E-006 held-out set is:

- xor(1,5), xor(2,9)
- and(4,10), and(3,8)
- or(1,7), or(5,10)
- xnor(2,7), xnor(6,9)

All are cross-modal.

## Registered controls

Mean normal q2 accuracy was 0.8665. The corresponding mean control accuracies were:

| Control | Mean q2 |
|---|---:|
| no_memory | 0.6200 |
| shuffle_image | 0.7330 |
| text_only | 0.6170 |
| image_only | 0.6180 |
| audio_only | 0.5900 |

These are diagnostic controls, not separate primary endpoints.

Mean memory drop was **0.2465** and mean image-alignment drop was **0.1335**.

## Comparison to the development selection

The preregistered development comparison used a fixed 32-composition development set and excluded the sealed E2E-005 test set.

- hidden_dim=40: mean q2 **0.6950**
- hidden_dim=64: mean q2 **0.8650**
- paired mean gain: **+0.1700**
- hidden_dim=64 minimum seed: **0.7750**

This development result was used only to select the candidate configuration. E2E-005 held-out outcomes were not used for that selection.

## What is established

The evidence supports a bounded claim:

> The selected hidden_dim=64 integrated PLM configuration can carry synthetic information from text, image, and audio observations through persistent entity-addressed state into fixed query-conditioned computation and achieve 0.8665 mean accuracy on a fresh held-out cross-modal Boolean composition benchmark, under the registered no-auxiliary-payload-supervision and leakage controls.

This does **not** establish real-world language understanding, real-world image/audio understanding, semantic memory, learned operator discovery, scaling behavior, AGI, or hardware efficiency.

## Historical E2E-005 interpretation

E2E-003 and the original E2E-005-style results at ~0.52 are retained as historical artifacts. The earlier evaluator contained a probability-to-class-logit interface error that structurally forced the positive class. The corrected E2E-005 confirmatory rerun produced **0.7700 mean q2**, below the registered 0.80 threshold. It therefore remained a bounded negative/insufficient result rather than an architecture impossibility finding.

## Provenance

- Experiment: TACOSM-PLM-INTEGRATED-E2E-006
- Workflow run: 37507709403
- Git commit: d2e053ced7adc63bb61013526b070dae056344f2
- Contract SHA-256: d0294ab500b3364518245f67d28d98c1e2b17564204f573ac3dd6ea7b4210d1a
- Fresh benchmark SHA-256: 7f786299aa19cc3347cc1246b451ed672add7349e3591de7965b1d246db918b1
- Frozen E2E-005 base benchmark SHA-256: ca6147a3ad0cf503f5dadd51283482621e164080f2eb266777c4ae72748142c6
- Python: 3.13.15
- PyTorch: 2.14.1+cu130

Independent result validation completed successfully in CI before artifact upload.
